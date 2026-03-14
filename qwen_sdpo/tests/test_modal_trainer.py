"""
TLDR: Unit tests for gradient-step skip logic in qwen_sdpo.modal_trainer.

Tests _should_skip_gradient_step and _skip_iter_log_updates so that when proof is
truncated (or success/server_error), the trainer skips the gradient update and
returns the expected iter_log shape. Also tests _kl_loss_scalar (drop first N tokens).
No Modal or GPU required.
"""

import pytest
import torch

from qwen_sdpo.modal_trainer import (
    KL_DROP_FIRST_N_TOKENS,
    _kl_loss_scalar,
    _should_skip_gradient_step,
    _skip_iter_log_updates,
)


class TestShouldSkipGradientStep:
    """_should_skip_gradient_step(payload) must be True when we should not update weights."""

    def test_skip_when_is_truncated_true(self):
        """Truncated generation → skip gradient (no complete proof to learn from)."""
        payload = {"is_success": False, "is_server_error": False, "is_truncated": True}
        assert _should_skip_gradient_step(payload) is True

    def test_skip_when_is_success_true(self):
        """Proof verified → skip gradient (no failure to learn from)."""
        payload = {"is_success": True, "is_server_error": False, "is_truncated": False}
        assert _should_skip_gradient_step(payload) is True

    def test_skip_when_is_server_error_true(self):
        """Kimina unavailable → skip gradient (feedback unreliable)."""
        payload = {"is_success": False, "is_server_error": True, "is_truncated": False}
        assert _should_skip_gradient_step(payload) is True

    def test_do_not_skip_when_all_false(self):
        """Normal failure (non-truncated, not success, not server error) → do gradient step."""
        payload = {"is_success": False, "is_server_error": False, "is_truncated": False}
        assert _should_skip_gradient_step(payload) is False

    def test_do_not_skip_when_truncation_key_missing(self):
        """Payload without is_truncated key is treated as not truncated (backward compat)."""
        payload = {"is_success": False, "is_server_error": False}
        assert _should_skip_gradient_step(payload) is False

    def test_skip_when_any_one_true(self):
        """If any of success/server_error/truncated is True, we skip."""
        assert _should_skip_gradient_step({"is_success": True}) is True
        assert _should_skip_gradient_step({"is_server_error": True}) is True
        assert _should_skip_gradient_step({"is_truncated": True}) is True


class TestSkipIterLogUpdates:
    """_skip_iter_log_updates(payload) must produce the iter_log shape used when skipping."""

    def test_truncated_payload_produces_loss_none_and_truncated_true(self):
        """When is_truncated=True, iter_log has loss=None and truncated=True (no gradient update)."""
        payload = {"is_truncated": True}
        updates = _skip_iter_log_updates(payload)
        assert updates["loss"] is None
        assert updates["reward"] is None
        assert updates["kl_div"] is None
        assert updates["entropy"] is None
        assert updates["grad_norm"] is None
        assert updates.get("truncated") is True
        assert "server_error" not in updates or updates.get("server_error") is not True

    def test_server_error_payload_produces_server_error_flag(self):
        """When is_server_error=True, iter_log has server_error=True."""
        payload = {"is_server_error": True}
        updates = _skip_iter_log_updates(payload)
        assert updates["loss"] is None
        assert updates.get("server_error") is True

    def test_success_payload_produces_no_extra_flags(self):
        """When only is_success=True (no truncation/server_error), only loss/reward/etc. set to None."""
        payload = {"is_success": True}
        updates = _skip_iter_log_updates(payload)
        assert updates["loss"] is None
        assert "truncated" not in updates or updates.get("truncated") is not True
        assert "server_error" not in updates or updates.get("server_error") is not True


class TestTruncatedProofGradientNotUpdated:
    """When the proof is truncated, the gradient must not be updated (skip path only)."""

    def test_when_truncated_should_skip_gradient_step_returns_true(self):
        """Payload with is_truncated=True must cause gradient step to be skipped."""
        truncated_payload = {
            "is_success": False,
            "is_server_error": False,
            "is_truncated": True,
        }
        assert _should_skip_gradient_step(truncated_payload) is True

    def test_when_truncated_skip_iter_log_has_no_loss(self):
        """Skip path for truncated must set loss=None (no backward/optimizer.step ran)."""
        truncated_payload = {"is_truncated": True}
        updates = _skip_iter_log_updates(truncated_payload)
        assert updates["loss"] is None
        assert updates["reward"] is None
        assert updates["grad_norm"] is None
        assert updates.get("truncated") is True

    def test_truncated_skip_path_implies_no_gradient_update(self):
        """Full contract: truncated payload -> skip -> iter_log has loss=None and truncated=True.
        Entrypoint uses this to continue without appending to metrics (no gradient was applied).
        """
        base_iter_log = {
            "iteration": 1,
            "student_prompt": "p",
            "teacher_prompt": "tp",
            "raw_output": "",
            "extracted_block": "sorry",
            "full_code": "sorry",
            "verification": {"truncated": True},
            "success": False,
            "num_tokens": 100,
        }
        truncated_payload = {
            "is_success": False,
            "is_server_error": False,
            "is_truncated": True,
        }
        assert _should_skip_gradient_step(truncated_payload) is True
        updates = _skip_iter_log_updates(truncated_payload)
        final_iter_log = {**base_iter_log, **updates}
        assert final_iter_log["loss"] is None
        assert final_iter_log["truncated"] is True
        # Entrypoint would do: iter_log.get("loss") or 0.0; when truncated we continue
        # before that, so no metrics append. Here we assert the log shape guarantees no update.
        assert "loss" in final_iter_log and final_iter_log["loss"] is None


class TestKlLossScalarDropFirstN:
    """Training loss excludes first N tokens (context noise)."""

    def test_drop_first_n_when_seq_longer_than_n(self):
        """When seq has > KL_DROP_FIRST_N_TOKENS, loss is mean of per_token_kl[n:], not full mean."""
        # 32 dropped + 2 kept: values 1..32 then 5, 6
        per_token_kl = torch.cat(
            [torch.arange(1.0, 33.0, dtype=torch.float32), torch.tensor([5.0, 6.0])],
            dim=0,
        )
        loss = _kl_loss_scalar(per_token_kl)
        # First 32 dropped; mean of (5, 6) = 5.5
        assert loss.item() == pytest.approx(5.5)
        assert loss.item() != per_token_kl.mean().item()

    def test_use_all_tokens_when_seq_at_most_four(self):
        """When seq has <= 4 tokens, no drop; loss is full mean."""
        per_token_kl = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        loss = _kl_loss_scalar(per_token_kl)
        assert loss.item() == pytest.approx(0.2)
        per_token_kl_4 = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        loss_4 = _kl_loss_scalar(per_token_kl_4)
        assert loss_4.item() == pytest.approx(0.25)

    def test_uses_KL_DROP_FIRST_N_TOKENS(self):
        """Default drop count is KL_DROP_FIRST_N_TOKENS (32)."""
        assert KL_DROP_FIRST_N_TOKENS == 32
        # 33 tokens: first 32 dropped, loss = mean of last token only = 1.0
        per_token_kl = torch.cat([torch.full((32,), 10.0), torch.tensor([1.0])], dim=0)
        loss = _kl_loss_scalar(per_token_kl)
        assert loss.item() == pytest.approx(1.0)


class TestKlLossScalarKlSpan:
    """When kl_span=(start, end) is provided, loss is mean only over that span (KL-mask-final-code)."""

    def test_kl_span_equals_slice_mean(self):
        """loss = per_token_kl[start:end].mean() when kl_span is provided."""
        per_token_kl = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        loss = _kl_loss_scalar(per_token_kl, kl_span=(1, 4))
        assert loss.item() == pytest.approx((2.0 + 3.0 + 4.0) / 3.0)
        assert loss.item() == per_token_kl[1:4].mean().item()

    def test_kl_span_differs_from_full_mean_and_drop_first_n(self):
        """When span is a strict subset, loss differs from full-mean and drop-first-n."""
        # 40 tokens so default drop_first_n would keep last 8
        per_token_kl = torch.arange(40.0, dtype=torch.float32)
        full_mean = per_token_kl.mean().item()
        drop_n_loss = _kl_loss_scalar(per_token_kl).item()
        span_loss = _kl_loss_scalar(per_token_kl, kl_span=(10, 20)).item()
        span_mean = per_token_kl[10:20].mean().item()
        assert span_loss == pytest.approx(span_mean)
        assert span_loss != full_mean
        assert span_loss != drop_n_loss

    def test_kl_span_clamped_to_valid_range(self):
        """Out-of-range span is clamped so we never index out of bounds."""
        per_token_kl = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        loss = _kl_loss_scalar(per_token_kl, kl_span=(0, 10))
        assert loss.item() == pytest.approx(2.0)
        loss_neg = _kl_loss_scalar(per_token_kl, kl_span=(-1, 2))
        assert loss_neg.item() == pytest.approx(1.5)


class TestRunSdpoStepMinibatchContract:
    """Contract for run_sdpo_step_minibatch: n_valid = count of non-skip payloads."""

    def test_n_valid_zero_when_all_skip(self):
        """When all 4 payloads are skip (e.g. truncated), n_valid should be 0 (no gradient step)."""
        payloads = [
            {"is_success": False, "is_server_error": False, "is_truncated": True},
            {"is_success": False, "is_server_error": True, "is_truncated": False},
            {"is_success": True, "is_server_error": False, "is_truncated": False},
            {"is_truncated": True},
        ]
        n_valid = sum(1 for p in payloads if not _should_skip_gradient_step(p))
        assert n_valid == 0

    def test_n_valid_two_when_two_train_two_skip(self):
        """When 2 payloads are train and 2 are skip, n_valid=2 → one gradient step over 2 forwards."""
        payloads = [
            {"is_success": False, "is_server_error": False, "is_truncated": False},
            {"is_success": False, "is_server_error": True, "is_truncated": False},
            {"is_success": False, "is_server_error": False, "is_truncated": False},
            {"is_truncated": True},
        ]
        n_valid = sum(1 for p in payloads if not _should_skip_gradient_step(p))
        assert n_valid == 2
