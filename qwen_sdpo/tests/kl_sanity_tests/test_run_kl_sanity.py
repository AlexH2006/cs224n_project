"""
TLDR: Tests for the KL sanity check pipeline — JSON loading, _add_tail, _compute_kl_no_grad, and e2e.

Unit tests run without GPU. Integration tests require bitsandbytes and a GPU.
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import pytest
import torch

from qwen_sdpo.tests.kl_sanity_tests.run_kl_sanity import (
    _add_tail,
    _compute_kl_no_grad,
    _load_prompts,
    run_kl_sanity_check,
)

_BITSANDBYTES_AVAILABLE = importlib.util.find_spec("bitsandbytes") is not None


# -----------------------------------------------------------------------------
# Unit tests: _add_tail
# -----------------------------------------------------------------------------


class TestAddTail:
    """Tail bucket logic — output shape and numerical sanity."""

    def test_add_tail_shape(self):
        """Output has one extra dimension (tail bucket) compared to input."""
        log_probs = torch.randn(3, 5)
        out = _add_tail(log_probs)
        assert out.shape == (3, 6)

    def test_add_tail_sums_to_log1(self):
        """exp(sum of tail + topk probs) ≈ 1 for normalized inputs."""
        log_probs = torch.randn(2, 4)
        log_probs = log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)
        out = _add_tail(log_probs)
        total = torch.exp(out).sum(dim=-1)
        assert torch.allclose(total, torch.ones(2), atol=1e-5)


# -----------------------------------------------------------------------------
# Unit tests: _load_prompts
# -----------------------------------------------------------------------------


class TestLoadPrompts:
    """JSON loading and validation."""

    def test_load_prompts_valid(self, tmp_path: Path):
        data = {
            "student_prompt": "A",
            "teacher_prompt": "B",
            "generated_solution": "C",
        }
        p = tmp_path / "prompts.json"
        with open(p, "w") as f:
            json.dump(data, f)
        loaded = _load_prompts(p)
        assert loaded["student_prompt"] == "A"
        assert loaded["teacher_prompt"] == "B"
        assert loaded["generated_solution"] == "C"

    def test_load_prompts_missing_keys_raises(self, tmp_path: Path):
        p = tmp_path / "prompts.json"
        with open(p, "w") as f:
            json.dump({"student_prompt": "A"}, f)
        with pytest.raises(ValueError, match="Missing required keys"):
            _load_prompts(p)

    def test_load_prompts_invalid_json_raises(self, tmp_path: Path):
        p = tmp_path / "prompts.json"
        p.write_text("{ invalid }")
        with pytest.raises(json.JSONDecodeError):
            _load_prompts(p)


# -----------------------------------------------------------------------------
# Integration tests: _compute_kl_no_grad
# -----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _BITSANDBYTES_AVAILABLE,
    reason="bitsandbytes required for integration test",
)
class TestComputeKlNoGrad:
    """Gradient-free KL computation with real model."""

    @pytest.fixture
    def model_and_tokenizer(self):
        from qwen_sdpo.tests.kl_sanity_tests.run_kl_sanity import _load_model_and_tokenizer

        return _load_model_and_tokenizer("Qwen/Qwen3.5-4B")

    def test_compute_kl_no_grad_shapes(self, model_and_tokenizer):
        """per_token_kl has shape [seq_len] and matches generated_ids length."""
        model, tokenizer = model_and_tokenizer
        student = "Prove: 1+1=2"
        teacher = "Prove: 1+1=2. Error: fix it."
        gen_ids = tokenizer("x", return_tensors="pt", add_special_tokens=False).input_ids
        seq_len = gen_ids.shape[1]

        per_token_kl = _compute_kl_no_grad(
            model=model,
            tokenizer=tokenizer,
            distillation_topk=20,
            student_prompt=student,
            teacher_prompt=teacher,
            generated_ids=gen_ids,
        )

        assert per_token_kl.dim() == 1
        assert per_token_kl.shape[0] == seq_len
        assert (per_token_kl >= 0).all().item()

    def test_compute_kl_no_grad_no_gradients(self, model_and_tokenizer):
        """Output tensors have no grad_fn (fully no_grad)."""
        model, tokenizer = model_and_tokenizer
        gen_ids = tokenizer("hi", return_tensors="pt", add_special_tokens=False).input_ids

        per_token_kl = _compute_kl_no_grad(
            model=model,
            tokenizer=tokenizer,
            distillation_topk=20,
            student_prompt="A",
            teacher_prompt="B",
            generated_ids=gen_ids,
        )

        assert per_token_kl.grad_fn is None


# -----------------------------------------------------------------------------
# Integration tests: run_kl_sanity_check e2e
# -----------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _BITSANDBYTES_AVAILABLE,
    reason="bitsandbytes required for e2e test",
)
class TestRunKlSanityCheckE2E:
    """End-to-end pipeline: JSON → KL → output files."""

    def test_e2e_creates_all_outputs(self, tmp_path: Path):
        """Run produces context.json, per_token_kl.json, kl_heatmap.png."""
        prompts_path = tmp_path / "prompts.json"
        prompts_path.write_text(
            json.dumps(
                {
                    "student_prompt": "A",
                    "teacher_prompt": "B",
                    "generated_solution": "x",
                }
            )
        )

        run_dir = run_kl_sanity_check(
            input_path=prompts_path,
            output_base_dir=tmp_path,
            model_name="Qwen/Qwen3.5-4B",
        )

        assert (run_dir / "context.json").exists()
        assert (run_dir / "per_token_kl.json").exists()
        assert (run_dir / "kl_heatmap.png").exists()

    def test_e2e_context_json_structure(self, tmp_path: Path):
        """context.json contains student_prompt, teacher_prompt, generated_solution."""
        prompts_path = tmp_path / "prompts.json"
        prompts_path.write_text(
            json.dumps(
                {
                    "student_prompt": "student",
                    "teacher_prompt": "teacher",
                    "generated_solution": "sol",
                }
            )
        )

        run_dir = run_kl_sanity_check(
            input_path=prompts_path,
            output_base_dir=tmp_path,
            model_name="Qwen/Qwen3.5-4B",
        )

        with open(run_dir / "context.json") as f:
            ctx = json.load(f)
        assert ctx["student_prompt"] == "student"
        assert ctx["teacher_prompt"] == "teacher"
        assert ctx["generated_solution"] == "sol"

    def test_e2e_per_token_kl_structure(self, tmp_path: Path):
        """per_token_kl.json is list of {pos, token_id, token, kl}."""
        prompts_path = tmp_path / "prompts.json"
        prompts_path.write_text(
            json.dumps(
                {
                    "student_prompt": "A",
                    "teacher_prompt": "B",
                    "generated_solution": "ab",
                }
            )
        )

        run_dir = run_kl_sanity_check(
            input_path=prompts_path,
            output_base_dir=tmp_path,
            model_name="Qwen/Qwen3.5-4B",
        )

        with open(run_dir / "per_token_kl.json") as f:
            records = json.load(f)
        assert isinstance(records, list)
        for r in records:
            assert "pos" in r and "token_id" in r and "token" in r and "kl" in r

    def test_e2e_empty_solution_raises(self, tmp_path: Path):
        """Empty generated_solution raises ValueError."""
        prompts_path = tmp_path / "prompts.json"
        prompts_path.write_text(
            json.dumps(
                {
                    "student_prompt": "A",
                    "teacher_prompt": "B",
                    "generated_solution": "",  # tokenizes to empty
                }
            )
        )

        with pytest.raises(ValueError, match="tokenizes to empty"):
            run_kl_sanity_check(
                input_path=prompts_path,
                output_base_dir=tmp_path,
                model_name="Qwen/Qwen3.5-4B",
            )
