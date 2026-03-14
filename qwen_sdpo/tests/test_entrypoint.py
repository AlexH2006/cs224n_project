"""
Unit tests for qwen_sdpo/entrypoint.py (config dict, think-mode wiring, batch driver).

TLDR: Tests the serialisable config dict and that use_think_mode is passed through.
Batch test mocks run_main and trainer.reset_to_base; no Modal, no network.
Payload contract: is_truncated must be included so the trainer can skip gradient when truncated.

Run with:
    python3 -m pytest qwen_sdpo/tests/test_entrypoint.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qwen_sdpo.checkpoint_manifest import read_manifest
from qwen_sdpo.config import SDPOConfig
from qwen_sdpo.entrypoint import _build_cfg_dict, _feedback_for_teacher_prompt, run_main_batch
from qwen_sdpo.parsing import extract_full_lean_block_parsed
from qwen_sdpo.modal_trainer import _should_skip_gradient_step, _skip_iter_log_updates


def test_truncation_detection_to_gradient_skip_flow():
    """Full chain: parse detects truncation -> payload has is_truncated -> trainer skips gradient.
    Ensures that when the proof is truncated, the gradient is not updated.
    """
    # 1. Truncation detection (same as in entrypoint after generate_only)
    raw_output = "<think>incomplete reasoning"
    finish_reason = "length"
    parse_result = extract_full_lean_block_parsed(raw_output, finish_reason=finish_reason)
    assert parse_result.truncated is True

    # 2. Payload as entrypoint builds it (is_truncated from parse_result)
    is_truncated = parse_result.truncated
    payload = {
        "iteration": 1,
        "base_prompt": "p",
        "teacher_prompt": "tp",
        "raw_output": raw_output,
        "generated_ids": [1, 2, 3],
        "teacher_response_ids": [1, 2, 3],
        "teacher_response_mode": "full",
        "cot_len": 0,
        "extracted_block": parse_result.block,
        "full_code": "sorry",
        "verification": {"truncated": True, "feedback": "Generation was truncated; no complete proof was produced."},
        "num_tokens": 3,
        "feedback": "Generation was truncated; no complete proof was produced.",
        "is_success": False,
        "is_server_error": False,
        "is_truncated": is_truncated,
    }
    assert payload["is_truncated"] is True

    # 3. Trainer skip logic: gradient step must be skipped
    assert _should_skip_gradient_step(payload) is True

    # 4. Skip path returns iter_log with no loss (no backward/optimizer.step)
    updates = _skip_iter_log_updates(payload)
    assert updates["loss"] is None
    assert updates.get("truncated") is True


def test_non_truncated_failure_does_not_skip_gradient():
    """When parse succeeds (or fails without truncation), is_truncated=False -> gradient step runs."""
    raw_output = "<think>done</think>\n```lean4\nsorry\n```"
    parse_result = extract_full_lean_block_parsed(raw_output, finish_reason="stop")
    assert parse_result.truncated is False

    payload = {
        "is_success": False,
        "is_server_error": False,
        "is_truncated": parse_result.truncated,
    }
    assert payload["is_truncated"] is False
    assert _should_skip_gradient_step(payload) is False


def test_payload_includes_is_truncated_for_trainer_skip():
    """Payload sent to run_sdpo_step must include is_truncated so trainer skips gradient when truncated."""
    # Contract: entrypoint builds this shape (see run_main payload dict); trainer reads is_truncated.
    payload_with_truncated = {
        "iteration": 1,
        "base_prompt": "p",
        "teacher_prompt": "tp",
        "raw_output": "",
        "generated_ids": [],
        "teacher_response_ids": [],
        "teacher_response_mode": "full",
        "cot_len": 0,
        "extracted_block": "sorry",
        "full_code": "sorry",
        "verification": {},
        "num_tokens": 0,
        "feedback": "Generation was truncated; no complete proof was produced.",
        "is_success": False,
        "is_server_error": False,
        "is_truncated": True,
    }
    assert "is_truncated" in payload_with_truncated
    assert payload_with_truncated["is_truncated"] is True


def test_build_cfg_dict_includes_use_think_mode():
    """_build_cfg_dict must include use_think_mode so Modal and logs see it."""
    cfg = SDPOConfig()
    d = _build_cfg_dict(cfg)
    assert "use_think_mode" in d, "cfg_dict must contain use_think_mode"
    assert d["use_think_mode"] is True, "Default should be thinking mode on"


def test_build_cfg_dict_use_think_mode_false():
    """_build_cfg_dict reflects use_think_mode=False for non-thinking runs."""
    cfg = SDPOConfig(use_think_mode=False)
    d = _build_cfg_dict(cfg)
    assert d["use_think_mode"] is False


def test_build_cfg_dict_has_required_keys():
    """cfg_dict contains all keys needed by Modal and results."""
    cfg = SDPOConfig(use_think_mode=False, teacher_response_mode="code_only")
    d = _build_cfg_dict(cfg)
    required = [
        "model_name", "use_think_mode", "teacher_response_mode",
        "max_new_tokens", "max_iterations", "learning_rate",
        "problem_idx", "default_header",
    ]
    for k in required:
        assert k in d, f"cfg_dict missing key {k!r}"


def test_config_max_feedback_errors_default():
    """SDPOConfig has max_feedback_errors; default is 10 (first N errors in teacher prompt)."""
    cfg = SDPOConfig()
    assert hasattr(cfg, "max_feedback_errors")
    assert cfg.max_feedback_errors == 10


def test_config_max_feedback_errors_override():
    """max_feedback_errors can be overridden."""
    cfg = SDPOConfig(max_feedback_errors=5)
    assert cfg.max_feedback_errors == 5


def test_config_minibatch_size_default():
    """SDPOConfig has minibatch_size; default is 1 (current single-sample behavior)."""
    cfg = SDPOConfig()
    assert hasattr(cfg, "minibatch_size")
    assert cfg.minibatch_size == 1


def test_config_minibatch_size_override():
    """minibatch_size can be overridden for minibatch on-policy training."""
    cfg = SDPOConfig(minibatch_size=4)
    assert cfg.minibatch_size == 4


def test_build_cfg_dict_includes_minibatch_size():
    """_build_cfg_dict includes minibatch_size so Modal trainer receives it."""
    cfg = SDPOConfig(minibatch_size=2)
    d = _build_cfg_dict(cfg)
    assert "minibatch_size" in d
    assert d["minibatch_size"] == 2


def test_payload_has_kl_final_code_span_when_kl_mask_true_and_parse_succeeds():
    """When kl_mask_final_code_only=True and parse succeeds with a code block, payload contains kl_final_code_start and kl_final_code_end."""
    with patch("qwen_sdpo.entrypoint._load_problem") as mock_load, \
         patch("qwen_sdpo.entrypoint._verify_with_retries") as mock_verify, \
         patch("qwen_sdpo.entrypoint.build_student_prompt") as mock_sp, \
         patch("qwen_sdpo.entrypoint.build_teacher_prompt") as mock_tp, \
         patch("qwen_sdpo.entrypoint.build_teacher_prompt_no_thinking") as mock_tp_no_think, \
         patch("qwen_sdpo.entrypoint.get_code_token_slice") as mock_get_slice:
        mock_load.return_value = {
            "problem_id": "p0",
            "formal_statement": "theorem t : True := by sorry",
            "informal_stmt": "prove True",
            "header": "",
            "problem_idx": 0,
        }
        mock_sp.return_value = "student_prompt"
        mock_tp.return_value = "teacher_prompt"
        mock_tp_no_think.return_value = "teacher_prompt"
        mock_verify.return_value = {"success": False, "complete": False, "errors": []}
        mock_get_slice.return_value = ([10, 20, 30], 5)

        cfg = SDPOConfig(
            model_name="Qwen/Qwen3.5-4B",
            max_iterations=1,
            minibatch_size=1,
            kl_mask_final_code_only=True,
        )
        mock_trainer = MagicMock()
        # Use a block that parses successfully (block != "sorry" so parse_result.failed is False).
        raw_with_block = "```lean4\nby trivial\n```"
        mock_trainer.generate_batch.remote.return_value = [
            (raw_with_block, [1, 2, 3, 4, 5, 6, 7, 8], "stop"),
        ]
        mock_trainer.run_sdpo_step_minibatch.remote.return_value = {
            "iteration": 1, "loss": 0.5, "reward": 0.0, "n_valid": 1, "minibatch_size": 1,
            "kl_div": 0.0, "entropy": 0.0, "grad_norm": 0.1,
        }
        mock_trainer.finalize_run.remote.return_value = {
            "success": False, "iteration_logs": [], "metrics": {}, "run_dir": "/out",
            "local_run_dir": "/local", "best_proof": None, "config": {}, "end_time": "", "total_generation_tokens": 0,
        }

        with patch("transformers.AutoTokenizer") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.pad_token = None
            mock_tok_cls.from_pretrained.return_value = mock_tok

            from qwen_sdpo.entrypoint import run_main
            run_main(trainer=mock_trainer, cfg=cfg)

        step_call = mock_trainer.run_sdpo_step_minibatch.remote.call_args
        payloads = step_call[0][1]
        assert len(payloads) == 1
        payload = payloads[0]
        assert payload["kl_final_code_start"] == 5
        assert payload["kl_final_code_end"] == 8


def test_payload_has_kl_final_code_span_none_when_kl_mask_false():
    """When kl_mask_final_code_only=False, payload has kl_final_code_start and kl_final_code_end as None."""
    with patch("qwen_sdpo.entrypoint._load_problem") as mock_load, \
         patch("qwen_sdpo.entrypoint._verify_with_retries") as mock_verify, \
         patch("qwen_sdpo.entrypoint.build_student_prompt") as mock_sp, \
         patch("qwen_sdpo.entrypoint.build_teacher_prompt") as mock_tp, \
         patch("qwen_sdpo.entrypoint.build_teacher_prompt_no_thinking") as mock_tp_no_think:
        mock_load.return_value = {
            "problem_id": "p0",
            "formal_statement": "theorem t : True := by sorry",
            "informal_stmt": "prove True",
            "header": "",
            "problem_idx": 0,
        }
        mock_sp.return_value = "student_prompt"
        mock_tp.return_value = "teacher_prompt"
        mock_tp_no_think.return_value = "teacher_prompt"
        mock_verify.return_value = {"success": False, "complete": False, "errors": []}

        cfg = SDPOConfig(
            model_name="Qwen/Qwen3.5-4B",
            max_iterations=1,
            minibatch_size=1,
            kl_mask_final_code_only=False,
        )
        mock_trainer = MagicMock()
        mock_trainer.generate_batch.remote.return_value = [
            ("```lean4\nsorry\n```", [1, 2, 3], "stop"),
        ]
        mock_trainer.run_sdpo_step_minibatch.remote.return_value = {
            "iteration": 1, "loss": 0.5, "reward": 0.0, "n_valid": 1, "minibatch_size": 1,
            "kl_div": 0.0, "entropy": 0.0, "grad_norm": 0.1,
        }
        mock_trainer.finalize_run.remote.return_value = {
            "success": False, "iteration_logs": [], "metrics": {}, "run_dir": "/out",
            "local_run_dir": "/local", "best_proof": None, "config": {}, "end_time": "", "total_generation_tokens": 0,
        }

        with patch("transformers.AutoTokenizer") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.pad_token = None
            mock_tok_cls.from_pretrained.return_value = mock_tok

            from qwen_sdpo.entrypoint import run_main
            run_main(trainer=mock_trainer, cfg=cfg)

        step_call = mock_trainer.run_sdpo_step_minibatch.remote.call_args
        payload = step_call[0][1][0]
        assert payload.get("kl_final_code_start") is None
        assert payload.get("kl_final_code_end") is None


def test_payload_has_kl_final_code_span_none_when_parse_fails():
    """When kl_mask_final_code_only=True but parse fails (no code block), payload has kl_final_code_start/end as None."""
    with patch("qwen_sdpo.entrypoint._load_problem") as mock_load, \
         patch("qwen_sdpo.entrypoint._verify_with_retries") as mock_verify, \
         patch("qwen_sdpo.entrypoint.build_student_prompt") as mock_sp, \
         patch("qwen_sdpo.entrypoint.build_teacher_prompt") as mock_tp, \
         patch("qwen_sdpo.entrypoint.build_teacher_prompt_no_thinking") as mock_tp_no_think:
        mock_load.return_value = {
            "problem_id": "p0",
            "formal_statement": "theorem t : True := by sorry",
            "informal_stmt": "prove True",
            "header": "",
            "problem_idx": 0,
        }
        mock_sp.return_value = "student_prompt"
        mock_tp.return_value = "teacher_prompt"
        mock_tp_no_think.return_value = "teacher_prompt"
        mock_verify.return_value = {"success": False, "complete": False, "errors": []}

        cfg = SDPOConfig(
            model_name="Qwen/Qwen3.5-4B",
            max_iterations=1,
            minibatch_size=1,
            kl_mask_final_code_only=True,
        )
        mock_trainer = MagicMock()
        raw_no_block = "no lean4 block here"
        mock_trainer.generate_batch.remote.return_value = [
            (raw_no_block, [1, 2, 3], "stop"),
        ]
        mock_trainer.run_sdpo_step_minibatch.remote.return_value = {
            "iteration": 1, "loss": 0.5, "reward": 0.0, "n_valid": 1, "minibatch_size": 1,
            "kl_div": 0.0, "entropy": 0.0, "grad_norm": 0.1,
        }
        mock_trainer.finalize_run.remote.return_value = {
            "success": False, "iteration_logs": [], "metrics": {}, "run_dir": "/out",
            "local_run_dir": "/local", "best_proof": None, "config": {}, "end_time": "", "total_generation_tokens": 0,
        }

        with patch("transformers.AutoTokenizer") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.pad_token = None
            mock_tok_cls.from_pretrained.return_value = mock_tok

            from qwen_sdpo.entrypoint import run_main
            run_main(trainer=mock_trainer, cfg=cfg)

        step_call = mock_trainer.run_sdpo_step_minibatch.remote.call_args
        payload = step_call[0][1][0]
        assert payload.get("kl_final_code_start") is None
        assert payload.get("kl_final_code_end") is None


def test_run_main_minibatch_size_calls_generate_batch_and_step_minibatch():
    """With minibatch_size=2, run_main calls generate_batch.remote with 2 prompts and run_sdpo_step_minibatch.remote with 2 payloads."""
    with patch("qwen_sdpo.entrypoint._load_problem") as mock_load, \
         patch("qwen_sdpo.entrypoint._verify_with_retries") as mock_verify, \
         patch("qwen_sdpo.entrypoint.build_student_prompt") as mock_sp, \
         patch("qwen_sdpo.entrypoint.build_teacher_prompt") as mock_tp, \
         patch("qwen_sdpo.entrypoint.build_teacher_prompt_no_thinking") as mock_tp_no_think:
        mock_load.return_value = {
            "problem_id": "p0",
            "formal_statement": "theorem t : True := by sorry",
            "informal_stmt": "prove True",
            "header": "",
            "problem_idx": 0,
        }
        mock_sp.return_value = "student_prompt"
        mock_tp.return_value = "teacher_prompt"
        mock_tp_no_think.return_value = "teacher_prompt"
        mock_verify.return_value = {"success": False, "complete": False, "errors": []}

        cfg = SDPOConfig(model_name="Qwen/Qwen3.5-4B", max_iterations=1, minibatch_size=2)
        mock_trainer = MagicMock()
        mock_trainer.generate_batch.remote.return_value = [
            ("```lean4\nsorry\n```", [1, 2, 3], "stop"),
            ("```lean4\nsorry\n```", [1, 2, 3], "stop"),
        ]
        mock_trainer.run_sdpo_step_minibatch.remote.return_value = {
            "iteration": 1, "loss": 0.5, "reward": 0.0, "n_valid": 2, "minibatch_size": 2,
            "kl_div": 0.0, "entropy": 0.0, "grad_norm": 0.1,
        }
        mock_trainer.finalize_run.remote.return_value = {
            "success": False, "iteration_logs": [], "metrics": {}, "run_dir": "/out",
            "local_run_dir": "/local", "best_proof": None, "config": {}, "end_time": "", "total_generation_tokens": 0,
        }

        with patch("transformers.AutoTokenizer") as mock_tok_cls:
            mock_tok = MagicMock()
            mock_tok.pad_token = None
            mock_tok_cls.from_pretrained.return_value = mock_tok

            from qwen_sdpo.entrypoint import run_main
            run_main(trainer=mock_trainer, cfg=cfg)

        assert mock_trainer.generate_batch.remote.call_count >= 1
        call_args = mock_trainer.generate_batch.remote.call_args
        prompts = call_args[0][1]  # second positional arg
        assert len(prompts) == 2
        assert prompts[0] == "student_prompt" and prompts[1] == "student_prompt"

        assert mock_trainer.run_sdpo_step_minibatch.remote.call_count >= 1
        step_call = mock_trainer.run_sdpo_step_minibatch.remote.call_args
        payloads = step_call[0][1]
        assert len(payloads) == 2


class TestFeedbackForTeacherPrompt:
    """_feedback_for_teacher_prompt truncates to first N errors; fallback when no errors list."""

    def test_truncates_to_first_10_when_many_errors(self):
        """When verification has 15 errors, result is first 10 joined by newline."""
        verification = {"errors": [f"error_{i}" for i in range(15)], "feedback": "full"}
        result = _feedback_for_teacher_prompt(verification, max_errors=10)
        lines = result.split("\n")
        assert len(lines) == 10
        assert lines[0] == "error_0"
        assert lines[9] == "error_9"
        assert "error_10" not in result

    def test_uses_all_errors_when_fewer_than_max(self):
        """When verification has 3 errors, result is all 3 joined."""
        verification = {"errors": ["e1", "e2", "e3"], "feedback": "ignored"}
        result = _feedback_for_teacher_prompt(verification, max_errors=10)
        assert result == "e1\ne2\ne3"

    def test_fallback_to_feedback_string_when_no_errors_list(self):
        """When errors is empty or missing, return verification feedback (e.g. truncation/parse message)."""
        verification = {"errors": [], "feedback": "Parse failed."}
        assert _feedback_for_teacher_prompt(verification, max_errors=10) == "Parse failed."
        verification_no_errors = {"feedback": "Generation was truncated."}
        assert _feedback_for_teacher_prompt(verification_no_errors, max_errors=10) == "Generation was truncated."

    def test_fallback_when_feedback_missing(self):
        """When no errors list and no feedback, return default message."""
        verification = {}
        assert _feedback_for_teacher_prompt(verification, max_errors=10) == "Proof verification failed."

    def test_default_config_max_feedback_errors_integration(self):
        """run_main uses cfg.max_feedback_errors; default 10 truncates to first 10 errors."""
        cfg = SDPOConfig()
        verification = {"errors": [f"err_{i}" for i in range(20)], "feedback": "full"}
        result = _feedback_for_teacher_prompt(verification, cfg.max_feedback_errors)
        assert len(result.split("\n")) == 10
        assert result.startswith("err_0")
        assert "err_10" not in result


def test_cli_non_thinking_config_build():
    """SDPOConfig built as modal_app would for --use-think-mode false is non-thinking."""
    # Same kwargs as modal_app.run_sdpo when user passes use_think_mode=False
    cfg = SDPOConfig(
        model_name="Qwen/Qwen3.5-4B",
        problem_idx=0,
        max_iterations=5,
        gpu="H100",
        use_think_mode=False,
        teacher_response_mode="full_output",
    )
    assert cfg.use_think_mode is False
    d = _build_cfg_dict(cfg)
    assert d["use_think_mode"] is False


def test_run_main_batch_appends_manifest_and_calls_reset_to_base(tmp_path: Path) -> None:
    """run_main_batch inits manifest, calls run_main per problem, appends entry, reset_to_base between."""
    cfg = SDPOConfig(model_name="Qwen/Qwen3.5-4B", max_iterations=2)
    manifest_path = tmp_path / "manifest.json"
    problem_id_by_idx = {10: "prob_10", 20: "prob_20"}

    mock_trainer = MagicMock()
    mock_trainer.reset_to_base.remote.return_value = {"status": "ok", "model_name": cfg.model_name}

    def fake_run_main(trainer, cfg):
        return {
            "run_dir": f"/output/run_{cfg.problem_idx}",
            "success": cfg.problem_idx == 10,
            "local_run_dir": str(tmp_path / f"local_run_{cfg.problem_idx}"),
            "problem_id": problem_id_by_idx.get(cfg.problem_idx, str(cfg.problem_idx)),
        }

    with patch("qwen_sdpo.entrypoint.run_main", side_effect=fake_run_main):
        summary = run_main_batch(
            trainer=mock_trainer,
            cfg=cfg,
            problem_indices=[10, 20],
            problem_id_by_idx=problem_id_by_idx,
            manifest_path=manifest_path,
        )

    assert summary["total"] == 2
    assert summary["success_count"] == 1
    assert summary["manifest_path"] == str(manifest_path)
    assert len(summary["results"]) == 2

    data = read_manifest(manifest_path)
    assert len(data["checkpoints"]) == 2
    assert data["checkpoints"][0]["problem_idx"] == 10
    assert data["checkpoints"][0]["problem_id"] == "prob_10"
    assert data["checkpoints"][0]["success"] is True
    assert data["checkpoints"][1]["problem_idx"] == 20
    assert data["checkpoints"][1]["problem_id"] == "prob_20"
    assert data["checkpoints"][1]["success"] is False

    # reset_to_base called once (after first problem, before second).
    assert mock_trainer.reset_to_base.remote.call_count == 1

    # run_main return contract: batch expects run_dir and local_run_dir for manifest.
    assert summary["results"][0]["run_dir"] == "/output/run_10"
    assert summary["results"][0]["local_run_dir"] == str(tmp_path / "local_run_10")
    assert data["checkpoints"][0]["modal_run_dir"] == "/output/run_10"
    assert data["checkpoints"][0]["local_run_dir"] == str(tmp_path / "local_run_10")


def test_run_main_batch_first_three_from_sampled_problems_json(tmp_path: Path) -> None:
    """Batch over first 3 problems from results/Qwen3.5_4B_sampled_no_thinking/sampled_problems.json."""
    import json

    project_root = Path(__file__).parent.parent.parent
    json_path = project_root / "results" / "Qwen3.5_4B_sampled_no_thinking" / "sampled_problems.json"
    if not json_path.exists():
        import pytest
        pytest.skip(f"sampled_problems.json not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)
    problem_indices = data["problem_indices"][:3]  # first three: 7, 9, 10
    problem_id_by_idx = {p["problem_idx"]: p["problem_id"] for p in data["problems"]}

    assert problem_indices == [7, 9, 10]
    assert problem_id_by_idx[7] == "mathd_algebra_44"
    assert problem_id_by_idx[9] == "mathd_numbertheory_1124"
    assert problem_id_by_idx[10] == "imo_1983_p6"

    cfg = SDPOConfig(model_name="Qwen/Qwen3.5-4B", max_iterations=2)
    manifest_path = tmp_path / "manifest.json"
    mock_trainer = MagicMock()
    mock_trainer.reset_to_base.remote.return_value = {"status": "ok", "model_name": cfg.model_name}

    def fake_run_main(trainer, cfg):
        return {
            "run_dir": f"/output/Qwen3.5-4B/run_{cfg.problem_idx}",
            "success": False,
            "local_run_dir": str(tmp_path / f"local_{cfg.problem_idx}"),
            "problem_id": problem_id_by_idx.get(cfg.problem_idx, str(cfg.problem_idx)),
        }

    with patch("qwen_sdpo.entrypoint.run_main", side_effect=fake_run_main):
        summary = run_main_batch(
            trainer=mock_trainer,
            cfg=cfg,
            problem_indices=problem_indices,
            problem_id_by_idx=problem_id_by_idx,
            manifest_path=manifest_path,
        )

    assert summary["total"] == 3
    assert summary["success_count"] == 0
    assert len(summary["results"]) == 3
    data = read_manifest(manifest_path)
    assert len(data["checkpoints"]) == 3
    assert data["checkpoints"][0]["problem_idx"] == 7
    assert data["checkpoints"][0]["problem_id"] == "mathd_algebra_44"
    assert data["checkpoints"][1]["problem_idx"] == 9
    assert data["checkpoints"][1]["problem_id"] == "mathd_numbertheory_1124"
    assert data["checkpoints"][2]["problem_idx"] == 10
    assert data["checkpoints"][2]["problem_id"] == "imo_1983_p6"
    assert data["base_model"] == "Qwen/Qwen3.5-4B"
    # reset_to_base called twice (after problem 1 and after problem 2, not after problem 3).
    assert mock_trainer.reset_to_base.remote.call_count == 2
