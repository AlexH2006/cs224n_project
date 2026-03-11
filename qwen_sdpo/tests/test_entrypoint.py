"""
Unit tests for qwen_sdpo/entrypoint.py (config dict, think-mode wiring, batch driver).

TLDR: Tests the serialisable config dict and that use_think_mode is passed through.
Batch test mocks run_main and trainer.reset_to_base; no Modal, no network.

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
from qwen_sdpo.entrypoint import _build_cfg_dict, run_main_batch


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
