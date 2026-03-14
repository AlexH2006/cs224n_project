"""
Unit tests that qwen_eval supports --results-base-dir for dynamic_sampling.

Verifies: (1) run_eval is implemented with results_base_dir parameter;
(2) make_run_dir uses cfg.results_base_dir when set.
"""

from pathlib import Path
import tempfile

import pytest

from qwen_eval.config import EvalConfig
from qwen_eval.results import make_run_dir


def test_run_eval_implemented_with_results_base_dir():
    """run_eval must be implemented with results_base_dir so --results-base-dir works."""
    modal_app_path = Path(__file__).resolve().parent.parent.parent / "qwen_eval" / "modal_app.py"
    source = modal_app_path.read_text()
    assert "results_base_dir" in source, "modal_app.run_eval must have results_base_dir parameter"
    assert "cfg.results_base_dir = results_base_dir" in source or 'cfg.results_base_dir = results_base_dir' in source, (
        "run_eval must set cfg.results_base_dir when provided"
    )


def test_make_run_dir_uses_results_base_dir():
    """make_run_dir must place run dir under cfg.results_base_dir when set."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "custom_base"
        cfg = EvalConfig(
            model_name="Qwen/Qwen3.5-4B",
            n_problems=1,
            pass_k=1,
            problem_indices=[0],
            results_base_dir=str(base),
        )
        run_dir = make_run_dir(cfg)
        assert run_dir.is_dir()
        assert run_dir.resolve().is_relative_to(base.resolve())
        assert run_dir.name.startswith("run_Qwen3.5-4B_")
        assert base.exists()


def test_eval_config_default_results_base_dir():
    """Default EvalConfig uses 'baseline' when results_base_dir not overridden."""
    cfg = EvalConfig(model_name="Qwen/Qwen3.5-4B", n_problems=1, pass_k=1)
    assert cfg.results_base_dir == "baseline"
