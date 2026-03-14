"""
Unit tests for dynamic_sampling.invoker (discover_run_dir, run_round with mocked subprocess).
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from dynamic_sampling.config import DynamicSamplingConfig
from dynamic_sampling.invoker import (
    RoundResult,
    discover_run_dir,
    run_round,
    _model_safe,
)


def test_model_safe():
    """_model_safe strips org prefix."""
    assert _model_safe("Qwen/Qwen3.5-4B") == "Qwen3.5-4B"
    assert _model_safe("SomeOrg/Model-1B") == "Model-1B"
    assert _model_safe("NoSlash") == "NoSlash"


def test_discover_run_dir_nonexistent(tmp_path: Path):
    """discover_run_dir returns None for nonexistent base."""
    assert discover_run_dir(tmp_path / "missing", "Qwen/Qwen3.5-4B") is None


def test_discover_run_dir_empty(tmp_path: Path):
    """discover_run_dir returns None when no run_* subdirs."""
    (tmp_path / "other_dir").mkdir()
    assert discover_run_dir(tmp_path, "Qwen/Qwen3.5-4B") is None


def test_discover_run_dir_single(tmp_path: Path):
    """discover_run_dir returns the only run_* subdir."""
    run_dir = tmp_path / "run_Qwen3.5-4B_20260312_120000"
    run_dir.mkdir()
    assert discover_run_dir(tmp_path, "Qwen/Qwen3.5-4B") == run_dir


def test_discover_run_dir_multiple_picks_newest(tmp_path: Path):
    """discover_run_dir returns newest by mtime when multiple run_* subdirs."""
    old_dir = tmp_path / "run_Qwen3.5-4B_20260312_100000"
    new_dir = tmp_path / "run_Qwen3.5-4B_20260312_120000"
    old_dir.mkdir()
    new_dir.mkdir()
    # Touch new_dir more recently
    (new_dir / "dummy").write_text("x")
    assert discover_run_dir(tmp_path, "Qwen/Qwen3.5-4B") == new_dir


def test_discover_run_dir_wrong_prefix_ignored(tmp_path: Path):
    """Subdirs not matching run_<model>_ are ignored."""
    (tmp_path / "run_Other_123").mkdir()
    assert discover_run_dir(tmp_path, "Qwen/Qwen3.5-4B") is None


def test_run_round_mocked_subprocess(tmp_path: Path):
    """run_round writes indices, runs modal (mocked), discovers run dir, returns RoundResult."""
    base_dir = tmp_path / "round_0"
    base_dir.mkdir(parents=True)
    run_dir = base_dir / "run_Qwen3.5-4B_20260312_120000"
    run_dir.mkdir()
    (run_dir / "logs.json").write_text("[]")
    summary = {"generation_metrics": {"n_requests": 6}}
    (run_dir / "summary.json").write_text(json.dumps(summary))

    config = DynamicSamplingConfig(budget=100, n_problems=10)
    with patch("dynamic_sampling.invoker.subprocess.run") as mock_run:
        result = run_round(
            config,
            remaining_indices=[0, 1, 2],
            pass_k=2,
            base_dir=base_dir,
            repo_root=tmp_path,
        )
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert "--pass-k" in call_args[0][0]
    assert "2" in call_args[0][0]
    assert "--results-base-dir" in call_args[0][0]
    assert str(base_dir.resolve()) in call_args[0][0]
    assert result.logs_path == run_dir / "logs.json"
    assert result.generations_this_round == 6

    index_path = base_dir / "problem_indices.json"
    assert index_path.is_file()
    data = json.loads(index_path.read_text())
    assert data["problem_indices"] == [0, 1, 2]


def test_run_round_generations_fallback_when_no_summary(tmp_path: Path):
    """When summary.json is missing, generations_this_round = n * pass_k."""
    base_dir = tmp_path / "round_0"
    base_dir.mkdir(parents=True)
    run_dir = base_dir / "run_Qwen3.5-4B_20260312_120000"
    run_dir.mkdir()
    (run_dir / "logs.json").write_text("[]")
    # No summary.json

    config = DynamicSamplingConfig(budget=100)
    with patch("dynamic_sampling.invoker.subprocess.run"):
        result = run_round(
            config,
            remaining_indices=[0, 1],
            pass_k=3,
            base_dir=base_dir,
            repo_root=tmp_path,
        )
    assert result.generations_this_round == 6


def test_run_round_adds_no_think_mode(tmp_path: Path):
    """When no_think_mode is True (default), --no-think-mode is in the command."""
    base_dir = tmp_path / "round_0"
    base_dir.mkdir(parents=True)
    run_dir = base_dir / "run_Qwen3.5-4B_20260312_120000"
    run_dir.mkdir()
    (run_dir / "logs.json").write_text("[]")
    (run_dir / "summary.json").write_text("{}")

    config = DynamicSamplingConfig(budget=100)  # default no_think_mode=True
    with patch("dynamic_sampling.invoker.subprocess.run") as mock_run:
        run_round(config, [0], 1, base_dir, repo_root=tmp_path)
    cmd = mock_run.call_args[0][0]
    assert "--no-think-mode" in cmd


def test_run_round_omits_no_think_mode_when_use_thinking(tmp_path: Path):
    """When no_think_mode is False (--use-thinking), --no-think-mode is not in the command."""
    base_dir = tmp_path / "round_0"
    base_dir.mkdir(parents=True)
    run_dir = base_dir / "run_Qwen3.5-4B_20260312_120000"
    run_dir.mkdir()
    (run_dir / "logs.json").write_text("[]")
    (run_dir / "summary.json").write_text("{}")

    config = DynamicSamplingConfig(budget=100, no_think_mode=False)
    with patch("dynamic_sampling.invoker.subprocess.run") as mock_run:
        run_round(config, [0], 1, base_dir, repo_root=tmp_path)
    cmd = mock_run.call_args[0][0]
    assert "--no-think-mode" not in cmd


def test_run_round_subprocess_failure(tmp_path: Path):
    """run_round propagates subprocess.CalledProcessError."""
    import subprocess
    base_dir = tmp_path / "round_0"
    base_dir.mkdir(parents=True)
    config = DynamicSamplingConfig(budget=100)
    with patch("dynamic_sampling.invoker.subprocess.run") as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, "modal")
        with pytest.raises(subprocess.CalledProcessError):
            run_round(config, [0], 1, base_dir, repo_root=tmp_path)


def test_run_round_no_run_dir_raises(tmp_path: Path):
    """When no run dir exists after subprocess, raises FileNotFoundError."""
    base_dir = tmp_path / "round_0"
    base_dir.mkdir(parents=True)
    # No run_* subdir created
    config = DynamicSamplingConfig(budget=100)
    with patch("dynamic_sampling.invoker.subprocess.run"):
        with pytest.raises(FileNotFoundError, match="No run dir found"):
            run_round(config, [0], 1, base_dir, repo_root=tmp_path)
