"""
Unit tests for dynamic_sampling.entrypoint (CLI parsing and run invocation).
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_entrypoint_parses_budget_and_calls_run(tmp_path: Path):
    """CLI --budget and --output-dir are passed through to run."""
    with patch("dynamic_sampling.entrypoint.run") as mock_run:
        mock_run.return_value = tmp_path / "out"
        sys.argv = [
            "entrypoint",
            "--budget",
            "100",
            "--output-dir",
            str(tmp_path / "custom_out"),
        ]
        from dynamic_sampling.entrypoint import main
        main()
    mock_run.assert_called_once()
    call_kw = mock_run.call_args[1]
    assert call_kw["output_dir"] is not None
    assert "run_" in str(call_kw["output_dir"])
    assert "custom_out" in str(call_kw["output_dir"])
    config = mock_run.call_args[0][0]
    assert config.budget == 100


def test_entrypoint_default_budget():
    """Default budget is used when not specified."""
    with patch("dynamic_sampling.entrypoint.run") as mock_run:
        mock_run.return_value = Path("/tmp/out")
        sys.argv = ["entrypoint"]
        from dynamic_sampling.entrypoint import main
        main()
    config = mock_run.call_args[0][0]
    from dynamic_sampling.constants import DEFAULT_BUDGET
    assert config.budget == DEFAULT_BUDGET


def test_entrypoint_use_thinking(tmp_path: Path):
    """--use-thinking sets config.no_think_mode to False (thinking enabled)."""
    with patch("dynamic_sampling.entrypoint.run") as mock_run:
        mock_run.return_value = tmp_path
        sys.argv = ["entrypoint", "--budget", "10", "--use-thinking"]
        from dynamic_sampling.entrypoint import main
        main()
    config = mock_run.call_args[0][0]
    assert config.no_think_mode is False


def test_entrypoint_repo_root(tmp_path: Path):
    """--repo-root is passed to run."""
    with patch("dynamic_sampling.entrypoint.run") as mock_run:
        mock_run.return_value = tmp_path
        sys.argv = ["entrypoint", "--budget", "10", "--repo-root", str(tmp_path)]
        from dynamic_sampling.entrypoint import main
        main()
    call_kw = mock_run.call_args[1]
    assert call_kw["repo_root"] == tmp_path


def test_entrypoint_problem_index_file(tmp_path: Path):
    """--problem-index-file sets config.problem_index_file."""
    with patch("dynamic_sampling.entrypoint.run") as mock_run:
        mock_run.return_value = tmp_path
        sys.argv = ["entrypoint", "--budget", "10", "--problem-index-file", "dynamic_sampling/problem_idx.json"]
        from dynamic_sampling.entrypoint import main
        main()
    config = mock_run.call_args[0][0]
    assert config.problem_index_file == "dynamic_sampling/problem_idx.json"
