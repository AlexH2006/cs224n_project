"""
Unit tests for dynamic_sampling.runner (run loop with mocked run_round).
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from dynamic_sampling.config import DynamicSamplingConfig
from dynamic_sampling.invoker import RoundResult
from dynamic_sampling.runner import run


def test_run_one_round_then_stop_when_all_passed(tmp_path: Path):
    """When first round passes all problems, only one round runs and output is written."""
    round_base = tmp_path / "out" / "round_0"
    round_base.mkdir(parents=True)
    run_dir = round_base / "run_Qwen3.5-4B_123"
    run_dir.mkdir()
    logs_path = run_dir / "logs.json"
    # All 2 problems pass
    logs_data = [
        {"problem": {"problem_idx": 0, "id": "p0"}, "success": True},
        {"problem": {"problem_idx": 1, "id": "p1"}, "success": True},
    ]
    logs_path.write_text(json.dumps(logs_data))
    (run_dir / "summary.json").write_text(json.dumps({"generation_metrics": {"n_requests": 2}}))

    # Budget 2 so pass_k = 2//2 = 1 (one attempt per problem)
    config = DynamicSamplingConfig(budget=2, n_problems=2)
    with patch("dynamic_sampling.runner.run_round") as mock_run_round:
        mock_run_round.return_value = RoundResult(logs_path=logs_path, generations_this_round=2)
        out_dir = run(config, output_dir=tmp_path / "out", repo_root=tmp_path)

    assert mock_run_round.call_count == 1
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "raw_logs.json").is_file()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["aggregate"]["n_passed"] == 2
    assert summary["aggregate"]["n_problems_attempted"] == 2
    assert summary["aggregate"]["total_generations"] == 2


def test_run_stops_when_budget_reached(tmp_path: Path):
    """When budget would be exceeded, we run one round and stop (no second round)."""
    round_base = tmp_path / "out" / "round_0"
    round_base.mkdir(parents=True)
    run_dir = round_base / "run_Qwen3.5-4B_123"
    run_dir.mkdir()
    logs_path = run_dir / "logs.json"
    # None pass so remaining stays [0, 1]
    logs_data = [
        {"problem": {"problem_idx": 0, "id": "p0"}, "success": False},
        {"problem": {"problem_idx": 1, "id": "p1"}, "success": False},
    ]
    logs_path.write_text(json.dumps(logs_data))
    (run_dir / "summary.json").write_text(json.dumps({"generation_metrics": {"n_requests": 2}}))

    config = DynamicSamplingConfig(budget=2, n_problems=2)  # budget exactly 2
    with patch("dynamic_sampling.runner.run_round") as mock_run_round:
        mock_run_round.return_value = RoundResult(logs_path=logs_path, generations_this_round=2)
        out_dir = run(config, output_dir=tmp_path / "out", repo_root=tmp_path)

    # One round used 2 generations; next round would need 2 more but budget is 2, so we stop
    assert mock_run_round.call_count == 1
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["aggregate"]["total_generations"] == 2
    assert summary["aggregate"]["n_passed"] == 0


def test_run_two_rounds_when_some_pass(tmp_path: Path):
    """When some pass in round 0, round 1 runs only on remaining."""
    base = tmp_path / "out"
    # Round 0: problems 0,1,2; 0 and 2 pass
    round0_base = base / "round_0"
    round0_base.mkdir(parents=True)
    run0_dir = round0_base / "run_Qwen3.5-4B_123"
    run0_dir.mkdir()
    logs0 = [
        {"problem": {"problem_idx": 0, "id": "p0"}, "success": True},
        {"problem": {"problem_idx": 1, "id": "p1"}, "success": False},
        {"problem": {"problem_idx": 2, "id": "p2"}, "success": True},
    ]
    (run0_dir / "logs.json").write_text(json.dumps(logs0))
    (run0_dir / "summary.json").write_text(json.dumps({"generation_metrics": {"n_requests": 3}}))
    # Round 1: only problem 1, pass
    round1_base = base / "round_1"
    round1_base.mkdir(parents=True)
    run1_dir = round1_base / "run_Qwen3.5-4B_124"
    run1_dir.mkdir()
    logs1 = [{"problem": {"problem_idx": 1, "id": "p1"}, "success": True}]
    (run1_dir / "logs.json").write_text(json.dumps(logs1))
    (run1_dir / "summary.json").write_text(json.dumps({"generation_metrics": {"n_requests": 1}}))

    # Budget 4: round 0 n=3 -> pass_k=1 (3 gens), round 1 n=1 -> pass_k=4 (4 gens) -> total 7
    config = DynamicSamplingConfig(budget=4, n_problems=3)
    call_count = 0
    def mock_run_round(*args, **kwargs):
        nonlocal call_count
        if call_count == 0:
            call_count += 1
            return RoundResult(logs_path=run0_dir / "logs.json", generations_this_round=3)
        call_count += 1
        return RoundResult(logs_path=run1_dir / "logs.json", generations_this_round=4)

    with patch("dynamic_sampling.runner.run_round", side_effect=mock_run_round):
        out_dir = run(config, output_dir=base, repo_root=tmp_path)

    assert call_count == 2
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["aggregate"]["n_passed"] == 3
    assert summary["aggregate"]["total_generations"] == 7
    raw = json.loads((out_dir / "raw_logs.json").read_text())
    assert len(raw["rounds"]) == 2
    assert raw["rounds"][0]["generations_this_round"] == 3
    assert raw["rounds"][1]["generations_this_round"] == 4


def test_run_zero_remaining_skips_round(tmp_path: Path):
    """When n_problems=0, remaining is empty so no round is run."""
    config = DynamicSamplingConfig(budget=100, n_problems=0)
    with patch("dynamic_sampling.runner.run_round") as mock_run_round:
        out_dir = run(config, output_dir=tmp_path / "out", repo_root=tmp_path)
    mock_run_round.assert_not_called()
    assert (out_dir / "summary.json").is_file()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["aggregate"]["n_problems_attempted"] == 0


def test_run_uses_problem_index_file_for_first_round(tmp_path: Path):
    """When problem_index_file is set, first round runs only on those indices; n_problems_total = len(indices)."""
    idx_file = tmp_path / "problem_idx.json"
    idx_file.write_text(json.dumps({"problem_indices": [2, 5, 7]}))
    round0_base = tmp_path / "out" / "round_0"
    round0_base.mkdir(parents=True)
    run0_dir = round0_base / "run_Qwen3.5-4B_123"
    run0_dir.mkdir()
    logs0_data = [
        {"problem": {"problem_idx": 2, "id": "p2"}, "success": True},
        {"problem": {"problem_idx": 5, "id": "p5"}, "success": False},
        {"problem": {"problem_idx": 7, "id": "p7"}, "success": True},
    ]
    (run0_dir / "logs.json").write_text(json.dumps(logs0_data))
    (run0_dir / "summary.json").write_text(json.dumps({"generation_metrics": {"n_requests": 3}}))
    round1_base = tmp_path / "out" / "round_1"
    round1_base.mkdir(parents=True)
    run1_dir = round1_base / "run_Qwen3.5-4B_124"
    run1_dir.mkdir()
    logs1_data = [{"problem": {"problem_idx": 5, "id": "p5"}, "success": False}]
    (run1_dir / "logs.json").write_text(json.dumps(logs1_data))
    (run1_dir / "summary.json").write_text(json.dumps({"generation_metrics": {"n_requests": 1}}))

    config = DynamicSamplingConfig(budget=10, problem_index_file=str(idx_file))
    with patch("dynamic_sampling.runner.run_round") as m:
        m.side_effect = [
            RoundResult(logs_path=run0_dir / "logs.json", generations_this_round=3),
            RoundResult(logs_path=run1_dir / "logs.json", generations_this_round=1),
        ]
        out_dir = run(config, output_dir=tmp_path / "out", repo_root=tmp_path)

    assert m.call_count >= 1
    assert m.call_args_list[0][0][1] == [2, 5, 7], "first round problem list from problem_index_file"
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["aggregate"]["n_problems_total"] == 3
    assert summary["aggregate"]["n_problems_attempted"] == 3
    assert summary["aggregate"]["n_passed"] == 2


def test_run_problem_index_file_not_found_raises(tmp_path: Path):
    """When problem_index_file path does not exist, run raises FileNotFoundError."""
    config = DynamicSamplingConfig(budget=10, problem_index_file=str(tmp_path / "nonexistent.json"))
    with pytest.raises(FileNotFoundError, match="problem_index_file not found"):
        run(config, output_dir=tmp_path / "out", repo_root=tmp_path)
