"""
Unit tests for dynamic_sampling.output (write_outputs: summary.json and raw_logs.json shape).
"""

import json
from pathlib import Path

import pytest

from dynamic_sampling.output import RoundRecord, write_outputs
from dynamic_sampling.state import ProblemResult, RoundState


def test_write_outputs_creates_summary_and_raw_logs(tmp_path: Path):
    """write_outputs creates summary.json and raw_logs.json with expected structure."""
    state = RoundState(
        remaining=[],
        total_generations=10,
        results={
            0: ProblemResult(0, "p0", True, 1, 0),
            1: ProblemResult(1, "p1", False, 3, 1),
        },
    )
    logs_path = tmp_path / "round_0" / "run_X_123" / "logs.json"
    logs_path.parent.mkdir(parents=True)
    logs_path.write_text('[{"problem":{"problem_idx":0,"id":"p0"},"success":true}]')
    round_records = [
        RoundRecord(round_index=0, pass_k=1, generations_this_round=1, logs_path=logs_path),
        RoundRecord(round_index=1, pass_k=2, generations_this_round=4, logs_path=tmp_path / "nonexistent" / "logs.json"),
    ]
    write_outputs(tmp_path, state, round_records, n_problems_total=244)

    summary_path = tmp_path / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert "aggregate" in summary
    assert summary["aggregate"]["n_problems_total"] == 244
    assert summary["aggregate"]["n_problems_attempted"] == 2
    assert summary["aggregate"]["n_passed"] == 1
    assert summary["aggregate"]["total_generations"] == 10
    assert summary["aggregate"]["pass_rate"] == 0.5
    assert "problems" in summary
    assert len(summary["problems"]) == 2
    by_idx = {p["problem_idx"]: p for p in summary["problems"]}
    assert by_idx[0]["passed"] is True and by_idx[0]["attempts_used"] == 1
    assert by_idx[1]["passed"] is False and by_idx[1]["attempts_used"] == 3

    raw_path = tmp_path / "raw_logs.json"
    assert raw_path.is_file()
    raw = json.loads(raw_path.read_text())
    assert "rounds" in raw
    assert len(raw["rounds"]) == 2
    assert raw["rounds"][0]["round_index"] == 0
    assert raw["rounds"][0]["pass_k"] == 1
    assert raw["rounds"][0]["generations_this_round"] == 1
    assert len(raw["rounds"][0]["logs"]) == 1
    assert raw["rounds"][1]["logs"] == []  # nonexistent path -> empty list


def test_write_outputs_empty_state(tmp_path: Path):
    """Empty state still produces valid summary with zero counts."""
    state = RoundState(remaining=[0, 1], total_generations=0, results={})
    write_outputs(tmp_path, state, [], n_problems_total=244)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["aggregate"]["n_problems_attempted"] == 0
    assert summary["aggregate"]["n_passed"] == 0
    assert summary["aggregate"]["pass_rate"] == 0.0
    assert summary["problems"] == []
    raw = json.loads((tmp_path / "raw_logs.json").read_text())
    assert raw["rounds"] == []


def test_write_outputs_pass_rate_one(tmp_path: Path):
    """pass_rate is 1.0 when all attempted passed."""
    state = RoundState(
        remaining=[],
        total_generations=2,
        results={
            0: ProblemResult(0, "a", True, 1, 0),
            1: ProblemResult(1, "b", True, 1, 0),
        },
    )
    write_outputs(tmp_path, state, [], n_problems_total=244)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["aggregate"]["pass_rate"] == 1.0


def test_write_outputs_rounds_order_preserved(tmp_path: Path):
    """raw_logs rounds are in the same order as round_records."""
    state = RoundState(remaining=[], total_generations=0, results={})
    recs = [
        RoundRecord(2, 3, 9, Path("/r2")),
        RoundRecord(0, 1, 2, Path("/r0")),
        RoundRecord(1, 2, 4, Path("/r1")),
    ]
    write_outputs(tmp_path, state, recs, n_problems_total=10)
    raw = json.loads((tmp_path / "raw_logs.json").read_text())
    assert [r["round_index"] for r in raw["rounds"]] == [2, 0, 1]
