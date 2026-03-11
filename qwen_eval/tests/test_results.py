"""
Unit tests for qwen_eval/results.py.

Tests save_success_rate_summary() and related output shape.
No Modal, no GPU. Uses a temp directory for file writes.

Run with:
    python -m pytest qwen_eval/tests/test_results.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_eval.results import save_success_rate_summary


def _make_problem_log(
    problem_id: str,
    attempt_successes: list[bool],
    problem_idx: int | None = None,
) -> dict:
    """Build a minimal problem_log with given success flags per attempt."""
    attempts = [
        {"attempt": i, "success": s}
        for i, s in enumerate(attempt_successes)
    ]
    problem: dict = {"id": problem_id}
    if problem_idx is not None:
        problem["problem_idx"] = problem_idx
    return {
        "problem": problem,
        "attempts": attempts,
    }


class TestSaveSuccessRateSummary:
    def test_empty_problem_logs(self):
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            save_success_rate_summary(run_dir, [], pass_k=4)
            path = run_dir / "success_rate_summary.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["pass_k"] == 4
            assert data["n_problems"] == 0
            assert data["problems"] == []

    def test_single_problem_all_success(self):
        logs = [_make_problem_log("prob_a", [True, True, True, True], problem_idx=0)]
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            save_success_rate_summary(run_dir, logs, pass_k=4)
            data = json.loads((run_dir / "success_rate_summary.json").read_text())
            assert data["n_problems"] == 1
            assert data["problems"][0]["problem_id"] == "prob_a"
            assert data["problems"][0]["problem_idx"] == 0
            assert data["problems"][0]["success_rate"] == 1.0

    def test_single_problem_no_success(self):
        logs = [_make_problem_log("prob_b", [False, False], problem_idx=5)]
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            save_success_rate_summary(run_dir, logs, pass_k=2)
            data = json.loads((run_dir / "success_rate_summary.json").read_text())
            assert data["problems"][0]["problem_id"] == "prob_b"
            assert data["problems"][0]["problem_idx"] == 5
            assert data["problems"][0]["success_rate"] == 0.0

    def test_single_problem_half_success(self):
        logs = [_make_problem_log("prob_c", [True, False, True, False], problem_idx=2)]
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            save_success_rate_summary(run_dir, logs, pass_k=4)
            data = json.loads((run_dir / "success_rate_summary.json").read_text())
            assert data["problems"][0]["problem_id"] == "prob_c"
            assert data["problems"][0]["problem_idx"] == 2
            assert data["problems"][0]["success_rate"] == 0.5

    def test_multiple_problems_mixed_rates(self):
        logs = [
            _make_problem_log("id_1", [True], problem_idx=0),
            _make_problem_log("id_2", [False, False], problem_idx=1),
            _make_problem_log("id_3", [True, True, False], problem_idx=2),
        ]
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            save_success_rate_summary(run_dir, logs, pass_k=3)
            data = json.loads((run_dir / "success_rate_summary.json").read_text())
            assert data["pass_k"] == 3
            assert data["n_problems"] == 3
            assert data["problems"][0]["problem_id"] == "id_1"
            assert data["problems"][0]["problem_idx"] == 0
            assert data["problems"][0]["success_rate"] == 1.0
            assert data["problems"][1]["problem_id"] == "id_2"
            assert data["problems"][1]["problem_idx"] == 1
            assert data["problems"][1]["success_rate"] == 0.0
            assert data["problems"][2]["problem_id"] == "id_3"
            assert data["problems"][2]["problem_idx"] == 2
            assert data["problems"][2]["success_rate"] == round(2 / 3, 4)

    def test_empty_attempts_zero_rate(self):
        logs = [{"problem": {"id": "no_attempts"}, "attempts": []}]
        with tempfile.TemporaryDirectory() as d:
            run_dir = Path(d)
            save_success_rate_summary(run_dir, logs, pass_k=4)
            data = json.loads((run_dir / "success_rate_summary.json").read_text())
            assert data["problems"][0]["problem_id"] == "no_attempts"
            assert data["problems"][0]["success_rate"] == 0.0
