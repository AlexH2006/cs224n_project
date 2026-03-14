"""
Unit tests for dynamic_sampling.state (RoundState, update_from_round_logs, extract_passed_indices).
"""

import pytest

from dynamic_sampling.state import (
    ProblemResult,
    RoundState,
    initial_state,
    initial_state_from_indices,
    extract_passed_indices,
)


def _log_entry(problem_idx: int, problem_id: str, success: bool) -> dict:
    """Minimal qwen_eval-style problem log entry."""
    return {
        "problem": {"problem_idx": problem_idx, "id": problem_id},
        "success": success,
        "attempts": [],
    }


def test_initial_state():
    """initial_state has all indices in remaining, zero generations."""
    state = initial_state(5)
    assert state.remaining == [0, 1, 2, 3, 4]
    assert state.total_generations == 0
    assert state.results == {}


def test_initial_state_from_indices():
    """initial_state_from_indices uses given list, sorted and deduplicated."""
    state = initial_state_from_indices([10, 2, 6, 2])
    assert state.remaining == [2, 6, 10]
    assert state.total_generations == 0
    assert state.results == {}


def test_extract_passed_indices_empty():
    """extract_passed_indices on empty list returns empty."""
    assert extract_passed_indices([]) == []


def test_extract_passed_indices_success_only():
    """extract_passed_indices returns only problem_idx where success is True."""
    logs = [
        _log_entry(0, "p0", True),
        _log_entry(1, "p1", False),
        _log_entry(2, "p2", True),
    ]
    assert extract_passed_indices(logs) == [0, 2]


def test_extract_passed_indices_missing_problem_skipped():
    """Entries without problem or problem_idx are skipped."""
    logs = [
        {"success": True},  # no problem
        {"problem": {"id": "x"}, "success": True},  # no problem_idx
        _log_entry(3, "p3", True),
    ]
    assert extract_passed_indices(logs) == [3]


def test_update_from_round_logs_pass_at_1():
    """One round pass@1: passers removed from remaining, attempts_used=1, total_generations updated."""
    state = initial_state(4)
    logs = [
        _log_entry(0, "p0", True),
        _log_entry(1, "p1", False),
        _log_entry(2, "p2", True),
        _log_entry(3, "p3", False),
    ]
    state.update_from_round_logs(logs, pass_k=1, round_index=0)
    assert state.remaining == [1, 3]
    assert state.total_generations == 4
    assert state.results[0].passed is True
    assert state.results[0].attempts_used == 1
    assert state.results[0].round_finished == 0
    assert state.results[1].passed is False
    assert state.results[1].attempts_used == 1
    assert state.results[2].passed is True
    assert state.results[3].passed is False


def test_update_from_round_logs_pass_at_k():
    """Round with pass_k=3: each problem gets 3 attempts_used, total_generations = n * pass_k."""
    state = initial_state(3)
    logs = [
        _log_entry(0, "p0", False),
        _log_entry(1, "p1", True),
        _log_entry(2, "p2", False),
    ]
    state.update_from_round_logs(logs, pass_k=3, round_index=0)
    assert state.remaining == [0, 2]
    assert state.total_generations == 9
    for idx in [0, 1, 2]:
        assert state.results[idx].attempts_used == 3
    assert state.results[1].passed is True


def test_update_from_round_logs_second_round_cumulative():
    """Second round adds to attempts_used and removes more passers."""
    state = initial_state(4)
    state.update_from_round_logs(
        [
            _log_entry(0, "p0", True),
            _log_entry(1, "p1", False),
            _log_entry(2, "p2", False),
            _log_entry(3, "p3", False),
        ],
        pass_k=1,
        round_index=0,
    )
    assert state.remaining == [1, 2, 3]
    assert state.total_generations == 4
    state.update_from_round_logs(
        [
            _log_entry(1, "p1", False),
            _log_entry(2, "p2", True),
            _log_entry(3, "p3", False),
        ],
        pass_k=2,
        round_index=1,
    )
    assert state.remaining == [1, 3]
    assert state.total_generations == 4 + 6  # 3 * 2
    assert state.results[1].attempts_used == 1 + 2
    assert state.results[2].attempts_used == 1 + 2
    assert state.results[2].passed is True
    assert state.results[2].round_finished == 1


def test_update_from_round_logs_subset_of_problems():
    """Round can contain only a subset of indices (e.g. remaining after previous round)."""
    state = RoundState(remaining=[1, 2], total_generations=2, results={
        0: ProblemResult(0, "p0", True, 1, 0),
    })
    logs = [
        _log_entry(1, "p1", True),
        _log_entry(2, "p2", False),
    ]
    state.update_from_round_logs(logs, pass_k=1, round_index=1)
    assert state.remaining == [2]
    assert state.total_generations == 2 + 2
    assert state.results[0].attempts_used == 1  # unchanged
    assert state.results[1].attempts_used == 1
    assert state.results[2].attempts_used == 1


def test_update_from_round_logs_id_from_log():
    """problem_id is taken from log problem.id."""
    state = initial_state(1)
    logs = [_log_entry(0, "mathd_algebra_478", True)]
    state.update_from_round_logs(logs, pass_k=1, round_index=0)
    assert state.results[0].problem_id == "mathd_algebra_478"
