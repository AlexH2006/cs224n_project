"""
TLDR: Round state for dynamic_sampling (remaining indices, total generations, per-problem results).

Updated after each round from qwen_eval logs.json. Invoker and output do not parse logs; only this module does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _problem_passed(log: dict[str, Any]) -> bool:
    """
    True iff the problem has at least one attempt that is success, complete, and does not contain sorry.

    Matches qwen_eval's notion of a passing proof (verification.success and complete and not has_sorry).
    If the log has no attempts with verification (e.g. minimal test logs), falls back to top-level
    success, complete (default True), and not has_sorry (default False).
    """
    for att in log.get("attempts") or []:
        v = att.get("verification") or {}
        if v.get("success") and v.get("complete") and not v.get("has_sorry"):
            return True
    # Fallback for logs without per-attempt verification (e.g. tests)
    return (
        bool(log.get("success"))
        and log.get("complete", True)
        and not log.get("has_sorry", False)
    )


@dataclass
class ProblemResult:
    """Result for a single problem: passed or not, total attempts used, round when last attempted or passed."""

    problem_idx: int
    problem_id: str
    passed: bool
    attempts_used: int
    round_finished: int  # 0-based round index when we last ran this problem (or when it passed)


@dataclass
class RoundState:
    """
    State across rounds: which problems remain, total generations so far, per-problem results.

    remaining: problem indices still to evaluate (not yet passed).
    total_generations: sum of all attempts so far (counts toward budget).
    results: problem_idx -> ProblemResult for every problem we have seen (passed or not).
    """

    remaining: list[int] = field(default_factory=list)
    total_generations: int = 0
    results: dict[int, ProblemResult] = field(default_factory=dict)

    def update_from_round_logs(
        self,
        problem_logs: list[dict[str, Any]],
        pass_k: int,
        round_index: int,
    ) -> None:
        """
        Update state from one round's logs.json content.

        Args:
            problem_logs: List of per-problem log dicts (each with "problem", "success", "attempts").
            pass_k: Number of attempts per problem in this round.
            round_index: 0-based round number.
        """
        remaining_set = set(self.remaining)
        for log in problem_logs:
            problem = log.get("problem") or {}
            problem_idx = problem.get("problem_idx")
            problem_id = problem.get("id", str(problem_idx))
            if problem_idx is None:
                continue
            passed = _problem_passed(log)
            if problem_idx not in self.results:
                self.results[problem_idx] = ProblemResult(
                    problem_idx=problem_idx,
                    problem_id=problem_id,
                    passed=False,
                    attempts_used=0,
                    round_finished=-1,
                )
            rec = self.results[problem_idx]
            rec.attempts_used += pass_k
            rec.round_finished = round_index
            if passed:
                rec.passed = True
                remaining_set.discard(problem_idx)
        self.remaining = sorted(remaining_set)
        n_problems_this_round = len(problem_logs)
        self.total_generations += n_problems_this_round * pass_k


def initial_state(n_problems: int) -> RoundState:
    """Build initial state: all indices in remaining, zero generations, no results."""
    return RoundState(
        remaining=list(range(n_problems)),
        total_generations=0,
        results={},
    )


def initial_state_from_indices(indices: list[int]) -> RoundState:
    """Build initial state from an explicit list of problem indices (e.g. from problem_idx.json)."""
    return RoundState(
        remaining=sorted(set(indices)),
        total_generations=0,
        results={},
    )


def extract_passed_indices(problem_logs: list[dict[str, Any]]) -> list[int]:
    """
    Return list of problem_idx for which the problem passed (success, complete, no sorry).
    Useful for tests and for sanity checks; runner uses RoundState.update_from_round_logs.
    """
    out: list[int] = []
    for log in problem_logs:
        if not _problem_passed(log):
            continue
        idx = (log.get("problem") or {}).get("problem_idx")
        if idx is not None:
            out.append(idx)
    return out
