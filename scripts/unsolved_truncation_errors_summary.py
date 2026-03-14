#!/usr/bin/env python3
"""
For unsolved problems only: compute per-problem stats from baseline + round_1 + round_2 logs (32 attempts total).
Output JSON in dynamic_sampling_results with:
- truncation_rate = (proof truncation count) / (total attempts)
- errors_per_non_truncated_proof = (total error messages) / (number of non-truncated attempts).
"""

from __future__ import annotations

import json
from pathlib import Path


def is_pass(attempt: dict) -> bool:
    v = attempt.get("verification") or {}
    if not v:
        return bool(attempt.get("success"))
    return (
        v.get("success", False)
        and v.get("complete", False)
        and not v.get("has_sorry", True)
    )


def n_errors(attempt: dict) -> int:
    v = attempt.get("verification") or {}
    return len(v.get("errors") or [])


def load_attempts_by_problem(logs_path: Path) -> dict[int, list[dict]]:
    """problem_idx -> list of attempt dicts."""
    with open(logs_path, encoding="utf-8") as f:
        logs = json.load(f)
    out: dict[int, list[dict]] = {}
    for log in logs:
        problem = log.get("problem", {}) or {}
        idx = problem.get("problem_idx")
        if idx is None:
            continue
        attempts = log.get("attempts", [])
        out.setdefault(idx, []).extend(attempts)
    return out


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    unsolved_path = repo / "qwen_eval" / "problem_idx.json"
    baseline_logs = repo / "results" / "Qwen3.5-4B_no_thinking" / "logs.json"
    round1_logs = repo / "dynamic_sampling_results" / "round_1" / "round_0" / "run_Qwen3.5-4B_20260312_030216" / "logs.json"
    round2_logs = repo / "dynamic_sampling_results" / "round_2" / "logs.json"
    out_path = repo / "dynamic_sampling_results" / "unsolved_truncation_errors_summary.json"

    with open(unsolved_path, encoding="utf-8") as f:
        unsolved_data = json.load(f)
    unsolved_indices = set(unsolved_data["problem_indices"])

    def merge_attempts(idx: int, *sources: dict[int, list[dict]]) -> list[dict]:
        acc = []
        for d in sources:
            acc.extend(d.get(idx, []))
        return acc

    baseline = load_attempts_by_problem(baseline_logs) if baseline_logs.is_file() else {}
    round1 = load_attempts_by_problem(round1_logs) if round1_logs.is_file() else {}
    round2 = load_attempts_by_problem(round2_logs) if round2_logs.is_file() else {}

    problems_out = []
    for idx in sorted(unsolved_indices):
        attempts = merge_attempts(idx, baseline, round1, round2)
        total = len(attempts)
        n_truncated = sum(1 for a in attempts if a.get("truncated") is True)
        n_non_truncated = total - n_truncated
        truncation_rate = round(n_truncated / total, 4) if total else 0.0
        total_errors = sum(n_errors(a) for a in attempts)
        errors_per_non_truncated_proof = round(total_errors / n_non_truncated, 4) if n_non_truncated else None
        n_successful = sum(1 for a in attempts if is_pass(a))
        n_unsuccessful = total - n_successful
        success_rate = round(n_successful / total, 4) if total else 0.0
        problems_out.append({
            "problem_idx": idx,
            "total_attempts": total,
            "n_successful": n_successful,
            "n_unsuccessful": n_unsuccessful,
            "success_rate": success_rate,
            "n_truncated": n_truncated,
            "n_non_truncated": n_non_truncated,
            "truncation_rate": truncation_rate,
            "total_error_messages": total_errors,
            "errors_per_non_truncated_proof": errors_per_non_truncated_proof,
        })

    # Sort by errors_per_non_truncated_proof (asc) then n_truncated (asc); nulls last
    def sort_key(p: dict) -> tuple[float, int]:
        e = p.get("errors_per_non_truncated_proof")
        return (e if e is not None else float("inf"), p["n_truncated"])

    problems_out.sort(key=sort_key)

    summary = {
        "description": "Per-unsolved-problem stats from baseline (8) + round_1 (14) + round_2 (10) = 32 attempts.",
        "sources": [
            str(baseline_logs.resolve()) if baseline_logs.is_file() else None,
            str(round1_logs.resolve()) if round1_logs.is_file() else None,
            str(round2_logs.resolve()) if round2_logs.is_file() else None,
        ],
        "n_problems": len(problems_out),
        "problems": problems_out,
    }
    summary["sources"] = [s for s in summary["sources"] if s]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path} (n={len(problems_out)} unsolved problems)", flush=True)


if __name__ == "__main__":
    main()
