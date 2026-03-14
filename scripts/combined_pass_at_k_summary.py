#!/usr/bin/env python3
"""
Build a combined pass@k summary:
- Baseline (Qwen3.5-4B_no_thinking/logs.json): record k at which each problem first passed (1..8 or 9).
- Dynamic sampling (pass_at_k_summary.json): record (k+8) for problems solved at k there (9..22; unsolved=23).
- For problems in both, use the minimum (earliest pass).
- Write summary to dynamic_sampling_results/ with pass@k curve for k=1..22.
"""

from __future__ import annotations

import json
from pathlib import Path


def is_pass(attempt: dict) -> bool:
    verification = attempt.get("verification") or {}
    if not verification:
        return bool(attempt.get("success"))
    return (
        verification.get("success", False)
        and verification.get("complete", False)
        and not verification.get("has_sorry", True)
    )


BASELINE_PASS_K = 8
UNSOLVED_SENTINEL = 23  # > 22 so not counted as solved in combined pass@22


def load_baseline_first_success(logs_path: Path) -> dict[int, int]:
    """problem_idx -> first_success_attempt (1..8 or UNSOLVED_SENTINEL)."""
    with open(logs_path, encoding="utf-8") as f:
        logs = json.load(f)
    out = {}
    for log in logs:
        problem = log.get("problem", {}) or {}
        idx = problem.get("problem_idx")
        if idx is None:
            continue
        attempts = log.get("attempts", [])
        first = UNSOLVED_SENTINEL
        for i, a in enumerate(attempts):
            if is_pass(a):
                first = i + 1
                break
        out[idx] = first
    return out


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    baseline_logs = repo / "results" / "Qwen3.5-4B_no_thinking" / "logs.json"
    dynamic_summary = repo / "dynamic_sampling_results" / "run_Qwen3.5-4B_20260312_030211" / "pass_at_k_summary.json"
    out_dir = repo / "dynamic_sampling_results"

    if not baseline_logs.is_file():
        raise SystemExit(f"Baseline logs not found: {baseline_logs}")
    if not dynamic_summary.is_file():
        raise SystemExit(f"Dynamic summary not found: {dynamic_summary}")

    baseline_first = load_baseline_first_success(baseline_logs)

    with open(dynamic_summary, encoding="utf-8") as f:
        dyn = json.load(f)
    dyn_pass_k = dyn["pass_k"]
    # Map dynamic first_success_attempt to combined attempt: k -> k+8 (solved), else 23
    dynamic_mapped: dict[int, int] = {}
    problem_id_from_dyn: dict[int, str] = {}
    for p in dyn["problems"]:
        idx = p["problem_idx"]
        k = p["first_success_attempt"]
        problem_id_from_dyn[idx] = p.get("problem_id", "")
        if k <= dyn_pass_k:
            dynamic_mapped[idx] = k + 8
        else:
            dynamic_mapped[idx] = UNSOLVED_SENTINEL

    all_indices = sorted(set(baseline_first) | set(dynamic_mapped))
    # Get problem_id from baseline for problems only there
    with open(baseline_logs, encoding="utf-8") as f:
        baseline_logs_list = json.load(f)
    problem_id_from_baseline: dict[int, str] = {}
    for log in baseline_logs_list:
        problem = log.get("problem", {}) or {}
        idx = problem.get("problem_idx")
        if idx is not None:
            problem_id_from_baseline[idx] = problem.get("id", problem.get("problem_id", ""))

    combined_first: dict[int, int] = {}
    problems_out = []
    for idx in all_indices:
        b = baseline_first.get(idx, UNSOLVED_SENTINEL)
        d = dynamic_mapped.get(idx, UNSOLVED_SENTINEL)
        first = min(b, d)
        combined_first[idx] = first
        problem_id = problem_id_from_dyn.get(idx) or problem_id_from_baseline.get(idx, "")
        problems_out.append({
            "problem_idx": idx,
            "problem_id": problem_id,
            "first_success_attempt": first,
        })

    max_k = 22
    n_problems = len(all_indices)
    first_success_values = [combined_first[i] for i in all_indices]
    pass_at_k = []
    for k in range(1, max_k + 1):
        pct = 100.0 * sum(1 for v in first_success_values if v <= k) / n_problems
        pass_at_k.append({"k": k, "problems_solved_pct": round(pct, 4)})
    n_solved = sum(1 for v in first_success_values if v <= max_k)

    summary = {
        "source_baseline_logs": str(baseline_logs.resolve()),
        "source_dynamic_summary": str(dynamic_summary.resolve()),
        "description": "Combined: baseline attempts 1-8, dynamic sampling attempts 9-22 (k+8). First pass = min of the two.",
        "pass_k": max_k,
        "n_problems": n_problems,
        "n_solved": n_solved,
        "pass_at_k": pass_at_k,
        "problems": problems_out,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "combined_pass_at_k_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path} (n_problems={n_problems}, n_solved={n_solved}, pass_k=1..{max_k})", flush=True)


if __name__ == "__main__":
    main()
