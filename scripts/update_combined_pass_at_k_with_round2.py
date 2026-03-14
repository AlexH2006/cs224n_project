#!/usr/bin/env python3
"""
Update dynamic_sampling_results/combined_pass_at_k_summary.json with round_2 results.
If a problem is solved at attempt k in round_2, set combined first_success_attempt = min(current, 22+k).
Extend pass_k to 32 (22+10) and recompute pass@k curve.
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


def load_first_success_from_logs(logs_path: Path, pass_k: int) -> dict[int, int]:
    """problem_idx -> first_success_attempt (1..pass_k or pass_k+1)."""
    with open(logs_path, encoding="utf-8") as f:
        logs = json.load(f)
    out = {}
    for log in logs:
        problem = log.get("problem", {}) or {}
        idx = problem.get("problem_idx")
        if idx is None:
            continue
        attempts = log.get("attempts", [])
        first = pass_k + 1
        for i, a in enumerate(attempts):
            if is_pass(a):
                first = i + 1
                break
        out[idx] = first
    return out


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    combined_path = repo / "dynamic_sampling_results" / "combined_pass_at_k_summary.json"
    round2_logs = repo / "dynamic_sampling_results" / "round_2" / "logs.json"

    if not combined_path.is_file():
        raise SystemExit(f"Combined summary not found: {combined_path}")
    if not round2_logs.is_file():
        raise SystemExit(f"Round 2 logs not found: {round2_logs}")

    with open(combined_path, encoding="utf-8") as f:
        combined = json.load(f)

    ROUND2_PASS_K = 10
    round2_first = load_first_success_from_logs(round2_logs, ROUND2_PASS_K)

    MAX_K = 32
    UNSOLVED = 33
    problems_out = combined["problems"]
    all_indices = sorted(p["problem_idx"] for p in problems_out)
    combined_first = {p["problem_idx"]: p["first_success_attempt"] for p in problems_out}
    # Old unsolved sentinel was 23; treat as 33 so they are not counted as solved in pass@32
    for idx in combined_first:
        if combined_first[idx] > 22:
            combined_first[idx] = UNSOLVED

    for idx, k in round2_first.items():
        if k <= ROUND2_PASS_K:
            new_val = 22 + k
            combined_first[idx] = min(combined_first.get(idx, UNSOLVED), new_val)
    for p in problems_out:
        p["first_success_attempt"] = combined_first[p["problem_idx"]]

    first_success_values = [combined_first[i] for i in all_indices]
    pass_at_k = []
    for k in range(1, MAX_K + 1):
        pct = 100.0 * sum(1 for v in first_success_values if v <= k) / len(all_indices)
        pass_at_k.append({"k": k, "problems_solved_pct": round(pct, 4)})
    n_solved = sum(1 for v in first_success_values if v <= MAX_K)

    combined["source_round2_logs"] = str(round2_logs.resolve())
    combined["description"] = "Combined: baseline 1-8, dynamic round1 9-22 (k+8), round2 23-32 (22+k). First pass = min."
    combined["pass_k"] = MAX_K
    combined["n_solved"] = n_solved
    combined["pass_at_k"] = pass_at_k
    combined["problems"] = problems_out

    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"Updated {combined_path} (n_problems={len(all_indices)}, n_solved={n_solved}, pass_k=1..{MAX_K})", flush=True)


if __name__ == "__main__":
    main()
