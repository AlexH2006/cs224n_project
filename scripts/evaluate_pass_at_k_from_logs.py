#!/usr/bin/env python3
"""
Evaluate pass@k from a single logs.json (list of problem logs with attempts).
Output: summary JSON with problem_idx, problem_id, first_success_attempt, and pass@k curve.

Usage:
    python scripts/evaluate_pass_at_k_from_logs.py \\
        --logs path/to/logs.json \\
        --output-dir path/to/output_dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_pass(attempt: dict) -> bool:
    """True iff attempt counts as a pass (success, complete, no sorry)."""
    verification = attempt.get("verification") or {}
    if not verification:
        return bool(attempt.get("success"))
    return (
        verification.get("success", False)
        and verification.get("complete", False)
        and not verification.get("has_sorry", True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pass@k from logs.json; write summary with first_success_attempt per problem"
    )
    parser.add_argument(
        "--logs",
        type=Path,
        required=True,
        help="Path to logs.json (list of problem logs with attempts)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write pass_at_k_summary.json",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        default=None,
        help="Max k for pass@k (default: infer from max attempts in logs)",
    )
    args = parser.parse_args()

    logs_path = Path(args.logs)
    if not logs_path.is_file():
        raise SystemExit(f"Logs not found: {logs_path}")

    with open(logs_path, encoding="utf-8") as f:
        problem_logs = json.load(f)
    if not isinstance(problem_logs, list):
        raise SystemExit("Expected logs.json to be a list of problem logs")

    max_attempts = args.pass_k
    if max_attempts is None:
        max_attempts = max(len(log.get("attempts", [])) for log in problem_logs) if problem_logs else 0

    problems_out = []
    first_success_values = []

    for log in problem_logs:
        problem = log.get("problem", {}) or {}
        problem_idx = problem.get("problem_idx")
        problem_id = problem.get("id", problem.get("problem_id", ""))
        attempts = log.get("attempts", [])

        first_success_attempt = len(attempts) + 1  # 1-based; never = n+1
        for i, a in enumerate(attempts):
            if is_pass(a):
                first_success_attempt = i + 1
                break

        problems_out.append({
            "problem_idx": problem_idx,
            "problem_id": problem_id,
            "first_success_attempt": first_success_attempt,
        })
        first_success_values.append(first_success_attempt)
    # Cap first_success for pass@k at max_attempts+1
    sentinel = max_attempts + 1
    first_success_capped = [min(v, sentinel) for v in first_success_values]

    n_problems = len(problem_logs)
    pass_at_k = []
    for k in range(1, max_attempts + 1):
        pct = 100.0 * sum(1 for v in first_success_capped if v <= k) / n_problems if n_problems else 0.0
        pass_at_k.append({"k": k, "problems_solved_pct": round(pct, 4)})

    n_solved = sum(1 for v in first_success_capped if v <= max_attempts)
    summary = {
        "source_logs": str(logs_path.resolve()),
        "pass_k": max_attempts,
        "n_problems": n_problems,
        "n_solved": n_solved,
        "pass_at_k": pass_at_k,
        "problems": problems_out,
    }

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pass_at_k_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    msg = f"Wrote {out_path} (n={n_problems}, pass_k={max_attempts}, n_solved={n_solved})"
    print(msg, flush=True)
