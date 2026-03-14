#!/usr/bin/env python3
"""
Generate a summary in the same format as interesting_problem_idx.json for all
problems in dynamic_sampling_results/imo_round (one problem per problem{N}/logs.json).
Reads each imo_round/problem*/logs.json, computes per-problem stats (success_rate,
truncation_rate, total_error_messages, etc.), and writes imo_round/interesting_problem_idx.json.
"""
from __future__ import annotations

import json
from pathlib import Path


def is_pass(attempt: dict) -> bool:
    v = attempt.get("verification") or {}
    return (
        v.get("success", False)
        and v.get("complete", False)
        and not v.get("has_sorry", True)
    )


def summarize_logs(logs_path: Path) -> list[dict]:
    """Summarize one logs.json (list of problem entries) into interesting_problem_idx format."""
    with open(logs_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("logs.json should be a list of problem entries")

    problems_out = []
    for entry in data:
        problem = entry.get("problem") or {}
        problem_id = problem.get("id") or problem.get("problem_id") or ""
        problem_idx = problem.get("problem_idx")
        attempts = entry.get("attempts") or []

        total = len(attempts)
        n_successful = sum(1 for a in attempts if is_pass(a))
        n_truncated = sum(1 for a in attempts if a.get("truncated"))
        n_non_truncated = total - n_truncated
        truncation_rate = round(n_truncated / total, 4) if total else 0
        success_rate = round(n_successful / total, 4) if total else 0

        total_error_messages = 0
        for a in attempts:
            v = a.get("verification") or {}
            total_error_messages += len(v.get("errors") or [])

        errors_per_non_truncated = (
            round(total_error_messages / n_non_truncated, 4)
            if n_non_truncated
            else 0
        )

        problems_out.append({
            "problem_idx": problem_idx,
            "problem_id": problem_id,
            "total_attempts": total,
            "n_successful": n_successful,
            "success_rate": success_rate,
            "n_truncated": n_truncated,
            "n_non_truncated": n_non_truncated,
            "truncation_rate": truncation_rate,
            "total_error_messages": total_error_messages,
            "errors_per_non_truncated_proof": errors_per_non_truncated,
        })

    return problems_out


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    imo_round_dir = repo / "dynamic_sampling_results" / "imo_round"
    out_path = imo_round_dir / "interesting_problem_idx.json"

    if not imo_round_dir.is_dir():
        raise SystemExit(f"Not a directory: {imo_round_dir}")

    # Find all problem*/logs.json
    all_problems = []
    for log_path in sorted(imo_round_dir.glob("problem*/logs.json")):
        try:
            problems = summarize_logs(log_path)
            all_problems.extend(problems)
        except Exception as e:
            raise SystemExit(f"Error processing {log_path}: {e}") from e

    all_problems.sort(key=lambda p: (p["problem_idx"] if p["problem_idx"] is not None else -1))

    out = {"problems": all_problems}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path} ({len(all_problems)} problems)", flush=True)


if __name__ == "__main__":
    main()
