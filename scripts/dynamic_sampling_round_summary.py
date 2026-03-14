#!/usr/bin/env python3
"""
From dynamic_sampling round logs.json (array of { problem, attempts }), produce a summary
with truncation rates and error counts per problem, in the same format as interesting_problem_idx.json.
"""
from __future__ import annotations

import argparse
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

    # Sort by problem_idx
    problems_out.sort(key=lambda p: (p["problem_idx"] if p["problem_idx"] is not None else -1))
    return problems_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize dynamic_sampling round logs: truncation + error stats per problem"
    )
    parser.add_argument(
        "logs_json",
        type=Path,
        help="Path to round logs.json (e.g. dynamic_sampling_results/round_2/logs.json)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: same dir as logs, round_summary.json)",
    )
    args = parser.parse_args()

    logs_path = Path(args.logs_json).resolve()
    if not logs_path.is_file():
        raise SystemExit(f"Not a file: {logs_path}")

    problems = summarize_logs(logs_path)
    out = {
        "description": "Per-problem stats from round logs: truncation_rate, total_error_messages, errors_per_non_truncated_proof (same format as interesting_problem_idx.json).",
        "source": logs_path.name,
        "problems": problems,
    }

    out_path = Path(args.output).resolve() if args.output else logs_path.parent / "round_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path} ({len(problems)} problems)", flush=True)


if __name__ == "__main__":
    main()
