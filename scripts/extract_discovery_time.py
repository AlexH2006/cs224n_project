"""
Extract average discovery time from a pass@K run (single problem).

Discovery time is estimated as: total_attempts / proofs_passed (same as problem 119 summary).
Pass = verification success AND complete AND no sorry.

Usage:
    python scripts/extract_discovery_time.py baseline/run_Qwen3.5-4B_20260312_XXXXXX
    python scripts/extract_discovery_time.py baseline/run_Qwen3.5-4B_20260312_XXXXXX -o results/Qwen3.5_4B_discovery_time/problem_53/summary_problem_53.json
"""

from __future__ import annotations

import argparse
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


def extract_discovery_time(run_dir: Path, problem_idx: int | None = None) -> dict:
    """
    Load logs.json from run_dir, find the single problem (or problem_idx if given),
    count passes and compute estimated discovery time = total_attempts / proofs_passed.
    """
    logs_path = run_dir / "logs.json"
    if not logs_path.exists():
        raise FileNotFoundError(f"Missing: {logs_path}")

    with open(logs_path, encoding="utf-8") as f:
        logs = json.load(f)

    if not logs:
        raise ValueError("Empty logs")

    # If multiple problems, use first; else filter by problem_idx if given
    if problem_idx is not None:
        logs = [log for log in logs if log.get("problem", {}).get("problem_idx") == problem_idx]
    if not logs:
        raise ValueError(f"No problem with problem_idx={problem_idx} in logs")

    log = logs[0]
    problem = log.get("problem", {}) or {}
    attempts = log.get("attempts", [])
    total_attempts = len(attempts)
    proofs_passed = sum(1 for a in attempts if is_pass(a))

    if proofs_passed == 0:
        estimated = None
    else:
        estimated = round(total_attempts / proofs_passed, 2)

    first_pass_step = None
    for i, a in enumerate(attempts):
        if is_pass(a):
            first_pass_step = i + 1
            break

    return {
        "problem_idx": problem.get("problem_idx"),
        "problem_id": problem.get("id", ""),
        "total_attempts": total_attempts,
        "proofs_passed": proofs_passed,
        "first_pass_at_step": first_pass_step,
        "estimated_discovery_time": estimated,
        "note": "Estimated discovery time = total_attempts / proofs_passed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract average (estimated) discovery time from a single-problem pass@K run"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing logs.json (e.g. baseline/run_Qwen3.5-4B_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--problem-idx",
        type=int,
        default=None,
        help="Problem index to use if logs contain multiple problems (default: use only problem)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write summary JSON here (default: print to stdout)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    summary = extract_discovery_time(run_dir, problem_idx=args.problem_idx)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved → {args.output}")
    else:
        print(json.dumps(summary, indent=2))
    print(f"  Average (estimated) discovery time: {summary['estimated_discovery_time']}")


if __name__ == "__main__":
    main()
