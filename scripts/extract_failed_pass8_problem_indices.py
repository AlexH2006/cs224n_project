#!/usr/bin/env python3
"""
Extract problem_idx for problems that need more than k trials to pass (default k=8), or are
not solved in the log at all.

Includes: (1) problems that never pass in the log, (2) problems whose first pass is at
attempt k+1 or later (need more than k trials to pass). Uses the same pass criterion as
scripts/plot_pass_at_k_multi_run.py.

Output: dynamic_sampling/problem_idx.json with {"problem_indices": [...]} (sorted).

Usage (from project root):
    python scripts/extract_failed_pass8_problem_indices.py results/Qwen3.5-4B_no_thinking
    python scripts/extract_failed_pass8_problem_indices.py results/run_Qwen3.5-4B_no_think_mode/logs.json
    # Include unevaluated indices (e.g. 32-243) when logs only cover a subset:
    python scripts/extract_failed_pass8_problem_indices.py baseline/test/run_xxx --total-problems 244 --output dynamic_sampling/problem_idx.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# -----------------------------------------------------------------------------
# Pass criterion (same as plot_pass_at_k_multi_run.py)
# -----------------------------------------------------------------------------


def is_pass(attempt: dict) -> bool:
    """
    True iff this attempt counts as a pass: success, complete, and contains no sorry.

    Uses verification.success, verification.complete, verification.has_sorry when
    present; otherwise falls back to top-level "success".
    """
    verification = attempt.get("verification") or {}
    if not verification:
        return bool(attempt.get("success"))
    return (
        verification.get("success", False)
        and verification.get("complete", False)
        and not verification.get("has_sorry", True)
    )


# -----------------------------------------------------------------------------
# Resolve logs path
# -----------------------------------------------------------------------------


def resolve_logs_path(path: Path) -> Path:
    """
    Resolve path to a single logs.json file.

    - If path is a file and name is logs.json, return it.
    - If path is a directory and contains logs.json, return path/logs.json.
    - If path is a directory of run subdirs (each with logs.json), return the first run's logs.json.
    """
    path = path.resolve()
    if path.is_file():
        if path.name == "logs.json":
            return path
        raise ValueError(f"Expected logs.json file or directory, got file: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Not found: {path}")

    direct = path / "logs.json"
    if direct.exists():
        return direct

    run_dirs = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "logs.json").exists():
            run_dirs.append(child)
    if not run_dirs:
        raise FileNotFoundError(f"No logs.json in {path} or in any subdirectory")
    return run_dirs[0] / "logs.json"


def load_problem_logs(logs_path: Path) -> list[dict]:
    """Load and validate logs.json. Returns list of per-problem log dicts."""
    with open(logs_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of problem logs in {logs_path}")
    return data


# -----------------------------------------------------------------------------
# Compute problem indices that need more than k trials to pass or are not solved
# -----------------------------------------------------------------------------


def problem_indices_need_more_than_k_trials_or_unsolved(
    problem_logs: list[dict],
    max_k: int = 8,
    total_problems: int | None = None,
) -> list[int]:
    """
    Return sorted list of problem_idx for problems that either:
    - never pass in the log (not solved), or
    - first pass is at attempt max_k+1 or later (need more than max_k trials to pass).

    If total_problems is set (e.g. 244 for full MiniF2F test), also include every
    index in [0, total_problems) that does not appear in the logs (unevaluated,
    treated as needing attention).
    """
    included = []
    indices_in_logs = set()
    for log in problem_logs:
        problem = log.get("problem") or {}
        problem_idx = problem.get("problem_idx")
        if problem_idx is None:
            continue
        indices_in_logs.add(problem_idx)
        attempts = log.get("attempts", [])
        first_pass_index = None
        for i, a in enumerate(attempts):
            if is_pass(a):
                first_pass_index = i
                break
        if first_pass_index is None:
            included.append(problem_idx)
        elif first_pass_index >= max_k:
            included.append(problem_idx)

    if total_problems is not None and total_problems > 0:
        for idx in range(total_problems):
            if idx not in indices_in_logs:
                included.append(idx)

    return sorted(set(included))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract problem_idx of problems that need more than k trials to pass or are not solved; write to dynamic_sampling/problem_idx.json."
    )
    parser.add_argument(
        "logs_path",
        type=Path,
        help="Path to logs.json or to a directory containing logs.json (or run subdirs with logs.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: dynamic_sampling/problem_idx.json)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=8,
        help="Consider first k attempts per problem (default: 8)",
    )
    parser.add_argument(
        "--total-problems",
        type=int,
        default=None,
        metavar="N",
        help="Full dataset size (e.g. 244 for MiniF2F test). Include indices in [0, N) not in the logs as unevaluated.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_path = args.output
    if output_path is None:
        output_path = repo_root / "dynamic_sampling" / "problem_idx.json"
    else:
        output_path = Path(output_path).resolve()

    logs_file = resolve_logs_path(args.logs_path)
    problem_logs = load_problem_logs(logs_file)
    indices = problem_indices_need_more_than_k_trials_or_unsolved(
        problem_logs,
        max_k=args.max_k,
        total_problems=args.total_problems,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"problem_indices": indices}, f, indent=2)

    print(f"Loaded {len(problem_logs)} problems from {logs_file}")
    if args.total_problems is not None:
        in_logs = {(log.get("problem") or {}).get("problem_idx") for log in problem_logs} - {None}
        n_uneval = max(0, args.total_problems - len(in_logs))
        print(f"Problems that need >{args.max_k} trials or unsolved (incl. {n_uneval} unevaluated): {len(indices)}")
    else:
        print(f"Problems that need >{args.max_k} trials or unsolved: {len(indices)}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
