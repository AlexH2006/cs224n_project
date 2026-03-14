"""
Merge four pass@8 runs into one pass@32 and plot Pass@k (k=1..32).

TL;DR:
  - Reads the four run dirs under a parent (e.g. parallel_samplin): each has logs.json
    (list of problem logs with "attempts" per problem, pass@8).
  - Merges in order: run0 attempts 1–8, run1 attempts 9–16, run2 17–24, run3 25–32.
  - For each problem, first_success_merged = minimum global attempt index (1–32) at
    which any of the four runs has a pass (going in run order).
  - Plots Pass@k vs k for k=1..32 and saves to a separate file (e.g. pass_at_k_32_merged.png)
  so it does not overwrite the existing pass_at_k_50sample.png.

Usage:
  python scripts/pass_at_k_parallel_merged_pass32.py results/Qwen3.5_4B_sampled_no_thinking_parallel_samplin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Pass criterion (same as plot_pass_at_k_multi_run)
# -----------------------------------------------------------------------------


def is_pass(attempt: dict[str, Any]) -> bool:
    """True iff this attempt counts as a pass: success, complete, no sorry."""
    verification = attempt.get("verification") or {}
    if not verification:
        return bool(attempt.get("success"))
    return (
        verification.get("success", False)
        and verification.get("complete", False)
        and not verification.get("has_sorry", True)
    )


def first_success_attempt_in_run(attempts: list[dict[str, Any]]) -> int | None:
    """1-based attempt index of first pass in this run, or None if never."""
    for i, a in enumerate(attempts):
        if is_pass(a):
            return i + 1
    return None


# -----------------------------------------------------------------------------
# Load runs
# -----------------------------------------------------------------------------


def load_run_logs(run_dir: Path) -> list[dict[str, Any]]:
    """Load logs.json from a run directory (list of problem logs)."""
    path = run_dir / "logs.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of problem logs in {path}")
    return data


def collect_run_dirs(parent_dir: Path) -> list[Path]:
    """Return run directories (immediate children) that contain logs.json, sorted."""
    if not parent_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {parent_dir}")
    run_dirs = []
    for child in sorted(parent_dir.iterdir()):
        if child.is_dir() and (child / "logs.json").exists():
            run_dirs.append(child)
    return run_dirs


# -----------------------------------------------------------------------------
# Merge four pass@8 runs into one pass@32
# -----------------------------------------------------------------------------

ATTEMPTS_PER_RUN = 8


def merged_first_success_for_problem(
    problem_logs_across_runs: list[dict[str, Any]],
) -> int | None:
    """
    Given one problem's log from each run (same order: run0, run1, run2, run3),
    return the minimum global attempt index (1..32) at which any run has a pass,
    or None if no run ever passes.

    Global attempt index: run_i contributes attempts (run_i*8+1) .. (run_i*8+8).
    """
    best: int | None = None
    for run_index, log in enumerate(problem_logs_across_runs):
        attempts = log.get("attempts") or []
        first_in_run = first_success_attempt_in_run(attempts)
        if first_in_run is not None:
            global_attempt = run_index * ATTEMPTS_PER_RUN + first_in_run
            if best is None or global_attempt < best:
                best = global_attempt
    return best


def compute_merged_first_success(
    run_dirs: list[Path],
) -> list[tuple[int, str, int | None]]:
    """
    Load all runs, align by list index (runs must have same 50 problems in same order),
    and return list of (problem_idx, problem_id, merged_first_success).
    """
    runs_data = [load_run_logs(d) for d in run_dirs]
    n_problems = len(runs_data[0])
    for i, data in enumerate(runs_data):
        if len(data) != n_problems:
            raise ValueError(
                f"Run {run_dirs[i].name} has {len(data)} problems, expected {n_problems}"
            )

    records = []
    for p in range(n_problems):
        problem_logs = [run_data[p] for run_data in runs_data]
        problem = problem_logs[0].get("problem") or {}
        problem_idx = problem.get("problem_idx", p)
        problem_id = problem.get("id") or problem.get("problem_id") or f"problem_{problem_idx}"
        first = merged_first_success_for_problem(problem_logs)
        records.append((problem_idx, problem_id, first))
    return records


# -----------------------------------------------------------------------------
# Pass@k and plot
# -----------------------------------------------------------------------------


def compute_pass_at_k(
    records: list[tuple[int, str, int | None]],
    max_k: int = 32,
) -> tuple[list[int], list[float]]:
    """Pass@k for k=1..max_k. Returns (ks, pct_list)."""
    first_success = [r[2] for r in records]
    n = len(first_success)
    ks = list(range(1, max_k + 1))
    pcts = [100.0 * sum(1 for a in first_success if a is not None and a <= k) / n for k in ks]
    return ks, pcts


def plot_pass_at_k_32(
    parent_dir: Path,
    output_path: Path | None = None,
    max_k: int = 32,
    n_problems: int = 50,
) -> Path:
    """Compute merged pass@32, plot, and save to a separate file (not pass_at_k_50sample.png)."""
    import matplotlib.pyplot as plt

    run_dirs = collect_run_dirs(parent_dir)
    if len(run_dirs) != 4:
        raise ValueError(
            f"Expected exactly 4 run directories with logs.json, found {len(run_dirs)}"
        )

    records = compute_merged_first_success(run_dirs)
    n_actual = len(records)
    if n_problems is None:
        n_problems = n_actual

    ks, pcts = compute_pass_at_k(records, max_k=max_k)

    fig, ax = plt.subplots()
    ax.plot(ks, pcts, marker="o", linestyle="-", markersize=6, color="C0")
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title(f"Pass@k (n={n_problems} problems, merged pass@32)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    if output_path is None:
        output_path = parent_dir / "pass_at_k_32_merged.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge four pass@8 runs into one pass@32 and plot Pass@k.",
    )
    parser.add_argument(
        "parent_dir",
        type=Path,
        help="Directory containing the four run_* subdirs (each with logs.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <parent_dir>/pass_at_k_32_merged.png)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=32,
        help="Maximum k (default: 32)",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=50,
        help="Number of problems for title (default: 50)",
    )
    args = parser.parse_args()

    parent_dir = args.parent_dir.resolve()
    if not parent_dir.is_dir():
        raise SystemExit(f"Not a directory: {parent_dir}")

    out = plot_pass_at_k_32(
        parent_dir,
        output_path=args.output,
        max_k=args.max_k,
        n_problems=args.n_problems,
    )
    print(f"Saved merged pass@32 plot → {out}")


if __name__ == "__main__":
    main()
