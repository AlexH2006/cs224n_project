"""
Plot average Pass@k (± std) across multiple pass@8 runs in a folder.

Each subfolder of the given directory is treated as one run (must contain logs.json).
Pass = verification success AND complete AND no sorry. For each k in 1..8 we compute
the fraction of problems solved (at least one pass in the first k attempts, chronological).
Then we average and take std across runs and plot with error bars.

Usage:
    python scripts/plot_pass_at_k_multi_run.py results/Qwen3.5_4B__no_thinking_sampled
    python scripts/plot_pass_at_k_multi_run.py results/Qwen3.5_4B__no_thinking_sampled -o results/Qwen3.5_4B__no_thinking_sampled/pass_at_k_50sample.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Pass criterion
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
# Per-run pass@k
# -----------------------------------------------------------------------------


def compute_pass_at_k_one_run(
    problem_logs: list[dict],
    max_k: int = 8,
) -> list[float]:
    """
    For each k in 1..max_k, fraction (0–100) of problems that have at least one
    pass in the first k attempts (attempts in chronological order).

    Returns a list of length max_k: [pct at k=1, pct at k=2, ..., pct at k=max_k].
    """
    if not problem_logs:
        return []

    n_problems = len(problem_logs)
    pct_solved = []

    for k in range(1, max_k + 1):
        n_solved = 0
        for log in problem_logs:
            attempts = log.get("attempts", [])
            first_k = attempts[:k]
            if any(is_pass(a) for a in first_k):
                n_solved += 1
        pct_solved.append(100.0 * n_solved / n_problems)

    return pct_solved


# -----------------------------------------------------------------------------
# Load runs and aggregate
# -----------------------------------------------------------------------------


def load_run_logs(run_dir: Path) -> list[dict]:
    """Load logs.json from a run directory. Raises if missing or invalid."""
    path = run_dir / "logs.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of problem logs in {path}")
    return data


def collect_run_dirs(parent_dir: Path) -> list[Path]:
    """Return list of subdirectories that contain logs.json, sorted by name."""
    if not parent_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {parent_dir}")
    run_dirs = []
    for child in sorted(parent_dir.iterdir()):
        if child.is_dir() and (child / "logs.json").exists():
            run_dirs.append(child)
    return run_dirs


def aggregate_pass_at_k(
    parent_dir: Path,
    max_k: int = 8,
) -> tuple[list[int], np.ndarray, np.ndarray, int]:
    """
    Load all runs under parent_dir, compute pass@k for each, return aggregates.

    Returns:
        ks: [1, 2, ..., max_k]
        mean: shape (max_k,) — mean percentage solved per k
        std: shape (max_k,) — standard deviation across runs per k
        n_runs: number of runs used
    """
    run_dirs = collect_run_dirs(parent_dir)
    if not run_dirs:
        raise ValueError(f"No run subdirectories with logs.json found in {parent_dir}")

    rows = []
    for run_dir in run_dirs:
        problem_logs = load_run_logs(run_dir)
        row = compute_pass_at_k_one_run(problem_logs, max_k=max_k)
        if len(row) != max_k:
            raise ValueError(
                f"Run {run_dir.name}: expected {max_k} pass@k values, got {len(row)}"
            )
        rows.append(row)

    arr = np.array(rows)
    ks = list(range(1, max_k + 1))
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return ks, mean, std, len(run_dirs)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_pass_at_k_multi_run(
    parent_dir: Path,
    output_path: Path | None = None,
    max_k: int = 8,
    n_problems: int | None = None,
) -> Path:
    """
    Compute mean ± std pass@k across runs in parent_dir and save plot.

    If n_problems is None, we do not include it in the title (or could infer from
    first run's logs). Output path defaults to parent_dir / "pass_at_k_50sample.png".
    """
    import matplotlib.pyplot as plt

    ks, mean, std, n_runs = aggregate_pass_at_k(parent_dir, max_k=max_k)

    if output_path is None:
        output_path = parent_dir / "pass_at_k_50sample.png"
    output_path = Path(output_path)

    fig, ax = plt.subplots()
    ax.errorbar(
        ks,
        mean,
        yerr=std,
        marker="o",
        linestyle="-",
        markersize=6,
        capsize=4,
        capthick=1,
        color="C0",
    )
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    title = "Pass@k (n=50 problems)"
    if n_problems is not None:
        title = f"Pass@k (n={n_problems} problems)"
    ax.set_title(title)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot average Pass@k ± std across multiple runs (each subfolder = one run with logs.json)"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Directory containing one subfolder per pass@8 run (each with logs.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <folder>/pass_at_k_50sample.png)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=8,
        help="Maximum k for pass@k (default: 8)",
    )
    parser.add_argument(
        "-n",
        "--n-problems",
        type=int,
        default=50,
        help="Number of problems for plot title (default: 50)",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    out = plot_pass_at_k_multi_run(
        folder,
        output_path=args.output,
        max_k=args.max_k,
        n_problems=args.n_problems,
    )
    print(f"Saved plot → {out}")


if __name__ == "__main__":
    main()
