"""
Overlay the two pass@32 curves on one plot: parallel (merged) vs SDPO.

TL;DR:
  - Loads/computes pass@k (k=1..32) for (1) parallel_samplin (four runs merged in order)
    and (2) SDPO pass@32 (two runs, 50 problems combined).
  - Plots both curves on the same axes with distinct colors and a legend.
  - Saves to a single PNG (default: results/pass_at_k_32_comparison.png).

Usage:
  python scripts/plot_pass_at_k_32_comparison.py
  python scripts/plot_pass_at_k_32_comparison.py -o results/pass_at_k_32_comparison.png
  python scripts/plot_pass_at_k_32_comparison.py --parallel-dir ... --sdpo-dir ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse data logic from existing scripts (no plot output from them)
from pass_at_k_parallel_merged_pass32 import (
    collect_run_dirs as parallel_collect_run_dirs,
    compute_merged_first_success,
    compute_pass_at_k as compute_pass_at_k_parallel,
)
from pass_at_k_sdpo_pass32 import (
    build_summary_records,
    collect_all_problems,
    compute_pass_at_k as compute_pass_at_k_sdpo,
)


def get_parallel_pass_at_k(parallel_dir: Path, max_k: int = 32) -> tuple[list[int], list[float]]:
    """Compute pass@k for merged parallel (four pass@8 runs)."""
    run_dirs = parallel_collect_run_dirs(parallel_dir)
    if len(run_dirs) != 4:
        raise ValueError(
            f"Expected exactly 4 run dirs with logs.json under {parallel_dir}, found {len(run_dirs)}"
        )
    records = compute_merged_first_success(run_dirs)
    return compute_pass_at_k_parallel(records, max_k=max_k)


def get_sdpo_pass_at_k(sdpo_pass32_dir: Path, max_k: int = 32) -> tuple[list[int], list[float]]:
    """Compute pass@k for SDPO pass@32 (two runs, 50 problems)."""
    collected = collect_all_problems(sdpo_pass32_dir)
    records = build_summary_records(collected)
    return compute_pass_at_k_sdpo(records, max_k=max_k)


def plot_comparison(
    parallel_dir: Path,
    sdpo_pass32_dir: Path,
    output_path: Path,
    max_k: int = 32,
    n_problems: int = 50,
) -> Path:
    """Overlay both pass@32 curves and save."""
    import matplotlib.pyplot as plt

    ks_par, pcts_par = get_parallel_pass_at_k(parallel_dir, max_k=max_k)
    ks_sdpo, pcts_sdpo = get_sdpo_pass_at_k(sdpo_pass32_dir, max_k=max_k)
    if ks_par != ks_sdpo:
        ks = list(range(1, max_k + 1))
        if ks_par != ks or ks_sdpo != ks:
            raise ValueError("k ranges differ between sources")
    ks = ks_par

    fig, ax = plt.subplots()
    ax.plot(
        ks,
        pcts_par,
        marker="o",
        linestyle="-",
        markersize=5,
        color="C0",
        label="Parallel (merged pass@32)",
    )
    ax.plot(
        ks,
        pcts_sdpo,
        marker="s",
        linestyle="-",
        markersize=5,
        color="C1",
        label="SDPO (pass@32)",
    )
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title(f"Pass@k (n={n_problems} problems)")
    ax.set_xticks(ks)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay parallel (merged) pass@32 and SDPO pass@32 on one plot.",
    )
    parser.add_argument(
        "--parallel-dir",
        type=Path,
        default=Path("results/Qwen3.5_4B_sampled_no_thinking_parallel_samplin"),
        help="Directory containing the four run_* subdirs (parallel sampling)",
    )
    parser.add_argument(
        "--sdpo-dir",
        type=Path,
        default=Path("results/Qwen3.5_4B_sampled_no_thinking_sdpo/pass@32"),
        help="SDPO pass@32 directory (run_*/runs/problem_*/logs.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("results/pass_at_k_32_comparison.png"),
        help="Output PNG path",
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

    parallel_dir = args.parallel_dir.resolve()
    sdpo_dir = args.sdpo_dir.resolve()
    if not parallel_dir.is_dir():
        raise SystemExit(f"Not a directory: {parallel_dir}")
    if not sdpo_dir.is_dir():
        raise SystemExit(f"Not a directory: {sdpo_dir}")

    out = plot_comparison(
        parallel_dir,
        sdpo_dir,
        args.output.resolve(),
        max_k=args.max_k,
        n_problems=args.n_problems,
    )
    print(f"Saved comparison plot → {out}")


if __name__ == "__main__":
    main()
