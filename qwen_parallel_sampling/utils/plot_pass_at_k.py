"""
Plot success@k vs k from an eval run's logs.json.

Y-axis: percentage of problems solved (at least one success in the first k attempts).
X-axis: k (number of attempts considered).

Usage:
    python -m qwen_eval.utils.plot_pass_at_k /path/to/run_dir/logs.json
    python -m qwen_eval.utils.plot_pass_at_k /path/to/run_dir/logs.json -o /path/to/run_dir/pass_at_k.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_pass_at_k(problem_logs: list[dict]) -> tuple[list[int], list[float]]:
    """
    For each k in 1..max_attempts, compute the number and percentage of problems
    that have at least one successful attempt in the first k attempts.

    Returns:
        ks: list of k values (1, 2, ..., max_attempts)
        pct_solved: list of percentages (0–100) for each k
    """
    if not problem_logs:
        return [], []

    max_k = max(len(log.get("attempts", [])) for log in problem_logs)
    if max_k == 0:
        return [], []

    ks = list(range(1, max_k + 1))
    n_problems = len(problem_logs)
    counts = []

    for k in ks:
        n_solved = 0
        for log in problem_logs:
            attempts = log.get("attempts", [])
            first_k = attempts[:k]
            if any(a.get("success") for a in first_k):
                n_solved += 1
        counts.append(n_solved)

    pct_solved = [100.0 * c / n_problems for c in counts]
    return ks, pct_solved


def plot_pass_at_k(
    logs_path: Path,
    output_path: Path | None = None,
    title: str | None = None,
) -> Path:
    """
    Load logs.json, compute success@k, and save a plot.

    If output_path is None, writes to the same directory as logs_path with
    filename pass_at_k.png.
    """
    import matplotlib.pyplot as plt

    with open(logs_path, encoding="utf-8") as f:
        problem_logs = json.load(f)

    ks, pct_solved = compute_pass_at_k(problem_logs)
    if not ks:
        raise ValueError(f"No attempts found in {logs_path}")

    fig, ax = plt.subplots()
    ax.plot(ks, pct_solved, marker="o", linestyle="-", markersize=6)
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    if title:
        ax.set_title(title)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    if output_path is None:
        output_path = logs_path.parent / "pass_at_k.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot success@k vs k from eval logs.json"
    )
    parser.add_argument(
        "logs_json",
        type=Path,
        help="Path to logs.json (e.g. baseline/run_XXX/logs.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: same dir as logs.json, pass_at_k.png)",
    )
    parser.add_argument(
        "-t",
        "--title",
        type=str,
        default=None,
        help="Plot title",
    )
    args = parser.parse_args()

    if not args.logs_json.exists():
        raise SystemExit(f"File not found: {args.logs_json}")

    out = plot_pass_at_k(args.logs_json, output_path=args.output, title=args.title)
    print(f"Saved plot → {out}")


if __name__ == "__main__":
    main()
