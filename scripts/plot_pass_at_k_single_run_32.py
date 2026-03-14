"""
Plot Pass@k (k=1..32) for a single run that has 32 attempts per problem.
Same style as pass_at_k_32_merged.png (blue line, markers, grid).

Usage:
  python scripts/plot_pass_at_k_single_run_32.py results/Qwen3.5_4B_sampled_no_thinking_parallel_samplin/pass@32_round2/run_Qwen3.5-4B_20260312_121334
  python scripts/plot_pass_at_k_single_run_32.py <run_dir> -o <output.png>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def is_pass(attempt: dict[str, Any]) -> bool:
    verification = attempt.get("verification") or {}
    if not verification:
        return bool(attempt.get("success"))
    return (
        verification.get("success", False)
        and verification.get("complete", False)
        and not verification.get("has_sorry", True)
    )


def first_success_attempt(attempts: list[dict[str, Any]]) -> int | None:
    """1-based attempt index of first pass, or None if never."""
    for i, a in enumerate(attempts):
        if is_pass(a):
            return i + 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Pass@k (k=1..32) for a single run with 32 attempts per problem.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing logs.json (list of problems, each with 32 attempts)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <parent_of_run_dir>/pass_at_k_32_merged.png)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=32,
        help="Maximum k (default: 32)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    logs_path = run_dir / "logs.json"
    if not logs_path.is_file():
        raise SystemExit(f"Missing: {logs_path}")

    with open(logs_path, encoding="utf-8") as f:
        problem_logs = json.load(f)
    if not isinstance(problem_logs, list):
        raise SystemExit("Expected logs.json to be a list of problem logs")

    records = []
    for log in problem_logs:
        problem = log.get("problem") or {}
        problem_idx = problem.get("problem_idx", len(records))
        problem_id = problem.get("id") or problem.get("problem_id") or f"problem_{problem_idx}"
        attempts = log.get("attempts") or []
        first = first_success_attempt(attempts)
        records.append((problem_idx, problem_id, first))

    n_problems = len(records)
    max_k = args.max_k
    ks = list(range(1, max_k + 1))
    first_success = [r[2] for r in records]
    pcts = [100.0 * sum(1 for a in first_success if a is not None and a <= k) / n_problems for k in ks]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(ks, pcts, marker="o", linestyle="-", markersize=6, color="C0")
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title(f"Pass@k (n={n_problems} problems, merged pass@32)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = run_dir.parent / "pass_at_k_32_merged.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    main()
