#!/usr/bin/env python3
"""
Combine two SDPO run directories into a single result: union of problems by problem_idx,
taking the best result when a problem appears in both runs. Writes JSON summaries and a
pass@k plot matching the format of results/Qwen3.5_4B_sampled_no_thinking.

Usage (from project root):
    python scripts/combine_sdpo_runs.py \\
        sdpo_results/Qwen3.5-4B/run_Qwen3.5-4B_20260311_142317 \\
        sdpo_results/Qwen3.5-4B/run_Qwen3.5-4B_20260311_111404 \\
        -o results/Qwen3.5_4B_sdpo_combined
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_sdpo_run(runs_dir: Path) -> dict[int, dict]:
    """
    Load per-problem data from SDPO runs/problem_*/logs.json.
    Returns dict: problem_idx -> { problem_id, first_success_at, solved, attempts_used }.
    first_success_at is 1-8 for the attempt at which the problem was first solved, or 9
    as a sentinel meaning "never solved" (no success in any of the 8 iterations).
    """
    runs_dir = Path(runs_dir)
    out = {}
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("problem_"):
            continue
        logs_path = path / "logs.json"
        if not logs_path.exists():
            continue
        with open(logs_path, encoding="utf-8") as f:
            logs = json.load(f)
        problem = logs.get("problem", {}) or {}
        problem_idx = problem.get("problem_idx")
        if problem_idx is None:
            try:
                problem_idx = int(path.name.replace("problem_", ""))
            except ValueError:
                continue
        iteration_logs = logs.get("iteration_logs", [])
        attempts_used = len(iteration_logs)
        first_success_at = 9
        for i, it in enumerate(iteration_logs):
            if it.get("success"):
                first_success_at = i + 1
                break
        solved = logs.get("success", False)
        out[problem_idx] = {
            "problem_id": problem.get("problem_id", ""),
            "first_success_at": first_success_at,
            "solved": solved,
            "attempts_used": attempts_used,
        }
    return out


def compute_pass_at_k(first_success_values: list[int], max_k: int = 8) -> tuple[list[int], list[float]]:
    """
    first_success_values: per-problem attempt at first success (1-8 or 9 if never).
    Returns (ks, pct_solved): k in 1..max_k, and fraction (0-100) of problems solved within k.
    """
    n = len(first_success_values)
    if n == 0:
        return [], []
    ks = list(range(1, max_k + 1))
    pct = [100.0 * sum(1 for v in first_success_values if v <= k) / n for k in ks]
    return ks, pct


def merge_runs(
    run1_dir: Path,
    run2_dir: Path,
    data1: dict[int, dict],
    data2: dict[int, dict],
) -> dict[int, dict]:
    """
    Union of problem_idx from both runs. For each problem in both, take best first_success_at
    (min) and solved = solved1 or solved2. Attach source_runs for metadata.
    """
    name1 = run1_dir.name
    name2 = run2_dir.name
    merged = {}
    all_indices = set(data1) | set(data2)
    for idx in all_indices:
        d1 = data1.get(idx)
        d2 = data2.get(idx)
        if d1 is None:
            merged[idx] = {
                **d2,
                "source_runs": [name2],
            }
        elif d2 is None:
            merged[idx] = {
                **d1,
                "source_runs": [name1],
            }
        else:
            first_success_at = min(d1["first_success_at"], d2["first_success_at"])
            solved = d1["solved"] or d2["solved"]
            merged[idx] = {
                "problem_id": d1.get("problem_id") or d2.get("problem_id", ""),
                "first_success_at": first_success_at,
                "solved": solved,
                "attempts_used": max(d1.get("attempts_used", 0), d2.get("attempts_used", 0)),
                "source_runs": [name1, name2],
            }
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine two SDPO runs by problem_idx (best result), write JSON and pass@k plot"
    )
    parser.add_argument(
        "run1",
        type=Path,
        help="First SDPO run directory (contains runs/problem_*/)",
    )
    parser.add_argument(
        "run2",
        type=Path,
        help="Second SDPO run directory (contains runs/problem_*/)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("results/Qwen3.5_4B_sdpo_combined"),
        help="Output directory for JSON and plot (default: results/Qwen3.5_4B_sdpo_combined)",
    )
    args = parser.parse_args()

    run1_dir = Path(args.run1).resolve()
    run2_dir = Path(args.run2).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs1 = run1_dir / "runs"
    runs2 = run2_dir / "runs"
    if not runs1.is_dir():
        raise SystemExit(f"Not a directory: {runs1}")
    if not runs2.is_dir():
        raise SystemExit(f"Not a directory: {runs2}")

    data1 = load_sdpo_run(runs1)
    data2 = load_sdpo_run(runs2)
    merged = merge_runs(run1_dir, run2_dir, data1, data2)

    if not merged:
        raise SystemExit("No problems found in either run.")

    problem_indices = sorted(merged.keys())
    n_problems = len(problem_indices)
    first_success_values = [merged[idx]["first_success_at"] for idx in problem_indices]
    pass_at_k_8 = round(
        sum(1 for v in first_success_values if v <= 8) / n_problems, 4
    )

    # combined_summary.json
    problems_summary = [
        {
            "problem_idx": idx,
            "problem_id": merged[idx]["problem_id"],
            "first_success_at": merged[idx]["first_success_at"],
            "solved": merged[idx]["solved"],
            "source_runs": merged[idx]["source_runs"],
        }
        for idx in problem_indices
    ]
    combined = {
        "run_sources": [run1_dir.name, run2_dir.name],
        "run_source_paths": [str(run1_dir), str(run2_dir)],
        "n_problems": n_problems,
        "problem_indices": problem_indices,
        "problems": problems_summary,
        "pass_k": 8,
        "pass_at_k": pass_at_k_8,
        "_comment_first_success_at": "1-8 = attempt at first success; 9 = never solved (sentinel, only 8 iterations run)",
    }
    combined_path = out_dir / "combined_summary.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved {combined_path}")

    # success_rate_summary_sampled.json (same shape as reference)
    success_rate_problems = [
        {
            "problem_id": merged[idx]["problem_id"],
            "success_rate": 1.0 if merged[idx]["solved"] else 0.0,
            "problem_idx": idx,
        }
        for idx in problem_indices
    ]
    success_rate_summary = {
        "pass_k": 8,
        "n_problems": n_problems,
        "problems": success_rate_problems,
    }
    success_rate_path = out_dir / "success_rate_summary_sampled.json"
    with open(success_rate_path, "w", encoding="utf-8") as f:
        json.dump(success_rate_summary, f, indent=2)
    print(f"Saved {success_rate_path}")

    # Pass@k plot
    ks, pct_solved = compute_pass_at_k(first_success_values, max_k=8)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(ks, pct_solved, marker="o", linestyle="-", markersize=6)
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title(f"Pass@k (n={n_problems} problems)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plot_path = out_dir / "pass_at_k_50sample.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
