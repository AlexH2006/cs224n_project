"""
Sample 50 random problems from the dataset (indices 0–243), record their
problem_idx and problem_id in results/sampled_problems.json, then plot
pass@k for this sample and save in the results folder.

Usage:
    python -m qwen_multiturn.utils.sample_and_plot_pass_at_k RUN_DIR [RESULTS_DIR]
    python -m qwen_multiturn.utils.sample_and_plot_pass_at_k test_results_multiturn/run_Qwen3.5-4B_20260310_110852
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from qwen_multiturn.utils.plot_pass_at_k import compute_pass_at_k

# Pass-rate buckets for pass_k=8: 0/8, 1/8, ..., 8/8
def _rate_bucket(success_rate: float, pass_k: int = 8) -> float:
    """Round success_rate to nearest 1/pass_k (e.g. 0, 0.125, ..., 1.0 for pass_k=8)."""
    return round(success_rate * pass_k) / pass_k


def build_success_rate_summary(problem_logs: list[dict], pass_k: int) -> dict:
    """
    Build a success_rate_summary-style dict (pass_k, n_problems, problems with
    problem_id, success_rate, problem_idx) from a list of problem logs.
    """
    problems_out = []
    for log in problem_logs:
        problem = log.get("problem", {}) or {}
        problem_id = problem.get("id", "")
        problem_idx = problem.get("problem_idx")
        attempts = log.get("attempts", [])
        total = len(attempts)
        n_ok = sum(1 for a in attempts if a.get("success"))
        success_rate = round(n_ok / total, 4) if total else 0.0
        entry = {"problem_id": problem_id, "success_rate": success_rate}
        if problem_idx is not None:
            entry["problem_idx"] = problem_idx
        problems_out.append(entry)
    return {
        "pass_k": pass_k,
        "n_problems": len(problem_logs),
        "problems": problems_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample 50 problems, write sampled_problems.json, plot pass@k for sample"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing logs.json and success_rate_summary.json",
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=Path("results"),
        help="Output directory for sampled_problems.json and plot (default: results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=50,
        help="Number of problems to sample (default: 50)",
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Match full-dataset pass@k proportion: sample so solved/unsolved ratio in sample equals full dataset",
    )
    parser.add_argument(
        "--stratify-by-rate",
        action="store_true",
        help="Stratify by pass-rate buckets (0/8, 1/8, ..., 8/8): sample proportion in each bucket matches full dataset",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load problem_idx -> problem_id from success_rate_summary
    summary_path = run_dir / "success_rate_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Not found: {summary_path}")
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    problems = summary["problems"]
    n_total = summary["n_problems"]
    if n_total == 0:
        raise SystemExit("No problems in success_rate_summary.json")

    n_sample = min(args.n_sample, n_total)
    random.seed(args.seed)
    pass_k = summary.get("pass_k", 8)

    if args.stratify_by_rate:
        # Stratify by pass-rate buckets (0/8, 0.125, ..., 1.0): match full-dataset proportion per bucket.
        buckets = defaultdict(list)
        for i in range(n_total):
            rate = problems[i].get("success_rate", 0.0)
            b = _rate_bucket(rate, pass_k)
            buckets[b].append(i)
        # Target count per bucket (proportional); ensure sum = n_sample
        bucket_rates = sorted(buckets.keys())
        targets = {b: round(n_sample * len(buckets[b]) / n_total) for b in bucket_rates}
        total_target = sum(targets.values())
        if total_target != n_sample:
            # Adjust largest bucket to hit n_sample
            largest_b = max(bucket_rates, key=lambda b: len(buckets[b]))
            targets[largest_b] = targets[largest_b] + (n_sample - total_target)
        indices = []
        for b in bucket_rates:
            take = min(targets[b], len(buckets[b]))
            indices.extend(random.sample(buckets[b], take))
        shortfall = n_sample - len(indices)
        if shortfall > 0:
            # Fill from buckets that have spare (largest first)
            remaining = [i for b in bucket_rates for i in buckets[b] if i not in indices]
            indices.extend(random.sample(remaining, min(shortfall, len(remaining))))
        indices = sorted(indices)
        # Report bucket counts
        sample_rates = [_rate_bucket(problems[i].get("success_rate", 0), pass_k) for i in indices]
        count_by_rate = Counter(sample_rates)
        msg = ", ".join(f"{r:.3g}:{count_by_rate[r]}" for r in sorted(count_by_rate.keys()))
        print(f"Stratified by rate (pass_k={pass_k}): {msg}")
    elif args.stratify:
        # Stratified sampling: same proportion of pass@k (solved) as in full dataset.
        # Solved = at least one success in attempts (success_rate > 0).
        solved_indices = [i for i in range(n_total) if problems[i].get("success_rate", 0) > 0]
        unsolved_indices = [i for i in range(n_total) if problems[i].get("success_rate", 0) == 0]
        n_solved_full = len(solved_indices)
        n_unsolved_full = len(unsolved_indices)
        # Target counts in sample to match proportion
        n_solved_sample = round(n_sample * n_solved_full / n_total)
        n_unsolved_sample = n_sample - n_solved_sample
        # Clamp to available (e.g. if sample is large and one stratum is small)
        n_solved_take = min(n_solved_sample, n_solved_full)
        n_unsolved_take = min(n_unsolved_sample, n_unsolved_full)
        shortfall = n_sample - n_solved_take - n_unsolved_take
        if shortfall > 0:
            if n_solved_take < n_solved_full:
                n_solved_take = min(n_solved_take + shortfall, n_solved_full)
            else:
                n_unsolved_take = min(n_unsolved_take + shortfall, n_unsolved_full)
        sampled_solved = random.sample(solved_indices, n_solved_take)
        sampled_unsolved = random.sample(unsolved_indices, n_unsolved_take)
        indices = sorted(sampled_solved + sampled_unsolved)
        print(f"Stratified sample: {n_solved_take} solved, {n_unsolved_take} unsolved (full: {n_solved_full}/{n_total} = {100*n_solved_full/n_total:.1f}% solved)")
    else:
        indices = sorted(random.sample(range(n_total), n_sample))

    sampled = [
        {"problem_idx": problems[i]["problem_idx"], "problem_id": problems[i]["problem_id"]}
        for i in indices
    ]

    # Write results/sampled_problems.json
    sampled_path = results_dir / "sampled_problems.json"
    out_meta = {"n_sampled": len(sampled), "problem_indices": indices, "problems": sampled}
    if args.stratify_by_rate:
        out_meta["stratify_by_rate"] = True
        out_meta["pass_k"] = pass_k
        sample_rates = [_rate_bucket(problems[i].get("success_rate", 0), pass_k) for i in indices]
        out_meta["count_by_rate"] = dict(sorted(Counter(sample_rates).items()))
    elif args.stratify:
        out_meta["stratified"] = True
        n_solved_in_sample = sum(1 for i in indices if problems[i].get("success_rate", 0) > 0)
        out_meta["n_solved"] = n_solved_in_sample
        out_meta["n_unsolved"] = len(sampled) - n_solved_in_sample
    with open(sampled_path, "w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2)
    print(f"Saved {len(sampled)} sampled problems → {sampled_path}")

    # Load logs and filter to sampled problem_indices
    logs_path = run_dir / "logs.json"
    if not logs_path.exists():
        raise SystemExit(f"Not found: {logs_path}")
    with open(logs_path, encoding="utf-8") as f:
        all_logs = json.load(f)
    sampled_set = set(indices)
    filtered_logs = [log for log in all_logs if log.get("problem", {}).get("problem_idx") in sampled_set]
    if len(filtered_logs) != len(sampled):
        print(f"Warning: only {len(filtered_logs)} of {len(sampled)} sampled problems found in logs")

    # Write success_rate_summary for the sample (same format as run's success_rate_summary.json)
    pass_k = len(filtered_logs[0]["attempts"]) if filtered_logs else summary.get("pass_k", 0)
    success_rate_sampled = build_success_rate_summary(filtered_logs, pass_k)
    success_rate_path = results_dir / "success_rate_summary_sampled.json"
    with open(success_rate_path, "w", encoding="utf-8") as f:
        json.dump(success_rate_sampled, f, indent=2, ensure_ascii=False)
    print(f"Saved success-rate summary (sampled) → {success_rate_path}")

    # Plot pass@k for sample
    import matplotlib.pyplot as plt

    ks, pct_solved = compute_pass_at_k(filtered_logs)
    if not ks:
        raise SystemExit("No attempts in filtered logs")
    fig, ax = plt.subplots()
    ax.plot(ks, pct_solved, marker="o", linestyle="-", markersize=6)
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title(f"Pass@k (n={len(filtered_logs)} sampled problems)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plot_path = results_dir / "pass_at_k_50sample.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {plot_path}")


if __name__ == "__main__":
    main()
