#!/usr/bin/env python3
"""
Compare successful samples between two baseline runs using their
success_rate_summary.json files. A problem is "successful" if success_rate > 0
(at least one correct solution in pass@k attempts).

Usage:
    python scripts/compare_baseline_runs.py \\
        baseline/run_A/success_rate_summary.json \\
        baseline/run_B/success_rate_summary.json

    # Single file: list successful problem_ids for that run
    python scripts/compare_baseline_runs.py baseline/run_B/success_rate_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> tuple[dict[str, float], int, int]:
    """Load success_rate_summary.json. Returns (problem_id -> success_rate, pass_k, n_problems)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rates = {p["problem_id"]: p["success_rate"] for p in data["problems"]}
    return rates, data.get("pass_k"), data.get("n_problems", len(rates))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare successful samples between two baseline runs"
    )
    parser.add_argument(
        "summary_a",
        type=Path,
        help="Path to first run's success_rate_summary.json",
    )
    parser.add_argument(
        "summary_b",
        type=Path,
        nargs="?",
        default=None,
        help="Path to second run's success_rate_summary.json (optional)",
    )
    args = parser.parse_args()

    path_a = Path(args.summary_a)
    if not path_a.is_file():
        raise SystemExit(f"File not found: {path_a}")

    rates_a, pass_k_a, n_a = load_summary(path_a)
    successful_a = {pid for pid, r in rates_a.items() if r > 0}

    if args.summary_b is None:
        print(f"Run: {path_a.parent.name}")
        print(f"pass_k={pass_k_a}, n_problems={n_a}, n_successful={len(successful_a)}")
        print("\nSuccessful problem_ids (success_rate > 0):")
        for pid in sorted(successful_a, key=lambda x: (rates_a[x], x), reverse=True):
            print(f"  {rates_a[pid]:.4f}  {pid}")
        return

    path_b = Path(args.summary_b)
    if not path_b.is_file():
        raise SystemExit(f"File not found: {path_b}")

    rates_b, pass_k_b, n_b = load_summary(path_b)
    successful_b = {pid for pid, r in rates_b.items() if r > 0}

    # All problem_ids (union)
    all_ids = sorted(set(rates_a) | set(rates_b))
    in_both = successful_a & successful_b
    only_a = successful_a - successful_b
    only_b = successful_b - successful_a

    name_a = path_a.parent.name
    name_b = path_b.parent.name

    print("=== Baseline run comparison ===\n")
    print(f"Run A: {name_a}  -> {len(successful_a)} successful (pass_k={pass_k_a})")
    print(f"Run B: {name_b}  -> {len(successful_b)} successful (pass_k={pass_k_b})")
    print()
    print(f"Successful in both:     {len(in_both)}")
    print(f"Successful only in A:  {len(only_a)}")
    print(f"Successful only in B:  {len(only_b)}")
    print()

    if only_a:
        print("--- Only in Run A (successful in A, not in B) ---")
        for pid in sorted(only_a):
            print(f"  {pid}  (A: {rates_a[pid]:.4f}, B: {rates_b.get(pid, 0):.4f})")
        print()

    if only_b:
        print("--- Only in Run B (successful in B, not in A) ---")
        for pid in sorted(only_b):
            print(f"  {pid}  (A: {rates_a.get(pid, 0):.4f}, B: {rates_b[pid]:.4f})")
        print()

    # Success rate differences (where both have the problem)
    diffs = []
    for pid in all_ids:
        ra, rb = rates_a.get(pid, 0), rates_b.get(pid, 0)
        if ra != rb:
            diffs.append((pid, ra, rb, rb - ra))
    if diffs:
        print("--- Success rate changes (B - A) ---")
        for pid, ra, rb, d in sorted(diffs, key=lambda x: -abs(x[3])):
            print(f"  {pid}: A={ra:.4f} B={rb:.4f}  delta={d:+.4f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
