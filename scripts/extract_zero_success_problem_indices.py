#!/usr/bin/env python3
"""
Extract problem_idx for problems with success_rate 0.0 from a success_rate_summary.json.

Output: dynamic_sampling/problem_idx.json with {"problem_indices": [...]} (sorted).

Usage (from project root):
    python scripts/extract_zero_success_problem_indices.py
    python scripts/extract_zero_success_problem_indices.py results/Qwen3.5-4B_no_thinking/success_rate_summary.json -o dynamic_sampling/problem_idx.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_summary = repo_root / "results" / "Qwen3.5-4B_no_thinking" / "success_rate_summary.json"
    default_output = repo_root / "dynamic_sampling" / "problem_idx.json"

    parser = argparse.ArgumentParser(
        description="Extract problem_idx with success_rate 0.0 from success_rate_summary.json; write to dynamic_sampling/problem_idx.json."
    )
    parser.add_argument(
        "summary_path",
        type=Path,
        nargs="?",
        default=default_summary,
        help=f"Path to success_rate_summary.json (default: {default_summary})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSON path (default: {default_output})",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_path).resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)

    problems = data.get("problems", [])
    if not isinstance(problems, list):
        raise ValueError("Expected 'problems' to be a list")

    indices = sorted(
        p["problem_idx"]
        for p in problems
        if isinstance(p, dict) and p.get("success_rate") == 0.0
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"problem_indices": indices}, f, indent=2)

    print(f"Loaded {len(problems)} problems from {summary_path}")
    print(f"Problems with success_rate 0.0: {len(indices)}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
