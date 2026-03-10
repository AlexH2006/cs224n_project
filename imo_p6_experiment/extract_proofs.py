#!/usr/bin/env python3
"""
Extract all extracted_block entries from a run's logs.json and write proofs.json
in the same directory.

Usage (from repo root):
  python qwen_sdpo_old/extract_proofs.py path/to/logs.json
  python qwen_sdpo_old/extract_proofs.py sdpo_results/local_verify/Qwen3.5-4B/minif2f-lean4/run_100_20260310_075625/logs.json

Or from the run directory:
  python path/to/qwen_sdpo_old/extract_proofs.py logs.json
"""

import argparse
import json
import sys
from pathlib import Path


def extract_proofs_from_logs(logs: dict) -> list[dict]:
    """Build list of extracted_block entries (iteration, extracted_block, success, problem_id)."""
    problem = logs.get("problem") or {}
    problem_id = problem.get("id") or problem.get("problem_id") or problem.get("name")
    iteration_logs = logs.get("iteration_logs", [])
    out = []
    for il in iteration_logs:
        entry = {
            "iteration": il.get("iteration"),
            "extracted_block": il.get("extracted_block", ""),
            "success": il.get("success", False),
        }
        if problem_id is not None:
            entry["problem_id"] = problem_id
        out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract all extracted_block entries from logs.json and write proofs.json in the same directory."
    )
    parser.add_argument(
        "logs_json",
        type=Path,
        help="Path to logs.json (e.g. sdpo_results/.../run_100_.../logs.json)",
    )
    args = parser.parse_args()
    logs_path = args.logs_json.resolve()
    if not logs_path.is_file():
        print(f"Error: not a file: {logs_path}", file=sys.stderr)
        sys.exit(1)

    with open(logs_path) as f:
        logs = json.load(f)
    proofs = extract_proofs_from_logs(logs)
    run_dir = logs_path.parent
    out_path = run_dir / "proofs.json"
    with open(out_path, "w") as f:
        json.dump(proofs, f, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
