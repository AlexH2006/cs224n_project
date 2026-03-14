#!/usr/bin/env python3
"""
Print failed problem IDs and verification feedback from a baseline run summary.json.

Usage:
    python baseline/scripts/inspect_baseline_summary.py results/run_<timestamp>/summary.json

The summary is produced by baseline/lean_baseline_eval_modal.py (verify step).
"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: inspect_baseline_summary.py <summary.json>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    results = summary.get("results", [])
    if not results:
        print("No 'results' in summary.")
        return
    n_total = summary.get("n_problems", len(results))
    n_success = summary.get("n_success", sum(1 for r in results if r.get("success")))
    print(f"Summary: {n_success}/{n_total} passed")
    failed = [r for r in results if not r.get("success")]
    if not failed:
        print("No failed problems.")
        return
    print(f"\nFailed problems ({len(failed)}):")
    for r in failed:
        pid = r.get("problem_id", "?")
        pidx = r.get("problem_idx", "?")
        ver = r.get("verification") or {}
        feedback = (ver.get("feedback") or "").strip()
        first_line = feedback.splitlines()[0][:200] if feedback else "(no feedback)"
        print(f"  problem_idx={pidx}  problem_id={pid}")
        print(f"    feedback: {first_line}")


if __name__ == "__main__":
    main()
