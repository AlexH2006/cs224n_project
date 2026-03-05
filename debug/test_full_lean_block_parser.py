"""
TLDR: Run the full Lean block parser on iteration_logs from a SDPO run logs.json.
Reports success/failure per iteration and whether the extracted block contains
the expected proof structure (e.g. "have h1" / "exact h1") to validate that
full-block extraction preserves lines that the tactic extractor drops.
"""

import json
import sys
from pathlib import Path

# Allow importing from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from debug.full_lean_block_parser import extract_full_lean_block


# Default logs file: run_141 (mathd_numbertheory_458) from the user's example.
DEFAULT_LOGS_PATH = _REPO_ROOT / "sdpo_results/kimina_2b_local_verify/minif2f-lean4/run_141_20260304_124651/logs.json"


def run_tests(logs_path: Path) -> dict:
    """
    Load logs.json and run extract_full_lean_block on each iteration's raw_output.
    Returns a summary dict and prints per-iteration results.
    """
    with open(logs_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    iteration_logs = logs.get("iteration_logs", [])
    if not iteration_logs:
        print("No iteration_logs in", logs_path)
        return {"total": 0, "extracted": 0, "with_have": 0, "results": []}

    results = []
    for entry in iteration_logs:
        it = entry.get("iteration", 0)
        raw = entry.get("raw_output", "")
        extracted = extract_full_lean_block(raw)
        success = extracted is not None
        length = len(extracted) if extracted else 0
        has_have_h1 = "have h1" in (extracted or "")
        has_have_h2 = "have h2" in (extracted or "")
        has_exact_h1 = "exact h1" in (extracted or "")
        has_exact_h2 = "exact h2" in (extracted or "")
        has_omega = "omega" in (extracted or "")

        results.append({
            "iteration": it,
            "success": success,
            "length": length,
            "has_have_h1": has_have_h1,
            "has_have_h2": has_have_h2,
            "has_exact_h1": has_exact_h1,
            "has_exact_h2": has_exact_h2,
            "has_omega": has_omega,
        })

        print(f"  Iteration {it}: extracted={success}, len={length}, "
              f"have_h1={has_have_h1}, have_h2={has_have_h2}, "
              f"exact_h1={has_exact_h1}, exact_h2={has_exact_h2}, omega={has_omega}")

        if extracted and len(extracted) <= 400:
            print(f"    block preview: {extracted[:200]}...")
        elif extracted:
            print(f"    block preview: {extracted[:150]}... ({length} chars)")

    extracted_count = sum(1 for r in results if r["success"])
    with_have_count = sum(1 for r in results if r["has_have_h1"] or r["has_have_h2"])

    summary = {
        "total": len(results),
        "extracted": extracted_count,
        "with_have": with_have_count,
        "results": results,
    }
    return summary


def main():
    logs_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOGS_PATH
    if not logs_path.is_file():
        print(f"Logs file not found: {logs_path}", file=sys.stderr)
        sys.exit(2)

    print(f"Testing full_lean_block_parser on: {logs_path}")
    print("Iteration results:")
    summary = run_tests(logs_path)
    print()
    print(f"Summary: {summary['extracted']}/{summary['total']} iterations had an extracted block; "
          f"{summary['with_have']} blocks contain 'have h1' or 'have h2' (full proof structure).")

    if summary["total"] == 0:
        sys.exit(1)
    # Consider success if we extracted at least one block and at least one has the have line.
    parser_ok = summary["extracted"] >= 1 and summary["with_have"] >= 1
    sys.exit(0 if parser_ok else 1)


if __name__ == "__main__":
    main()
