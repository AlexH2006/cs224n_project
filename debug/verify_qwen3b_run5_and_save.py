#!/usr/bin/env python3
"""
Verify the Lean code from sdpo_results/qwen_3b/.../run_5_20260303_022806 (full_code at line 60)
and write the verification result to debug/verify_results/.

Usage:
  python debug/verify_qwen3b_run5_and_save.py
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdpo_modal_local_verify_kimina.local_lean_verifier import verify as verify_lean

LEAN_CODE = r"""import tactic

theorem mathd_numbertheory_3 :
  (∑ x in Finset.range 10, ((x + 1)^2)) % 10 = 5 := by
  (∑ x in Finset.range (n + 1), x^2)
  let sum := sum_of_squares_of_first_n_integers 9
  sum % 10
  by
  let sum := sum_of_squares_of_first_n_integers 9
  let units_digit := sum % 10
  exact units_digit = 5
  #check mathd_numbertheory_3
"""

OUTPUT_DIR = Path(__file__).resolve().parent / "verify_results"
OUTPUT_FILE = OUTPUT_DIR / "qwen_3b_run5_mathd_numbertheory_3_verify_result.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Verifying qwen_3b run 5 Lean code (mathd_numbertheory_3)...", flush=True)
    try:
        result = verify_lean(LEAN_CODE, timeout=120, backend="kimina")
    except Exception as e:
        result = {
            "success": False,
            "complete": False,
            "feedback": str(e),
            "errors": [str(e)],
            "source": "error",
            "is_server_error": True,
        }
        print(f"Exception: {e}", file=sys.stderr)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Result written to: {OUTPUT_FILE}")
    print(f"  success={result.get('success')}, complete={result.get('complete')}")
    if result.get("feedback"):
        print(f"  feedback: {result['feedback'][:400]}")
    return 0 if result.get("success") and result.get("complete") else 1


if __name__ == "__main__":
    sys.exit(main())
