#!/usr/bin/env python3
"""
Verify the Lean code from sdpo_results/.../run_5_20260304_112601 (mathd_numbertheory_3)
and write the verification result to debug/verify_results/.

Uses sdpo_modal_local_verify_kimina.local_lean_verifier.verify() so backend is controlled by
LEAN_VERIFY_BACKEND (default kimina). Run with Kimina server up, e.g.:
  docker run --rm -p 8000:8000 projectnumina/kimina-lean-server:2.0.0

Usage:
  python debug/verify_run5_lean_and_save.py
  python debug/verify_run5_lean_and_save.py --file debug/lean_samples/mathd_numbertheory_3_from_run5.lean
"""
import argparse
import json
import sys
from pathlib import Path

# Repo root so we can import sdpo_modal_local_verify_kimina
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sdpo_modal_local_verify_kimina.local_lean_verifier import verify as verify_lean


LEAN_CODE_FROM_RUN5 = r"""import Mathlib

theorem mathd_numbertheory_3 :
  (∑ x in Finset.range 10, ((x + 1)^2)) % 10 = 5 := by
  norm_num [Finset.sum_range_succ]
"""

OUTPUT_DIR = Path(__file__).resolve().parent / "verify_results"
OUTPUT_FILE = OUTPUT_DIR / "mathd_numbertheory_3_verify_result.json"


def main():
    parser = argparse.ArgumentParser(description="Verify run5 Lean code and save result to debug/verify_results/")
    parser.add_argument("--file", "-f", help="Read Lean code from file (default: use embedded code from run 5)")
    parser.add_argument("--backend", choices=("local", "kimina"), default=None, help="Verification backend (default: env LEAN_VERIFY_BACKEND or kimina)")
    parser.add_argument("--timeout", type=int, default=120, help="Verification timeout in seconds")
    args = parser.parse_args()

    if args.file:
        lean_code = Path(args.file).read_text()
    else:
        lean_code = LEAN_CODE_FROM_RUN5

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Verifying Lean code (mathd_numbertheory_3 from run 5)...", flush=True)
    print(f"  Backend: {args.backend or 'from env (default kimina)'}", flush=True)
    try:
        result = verify_lean(
            lean_code,
            timeout=args.timeout,
            backend=args.backend or "kimina",
        )
    except Exception as e:
        result = {
            "success": False,
            "complete": False,
            "has_sorry": False,
            "feedback": str(e),
            "errors": [str(e)],
            "source": "error",
            "is_server_error": True,
            "debug": {"exception": str(e)},
        }
        print(f"Verification raised: {e}", file=sys.stderr)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Result written to: {OUTPUT_FILE}", flush=True)
    print(f"  success={result.get('success')}, complete={result.get('complete')}, source={result.get('source')}", flush=True)
    if result.get("feedback"):
        print(f"  feedback: {result['feedback'][:200]}...", flush=True)
    return 0 if result.get("success") and result.get("complete") else 1


if __name__ == "__main__":
    sys.exit(main())
