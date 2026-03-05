#!/usr/bin/env python3
"""
TL;DR: Contract tests for sdpo_modal.local_lean_verifier. Run from repo root.

Verifies that the local Lean verifier returns VerifyResult-shaped dicts and that
known-good and known-bad snippets produce expected success/complete/errors.
Requires: lake, mathlib4 built (see devlog/20260303_lean_verification_sdpo_integration_plan.md).
"""
import sys

# Run from repo root so sdpo_modal is importable
sys.path.insert(0, ".")

from sdpo_modal.local_lean_verifier import verify


# Minimal Lean 4 that should compile and be complete (no sorry)
GOOD_SNIPPET = """import Mathlib

theorem foo : True := trivial
"""

# Lean 4 that should fail verification (False cannot be proved by trivial)
BAD_SNIPPET = """import Mathlib

theorem bar : False := trivial
"""


def test_verify_result_contract(result: dict) -> None:
    """Assert result has required VerifyResult keys and types."""
    assert isinstance(result, dict), "Result must be a dict"
    required = {"success", "complete", "has_sorry", "feedback", "errors", "source"}
    for key in required:
        assert key in result, f"Missing key: {key}"
    assert isinstance(result["success"], bool), "success must be bool"
    assert isinstance(result["complete"], bool), "complete must be bool"
    assert isinstance(result["errors"], list), "errors must be list"
    assert result["source"] == "local_lean", "source must be local_lean"


def _env_likely_missing(result: dict) -> bool:
    """True if failure looks like missing lake/mathlib4 (so good/bad snippet tests are skipped)."""
    if result.get("success"):
        return False
    fb = (result.get("feedback") or "").lower()
    return "no such file" in fb or "not found" in fb or "no such file" in str(result.get("errors", [])).lower()


def test_known_good():
    """Known-good snippet should yield success=True, complete=True (skipped if env missing)."""
    result = verify(GOOD_SNIPPET, timeout=60)
    test_verify_result_contract(result)
    if _env_likely_missing(result):
        print("SKIP: known-good (Lean env not set up: missing lake/mathlib4)")
        return
    assert result["success"] is True, f"Expected success=True, got {result}"
    assert result["complete"] is True, f"Expected complete=True, got {result}"
    assert result["errors"] == [], f"Expected no errors, got {result['errors']}"
    print("PASS: known-good snippet -> success=True, complete=True")


def test_known_bad():
    """Known-bad snippet should yield success=False and non-empty errors (skipped if env missing)."""
    result = verify(BAD_SNIPPET, timeout=60)
    test_verify_result_contract(result)
    if _env_likely_missing(result):
        print("SKIP: known-bad (Lean env not set up)")
        return
    assert result["success"] is False, f"Expected success=False, got {result}"
    assert len(result["errors"]) > 0, f"Expected non-empty errors, got {result['errors']}"
    assert result["feedback"], "Expected non-empty feedback"
    print("PASS: known-bad snippet -> success=False, non-empty errors")


def test_error_path_contract():
    """When workspace is invalid, verify still returns a valid VerifyResult (contract holds)."""
    import os
    result = verify(GOOD_SNIPPET, timeout=5, lean_workspace="/nonexistent_mathlib4_path_12345")
    test_verify_result_contract(result)
    assert result["success"] is False, "Invalid workspace must yield success=False"
    assert result["errors"], "Invalid workspace must yield non-empty errors or feedback"
    print("PASS: error path returns valid VerifyResult contract")


def main():
    print("Contract tests for local_lean_verifier (requires lake + mathlib4 for good/bad snippet tests)...")
    try:
        test_error_path_contract()
        test_known_good()
        test_known_bad()
        print("\nAll contract tests passed.")
        return 0
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
