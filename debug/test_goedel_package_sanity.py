"""
TLDR: Sanity check for sdpo_modal_local_verify_goedel: parsing yields last lean4 block,
create_full_lean_code always prepends header. Run from repo root with PYTHONPATH=. .
"""

import json
from pathlib import Path


def test_parsing_last_lean4():
    """Reference run raw_output: last lean4 block is the Complete Lean 4 Proof."""
    from sdpo_modal_local_verify_goedel.parsing import extract_full_lean_block

    root = Path(__file__).resolve().parent.parent
    log_path = root / "sdpo_results" / "goedel_8b" / "run_0_20260228_211653" / "logs.json"
    if not log_path.exists():
        print("SKIP: reference logs.json not found")
        return True
    with open(log_path) as f:
        data = json.load(f)
    raw = (data.get("iteration_logs") or [{}])[0].get("raw_output") or ""
    ext = extract_full_lean_block(raw)
    assert ext.strip().lower() != "sorry", "expected non-sorry extraction"
    assert "exact h_main" in ext, "expected Complete Lean 4 Proof (exact h_main)"
    assert "sorry" not in ext, "expected complete proof without sorry"
    print("PASS: parsing returns last lean4 block (Complete Lean 4 Proof)")
    return True


def test_create_full_lean_code_always_prepends_header():
    """Goedel package always prepends header; never uses block as-is."""
    from sdpo_modal_local_verify_goedel.utils import create_full_lean_code

    default_header = "import Mathlib\n\nopen Nat"
    # Block that does NOT start with import (Goedel never outputs imports)
    block = "theorem foo : 1 = 1 := by rfl"
    full = create_full_lean_code(
        theorem_code="theorem foo : 1 = 1 := sorry",
        extracted_block=block,
        header="",
        default_header=default_header,
    )
    assert full.startswith("import Mathlib"), "expected header prepended"
    assert "theorem foo" in full and "by rfl" in full, "expected block content"
    # Must not be block-only (would happen if we used block as-is when it had no import)
    assert full.strip() != block.strip(), "expected header + block, not block only"
    print("PASS: create_full_lean_code always prepends header")
    return True


def main():
    ok = True
    ok = test_parsing_last_lean4() and ok
    ok = test_create_full_lean_code_always_prepends_header() and ok
    print("\nAll sanity checks passed." if ok else "\nSome checks failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
