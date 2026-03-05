"""
TLDR: Run Goedel-style parsing (last lean4 block only) and Kimina-style parsing
over all raw_outputs in sdpo_results/goedel_8b and report correctness.

Usage: python debug/test_goedel_parsing_on_logs.py
"""

import json
import re
from pathlib import Path

# Goedel: only match ```lean4 blocks, take the last one.
_CODE_BLOCK_LEAN4_ONLY = re.compile(
    r"```lean4\s*\n(.*?)```",
    re.DOTALL,
)


def _has_incomplete_reasoning(text: str) -> bool:
    return "<think>" in text and "</think>" not in text


def _search_region(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def extract_full_lean_block_goedel(raw_output: str) -> str:
    """Extract the last lean4 fenced block only (Goedel output has multiple sections)."""
    if not raw_output or not raw_output.strip():
        return "sorry"
    text = raw_output.strip()
    if _has_incomplete_reasoning(text):
        return "sorry"
    region = _search_region(text)
    matches = _CODE_BLOCK_LEAN4_ONLY.findall(region)
    if not matches:
        return "sorry"
    content = matches[-1].strip()
    return content if content else "sorry"


def count_lean4_blocks(raw_output: str) -> int:
    return len(_CODE_BLOCK_LEAN4_ONLY.findall(raw_output or ""))


def main():
    from sdpo_modal_local_verify_kimina.parsing import extract_full_lean_block as extract_kimina

    root = Path(__file__).resolve().parent.parent / "sdpo_results" / "goedel_8b"
    if not root.exists():
        print(f"Directory not found: {root}")
        return

    log_files = sorted(root.rglob("logs.json"))
    print(f"Found {len(log_files)} logs under {root}\n")

    total = 0
    goedel_ok = 0
    kimina_ok = 0
    same = 0
    goedel_complete_proof = 0  # has exact and no sorry (or minimal)
    issues = []

    for log_path in log_files:
        with open(log_path) as f:
            data = json.load(f)
        run_name = log_path.parent.name
        for entry in data.get("iteration_logs") or []:
            raw = entry.get("raw_output") or ""
            if not raw.strip():
                continue
            total += 1
            n_blocks = count_lean4_blocks(raw)
            ext_goedel = extract_full_lean_block_goedel(raw)
            ext_kimina = extract_kimina(raw)

            is_goedel_sorry = (ext_goedel or "").strip().lower() == "sorry"
            is_kimina_sorry = (ext_kimina or "").strip().lower() == "sorry"
            if not is_goedel_sorry:
                goedel_ok += 1
            if not is_kimina_sorry:
                kimina_ok += 1
            if ext_goedel == ext_kimina:
                same += 1
            # "Complete proof" = extracted block contains "exact" (e.g. exact h_main) and no "sorry"
            if not is_goedel_sorry and "exact" in ext_goedel and "sorry" not in ext_goedel:
                goedel_complete_proof += 1

            if n_blocks > 1 and ext_goedel != ext_kimina:
                issues.append(
                    f"{run_name} iter {entry.get('iteration')}: {n_blocks} lean4 blocks; Kimina != Goedel"
                )
            elif is_goedel_sorry and n_blocks > 0:
                issues.append(
                    f"{run_name} iter {entry.get('iteration')}: {n_blocks} lean4 blocks but Goedel got sorry"
                )

    print("Results (Goedel = last lean4 block only, Kimina = last lean4/lean/tactics)")
    print(f"  Total iteration outputs: {total}")
    print(f"  Goedel extracted non-sorry: {goedel_ok}")
    print(f"  Kimina extracted non-sorry: {kimina_ok}")
    print(f"  Same extraction (Goedel == Kimina): {same}")
    print(f"  Goedel extraction looks complete (has 'exact', no 'sorry'): {goedel_complete_proof}")
    if issues:
        print(f"\nPotential issues ({len(issues)}):")
        for i in issues[:20]:
            print(f"  - {i}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("\nNo issues recorded.")

    # Sanity: reference run should yield the "Complete Lean 4 Proof" (exact h_main, no sorry)
    ref_log = root / "run_0_20260228_211653" / "logs.json"
    if ref_log.exists():
        with open(ref_log) as f:
            ref = json.load(f)
        for entry in ref.get("iteration_logs") or []:
            raw = entry.get("raw_output") or ""
            if not raw.strip():
                continue
            ext = extract_full_lean_block_goedel(raw)
            has_exact = "exact h_main" in ext
            has_sorry = "sorry" in ext
            print("\nReference run_0_20260228_211653 (mathd_algebra_478):")
            print(f"  Goedel extraction: {len(ext)} chars, 'exact h_main'={has_exact}, contains 'sorry'={has_sorry}")
            if has_exact and not has_sorry:
                print("  OK: Last lean4 block is the complete proof.")
            else:
                print("  Sample (first 400 chars):", repr(ext[:400]))
    print("\nDone.")


if __name__ == "__main__":
    main()
