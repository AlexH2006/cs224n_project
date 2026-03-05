"""
TLDR: Run the full-block parser on every raw_output in sdpo_results/kimina_2b logs.
Validates that extract_full_lean_block behaves correctly and reports stats + examples.
Run from repo root: python debug/test_full_block_parser_kimina_2b.py [optional path]
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sdpo_modal_local_verify_kimina.parsing import extract_full_lean_block

# Default: sdpo_results/kimina_2b (all runs under it)
KIMINA_2B_ROOT = _REPO_ROOT / "sdpo_results" / "kimina_2b"


def find_all_logs(root: Path) -> list[Path]:
    """All logs.json under root."""
    return sorted(root.rglob("logs.json"))


def run_on_logs(logs_path: Path, collect_examples: bool = False, collect_all: bool = False) -> dict:
    """
    Load logs.json and run extract_full_lean_block on each iteration's raw_output.
    Returns summary for this file. If collect_examples, also return one sorry and one block example.
    If collect_all, append each result to results list (logs_path, iteration, extracted, is_sorry).
    """
    with open(logs_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    iteration_logs = logs.get("iteration_logs", [])
    if not iteration_logs:
        return {"total": 0, "n_sorry": 0, "n_block": 0, "block_has_structure": 0, "errors": [], "results": [], "example_sorry_raw": None, "example_block_raw": None, "example_block_extracted": None}

    n_sorry = 0
    n_block = 0
    block_has_structure = 0
    errors = []
    results = []
    example_sorry_raw = None
    example_block_raw = None
    example_block_extracted = None

    for entry in iteration_logs:
        raw = entry.get("raw_output", "")
        it = entry.get("iteration", 0)
        try:
            extracted = extract_full_lean_block(raw)
        except Exception as e:
            errors.append(f"iteration {it}: {e}")
            continue

        is_sorry = extracted.strip().lower() == "sorry"
        if collect_all:
            results.append({
                "logs_path": str(logs_path),
                "iteration": it,
                "is_sorry": is_sorry,
                "raw_output": raw,
                "extracted": extracted,
            })

        if is_sorry:
            n_sorry += 1
            if collect_examples and example_sorry_raw is None:
                example_sorry_raw = raw
        else:
            n_block += 1
            if "theorem " in extracted or "lemma " in extracted or "import " in extracted:
                block_has_structure += 1
            if collect_examples and example_block_raw is None:
                example_block_raw = raw
                example_block_extracted = extracted

    return {
        "total": len(iteration_logs),
        "n_sorry": n_sorry,
        "n_block": n_block,
        "block_has_structure": block_has_structure,
        "errors": errors,
        "results": results,
        "example_sorry_raw": example_sorry_raw,
        "example_block_raw": example_block_raw,
        "example_block_extracted": example_block_extracted,
    }


# Default path for saved JSON (relative to this file)
DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parent / "parsing_test_kimina_2b_results.json"


def main() -> int:
    argv = sys.argv[1:]
    out_path = None
    if "-o" in argv:
        idx = argv.index("-o")
        if idx + 1 < len(argv):
            out_path = Path(argv[idx + 1])
    skip = {"-o"}
    if out_path is not None:
        skip.add(argv[argv.index("-o") + 1])
    args = [a for a in argv if a not in skip]
    root = Path(args[0]) if args else KIMINA_2B_ROOT
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    logs_files = find_all_logs(root)
    if not logs_files:
        print(f"No logs.json under {root}", file=sys.stderr)
        return 1

    print(f"Found {len(logs_files)} logs under {root}")
    total_iters = 0
    total_sorry = 0
    total_block = 0
    total_structure = 0
    all_errors = []
    all_results = []
    example_sorry_raw = None
    example_block_raw = None
    example_block_extracted = None

    for path in logs_files:
        result = run_on_logs(
            path,
            collect_examples=(example_sorry_raw is None or example_block_raw is None),
            collect_all=True,
        )
        total_iters += result["total"]
        total_sorry += result["n_sorry"]
        total_block += result["n_block"]
        total_structure += result["block_has_structure"]
        all_errors.extend(result["errors"])
        all_results.extend(result["results"])
        if example_sorry_raw is None and result.get("example_sorry_raw"):
            example_sorry_raw = result["example_sorry_raw"]
        if example_block_raw is None and result.get("example_block_raw"):
            example_block_raw = result["example_block_raw"]
            example_block_extracted = result.get("example_block_extracted")

    print(f"Total iterations: {total_iters}")
    print(f"  → sorry:      {total_sorry}")
    print(f"  → full block: {total_block}")
    print(f"  → block has theorem/lemma/import: {total_structure}")

    # Build payload to save
    save_path = out_path or DEFAULT_OUTPUT_JSON
    payload = {
        "summary": {
            "root": str(root),
            "n_logs": len(logs_files),
            "total_iterations": total_iters,
            "n_sorry": total_sorry,
            "n_full_block": total_block,
            "n_block_has_structure": total_structure,
            "n_errors": len(all_errors),
        },
        "example_sorry": {
            "raw_output": example_sorry_raw,
            "extracted": extract_full_lean_block(example_sorry_raw) if example_sorry_raw else None,
        } if example_sorry_raw else None,
        "example_full_block": {
            "raw_output": example_block_raw,
            "extracted": example_block_extracted,
        } if (example_block_raw and example_block_extracted) else None,
        "results": all_results,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {save_path}")

    # Show parsing examples on stdout
    print("\n--- Example: parsing → 'sorry' (incomplete or no block) ---")
    if example_sorry_raw is not None:
        preview = example_sorry_raw[:600].replace("\n", " ")
        print(f"raw_output (first 600 chars): {preview}...")
        print(f"extract_full_lean_block(...) → {repr(payload['example_sorry']['extracted'])}")
    else:
        print("(no sorry example in this run)")

    print("\n--- Example: parsing → full block ---")
    if example_block_raw is not None and example_block_extracted is not None:
        preview = example_block_raw[:500].replace("\n", " ")
        print(f"raw_output (first 500 chars): {preview}...")
        print(f"extract_full_lean_block(...) → (first 800 chars of block):")
        print(example_block_extracted[:800])
        if len(example_block_extracted) > 800:
            print("...")
    else:
        print("(no full-block example in this run)")

    if all_errors:
        print(f"\nErrors ({len(all_errors)}):", file=sys.stderr)
        for e in all_errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more", file=sys.stderr)
        return 1

    if total_iters == 0:
        return 1

    print("\nOK: no parser errors; stats above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
