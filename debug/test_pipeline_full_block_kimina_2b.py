"""
TLDR: Run the full pipeline (extract_full_lean_block + create_full_lean_code) on every
raw_output in sdpo_results/kimina_2b logs. Validates that assembly produces non-empty
full_code and no exceptions. Run from repo root: python debug/test_pipeline_full_block_kimina_2b.py
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sdpo_modal_local_verify_kimina.parsing import extract_full_lean_block
from sdpo_modal_local_verify_kimina.utils import create_full_lean_code, get_field

KIMINA_2B_ROOT = _REPO_ROOT / "sdpo_results" / "kimina_2b"
DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parent / "pipeline_full_block_kimina_2b_results.json"


def find_all_logs(root: Path) -> list[Path]:
    return sorted(root.rglob("logs.json"))


def run_pipeline_on_logs(logs_path: Path) -> dict:
    """Run extract_full_lean_block + create_full_lean_code on each iteration. Return stats and any errors."""
    with open(logs_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    problem = logs.get("problem", {})
    config = logs.get("config", {})
    theorem_fields = config.get("theorem_fields", ["formal_statement", "lean4_code", "statement"])
    header_fields = config.get("header_fields", ["header", "imports", "preamble"])
    default_header = config.get("default_header", "")

    iteration_logs = logs.get("iteration_logs", [])
    if not iteration_logs:
        return {"total": 0, "ok": 0, "errors": [], "results": []}

    theorem_code = get_field(problem, theorem_fields)
    header = get_field(problem, header_fields)

    results = []
    errors = []
    ok = 0

    for entry in iteration_logs:
        raw = entry.get("raw_output", "")
        it = entry.get("iteration", 0)
        try:
            extracted_block = extract_full_lean_block(raw)
            full_code = create_full_lean_code(
                theorem_code=theorem_code,
                extracted_block=extracted_block,
                header=header,
                default_header=default_header,
            )
        except Exception as e:
            errors.append({"logs_path": str(logs_path), "iteration": it, "error": str(e)})
            results.append({"logs_path": str(logs_path), "iteration": it, "ok": False, "error": str(e)})
            continue

        if not full_code or not full_code.strip():
            # Empty full_code can happen when problem has no theorem/header (e.g. old PutnamBench config).
            if not theorem_code.strip():
                results.append({"logs_path": str(logs_path), "iteration": it, "ok": True, "empty_theorem_data": True})
                ok += 1
            else:
                errors.append({"logs_path": str(logs_path), "iteration": it, "error": "empty full_code"})
                results.append({"logs_path": str(logs_path), "iteration": it, "ok": False, "error": "empty full_code"})
            continue

        ok += 1
        results.append({
            "logs_path": str(logs_path),
            "iteration": it,
            "ok": True,
            "full_code_len": len(full_code),
            "is_sorry": extracted_block.strip().lower() == "sorry",
        })

    return {"total": len(iteration_logs), "ok": ok, "errors": errors, "results": results}


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else KIMINA_2B_ROOT
    out_path = DEFAULT_OUTPUT_JSON
    if "-o" in sys.argv:
        idx = sys.argv.index("-o")
        if idx + 1 < len(sys.argv):
            out_path = Path(sys.argv[idx + 1])

    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    logs_files = find_all_logs(root)
    if not logs_files:
        print(f"No logs.json under {root}", file=sys.stderr)
        return 1

    print(f"Running pipeline on {len(logs_files)} logs under {root}")
    total_iters = 0
    total_ok = 0
    all_errors = []
    all_results = []

    for path in logs_files:
        r = run_pipeline_on_logs(path)
        total_iters += r["total"]
        total_ok += r["ok"]
        all_errors.extend(r["errors"])
        all_results.extend(r["results"])

    payload = {
        "summary": {
            "root": str(root),
            "n_logs": len(logs_files),
            "total_iterations": total_iters,
            "ok": total_ok,
            "errors": len(all_errors),
        },
        "results": all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_path}")

    print(f"Total iterations: {total_iters}")
    print(f"  → OK (non-empty full_code): {total_ok}")
    print(f"  → Errors: {len(all_errors)}")

    if all_errors:
        for e in all_errors[:15]:
            print(f"  {e}", file=sys.stderr)
        if len(all_errors) > 15:
            print(f"  ... and {len(all_errors) - 15} more", file=sys.stderr)
        return 1

    print("OK: pipeline ran without errors on all iterations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
