# Debug

**TL;DR** — Scripts, tests, and artifacts for developing and debugging the SDPO + Lean verification pipeline. Nothing here is required at runtime; use for local checks, one-off fixes, and inspecting verification behavior.

## Layout

| Path | Purpose |
|------|--------|
| **`test_local_verify_pipeline.py`** | Smoke tests for the **`sdpo_modal_local_verify`** pipeline (config, payload, verifier contract, parse/full_code). No Modal. Run from repo root. |
| **`test_local_lean_verifier.py`** | Contract tests for **`sdpo_modal.local_lean_verifier`**: VerifyResult shape, known-good/bad Lean snippets, error paths. Requires lake + mathlib4 for full pass. Run from repo root. |
| **`test_goedel_package_sanity.py`** | Sanity checks for **`sdpo_modal_local_verify_goedel`**: config, prompt, parsing, full_code assembly. No Modal or Lean required. |
| **`test_goedel_parsing_on_logs.py`** | Runs `extract_full_lean_block` and `is_truncated_output` against real `logs.json` files from `sdpo_results/`. Prints extracted blocks and flags mismatches. |
| **`test_full_lean_block_parser.py`** | Unit tests for `full_lean_block_parser.py` (standalone parser prototype). |
| **`test_full_block_parser_kimina_2b.py`** | Runs the full-block parser against Kimina 2B result logs. |
| **`test_pipeline_full_block_kimina_2b.py`** | End-to-end pipeline test using full-block parsing on Kimina 2B outputs. |
| **`full_lean_block_parser.py`** | Standalone prototype for last-lean4-block extraction (used by Goedel pipeline). |
| **`run_verifier_tests.py`** | Single entry point: runs the main verifier test scripts. |
| **`verify_one_lean.py`** | CLI to verify one Lean 4 file (or stdin) via Kimina Lean Server HTTP API. Sample files in `lean_samples/`. |
| **`verify_run5_lean_and_save.py`** | Verify a specific run's Lean outputs and save results. |
| **`verify_qwen3b_run5_and_save.py`** | Verify Qwen 3B run 5 Lean outputs and save results. |
| **`lean_server/`** | Docs for Kimina verification debug logs on Modal (volume paths, JSONL format). |
| **`lean_samples/`** | Sample `.lean` files for manual or CLI verification; `known_bad/` has minimal failing examples. |
| **`verify_results/`** | Saved Kimina server JSON responses for reference. |
| **`one_offs/`** | Single-use scripts (e.g. doc updaters); run from repo root when needed. |

The **local Lean** verifier is implemented in `sdpo_modal_local_verify_goedel.local_lean_verifier`, `sdpo_modal_local_verify_kimina.local_lean_verifier`, and `sdpo_modal_local_verify.local_lean_verifier`; all share the same `VerifyResult` contract. The Goedel variant (`sdpo_modal_local_verify_goedel`) uses last-lean4-block parsing with truncation detection (`parsing.py`).

## Running tests

From repo root:

```bash
python debug/run_verifier_tests.py
# or individually:
python debug/test_local_verify_pipeline.py
python debug/test_local_lean_verifier.py      # needs lake + mathlib4 for good/bad snippet tests
python debug/test_goedel_package_sanity.py    # no Modal or Lean required
python debug/test_goedel_parsing_on_logs.py   # reads sdpo_results/ logs, no Lean required
```

## Verifying a Lean file with Kimina

Server must be running (e.g. Docker). Sample files are in `debug/lean_samples/`:

```bash
python debug/verify_one_lean.py --file debug/lean_samples/mathd_algebra_478.lean
python debug/verify_one_lean.py -f debug/lean_samples/imosl_2007_algebra_p6.lean -o debug/verify_results/out.json
```

## Related

- **Integration plan:** `devlog/20260303_lean_verification_sdpo_integration_plan.md`
- **Local verifier setup:** `devlog/20260303_local_lean_verifier_setup.md`
- **Parsing notes:** `devlog/20260304_parsing_central.md`
- **macOS REPL issue:** `devlog/20260304_dyld_data_const_macos_repl.md`
- **mathlib4 setup:** `devlog/20260303_mathlib4_missing_file_not_found.md`
