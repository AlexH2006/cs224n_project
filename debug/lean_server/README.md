# Lean server verification debug logs

**TL;DR** — Kimina Lean Server verification event logs written during SDPO runs that use **Kimina** on Modal. Not used by the **local Lean** pipeline (`lean_sdpo_local_verify_modal.py`), which verifies on your machine.

## Where the logs are written

- **On Modal:** When using Kimina-backed SDPO (e.g. `lean_sdpo_kimina_2b_modal.py`, `lean_sdpo_kimina_distill_1_7b_modal.py`), the SDPOTrainer writes verification events to the **Modal volume** at:
  - `sdpo-output` volume path: `kimina_2b/debug/lean_server/` (2B) or `kimina_distill_1_7b/debug/lean_server/` (Distill 1.7B)
  - One file per run: `verify_YYYYMMDD_HHMMSS.jsonl`

- **Locally:** After syncing the Modal volume (e.g. `modal volume get sdpo-output .`), the same files appear under your local copy of the volume. You can copy them here for inspection.

## Log format (JSONL)

Each line is a JSON object.

- **`verify_start`** — Right before calling the Lean verifier:
  - `event`: `"verify_start"`
  - `ts`: ISO timestamp
  - `iteration`: SDPO iteration (1-based)
  - `attempt`: Verification attempt (1-based, within retries)
  - `code_len`: Length of the Lean code sent for verification

- **`verify_end`** — Right after the verifier returns:
  - `event`: `"verify_end"`
  - `ts`: ISO timestamp
  - `iteration`, `attempt`: Same as above
  - `duration_s`: Verification duration in seconds (from LeanVerifier)
  - `success`, `complete`: Verification result flags
  - `is_server_error`: True if the error was a server/timeout error
  - `error_snippet`: First 500 chars of feedback if any

Use these events to see how long each verification took and whether timeouts or server errors occur (e.g. on iteration 3–4).

## Related

- **Devlog:** `devlog/20260302_lean_verification_system_deep_dive.md` — Full explanation of the verification pipeline and timeout behavior.
