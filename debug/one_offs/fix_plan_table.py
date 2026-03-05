#!/usr/bin/env python3
"""
One-off: rewrite Phase 3 (items 1–3) in the integration plan devlog.
Run from repo root: python debug/one_offs/fix_plan_table.py
"""
path = "devlog/20260303_lean_verification_sdpo_integration_plan.md"
with open(path, "r") as f:
    s = f.read()

# Phase 3: replace old items 1-3 with new short text
old_phase3 = """1. **Keep existing Kimina app**
   - Leave `modal_app.py` (or current app module) **unchanged**: same app name (e.g. `lean-sdpo-kimina`), `KiminaLeanServer` image, `LeanVerifier`, `SDPOTrainer` with full `run_sdpo` that calls Kimina. This remains the "all-on-Modal with Kimina verify" path.

2. **New Modal app for local-compile**
   - Create a **separate** Modal app (e.g. in `sdpo_modal/modal_app_local_compile.py` or a second app in the same file) with a **distinct app name** (e.g. `lean-sdpo-local-compile`).
   - This app uses only the **inference image** (no Kimina image). It defines a single class: `SDPOTrainer` (or `SDPOTrainerLocalCompile`) with `generate_batch` and `run_sdpo_step` only — no `KiminaLeanServer`, no `LeanVerifier`. Same volumes/secrets as needed for model and output.
   - Entrypoint for local-compile (e.g. `entrypoint_local_verify.py`) imports and uses **this** app's trainer, not the Kimina app's.

3. **Documentation**
   - Document two paths: (A) Kimina-on-Modal (existing); (B) Local compilation (new app + local verifier). No removal or deprecation of Kimina."""

new_phase3 = """1. **Original folder:** Leave `sdpo_modal/` **unchanged**. Kimina app and full `run_sdpo` remain as-is.

2. **New folder's `modal_app.py`:** In the copy, remove Kimina image, `KiminaLeanServer`, `LeanVerifier`. Keep only inference image and a trainer with `generate_batch` and `run_sdpo_step`; distinct app name (e.g. `lean-sdpo-local-compile`).

3. **New folder's entrypoint:** Runs the batched local loop (generate_batch on Modal, verify locally, run_sdpo_step on Modal).

4. **Documentation:** Two pipelines: (A) Original = Kimina-on-Modal; (B) New folder = local verification + batched generate/step."""

if old_phase3 in s:
    s = s.replace(old_phase3, new_phase3)
else:
    # Try with Unicode apostrophe
    old_phase3_unicode = old_phase3.replace("app's", "app\u2019s")
    if old_phase3_unicode in s:
        s = s.replace(old_phase3_unicode, new_phase3)
    else:
        print("Phase 3 block not found")
with open(path, "w") as f:
    f.write(s)
print("Done.")
