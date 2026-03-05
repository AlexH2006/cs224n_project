# Lean Verification + SDPO Modal Pipeline Integration Plan

**TLDR:** Do **not** change the existing pipeline code. **Copy** the entire pipeline folder (e.g. `sdpo_modal/`) into a **new folder** (e.g. `sdpo_modal_local_verify/`) with everything in it. All changes for local verification and the batched split loop happen **only inside the new folder**. The original folder stays completely unchanged. Two pipelines: (A) original folder = Kimina-on-Modal, unchanged; (B) new folder = local verification + batched generate/step, separate Modal app.

---

## 1. Current State (Findings)

### 1.1 Goedel-Prover-main

| Aspect | Details |
|--------|--------|
| **Verification** | `prover/lean/verifier.py`: `verify_lean4_file(code, ...)` runs `lake exe repl` in `mathlib4/` with one JSON command per run. No Kimina; no HTTP server. |
| **Lean env** | Local toolchain: `~/.elan/bin/lake`, workspace `mathlib4/` (git submodule). Setup: install Lean 4 via elan, `cd mathlib4 && lake build`. REPL binary is from the mathlib4 fork (`lake exe repl`). |
| **Input** | Single Lean 4 source string (full file content). |
| **Output** | Dict: `pass` (no errors), `complete` (pass + no sorries + no sorry/failed in warnings), `errors`, `sorries`, `warnings`, `system_errors`, `verify_time`, etc. |
| **Scheduler** | `Lean4ServerScheduler` in same file: multiprocessing pool of `Lean4ServerProcess` workers; each worker calls `verify_lean4_file(**task)`. Used for batch eval in `eval/step2_compile.py`. |

### 1.2 sdpo_modal

| Aspect | Details |
|--------|--------|
| **Pipeline** | Entrypoint (local) → `trainer.run_sdpo.remote(config_dict, problem)` (Modal). Entire SDPO loop runs **on Modal**: generate → parse → full_code → verify → feedback → loss → step. |
| **Verification** | `LeanVerifier` (Modal) → `KiminaLeanServer.verify.remote(lean_code)` (Modal). Kimina is a separate Modal class with image `projectnumina/kimina-lean-server:2.0.0`. |
| **Parsing** | `sdpo_modal/utils.py`: `extract_proof_tactics()` (from `<think>`/code blocks/`:= by`), `extract_tactics_from_code_block()`, `create_full_lean_code()` (theorem + tactics + header). Same parsing runs inside Modal in `trainer_core.run_sdpo`. |
| **Result contract** | `lean_verification.VerifyResult`: `success`, `complete`, `has_sorry`, `feedback`, `errors`, `messages`, `sorries`, `source`, `is_server_error`, etc. Kimina response is normalized by `parse_kimina_response()`. |

### 1.3 Why “send compilation results back to Modal” is possible

Modal **cannot** call back to the user’s machine. So “local compile → send results back to Modal” means:

- The **loop driver** runs **locally**.
- Each iteration: (1) Local calls Modal to **generate** → get back `(raw_text, generated_ids)`. (2) Local **parses** and **verifies** (Goedel). (3) Local calls Modal to run **one SDPO step** (loss + backward + optimizer step), passing in **compilation/verification result** plus prompts and token ids. Modal returns; no call from Modal to local.

So the pipeline becomes: **Modal generate → local compile → send compilation results (and related data) to Modal for training step.**

### 1.4 On-policy RL and low latency (constraint)

You are running **on-policy RL**: the policy that generated the data should be updated quickly so it doesn’t drift. That implies:

- **Loss is computed from each error feedback** (verification result) — already the case: SDPO uses verification to build teacher feedback and then computes KL(student ‖ teacher) on the generated response.
- **Gradient updates should be frequent** — e.g. every **N solutions** (N = 4 as a placeholder), not one update per “problem solved” or at the end of a long run.

To keep **latency low** with a split pipeline (Modal ↔ local), we must **minimize round-trips**:

- **Avoid:** 1 generate → 1 verify → 1 step (2 round-trips per gradient update). If we did one update per solution, that’s 2 round-trips per solution and high latency.
- **Prefer:** **Batch per update step:** request **N solutions** in one (or few) Modal call(s), verify all N locally, then send **one batch** to Modal for **one** gradient update. That gives **2 round-trips per N solutions** (one for “generate N”, one for “run_sdpo_step with batch of N”).

So the design should support:

1. **Batch generation:** `generate_batch(prompts: list[str])` on Modal returns a list of `(raw_text, generated_ids)` so we get N solutions in one round-trip (or one call that runs N generations in parallel on the same container).
2. **Batch step:** `run_sdpo_step(config, problem, batch_payload)` where `batch_payload` is a **list of N** iteration payloads; on Modal we compute loss as **mean over the N** samples, then one `backward` and one `optimizer.step()`.
3. **Config:** e.g. `gradient_update_every_N: int = 4` (or `batch_size`) so the local loop collects N solutions, verifies all N, then sends one batch for one update.

This keeps the loop on-policy (update every N solutions), uses each solution’s error feedback for the loss, and keeps latency low by batching at the Modal boundary.

---

## 2. Strategy: Copy folder, modify only the copy

**Rule:** The **original** pipeline folder (e.g. `sdpo_modal/`) is **never modified**. All work for the local-verification pipeline is done in a **new folder** that is a full copy of the original.

1. **Copy**
   - Copy the entire pipeline folder to a new name, e.g. `sdpo_modal/` → `sdpo_modal_local_verify/` (or `sdpo_modal_local_compile/`). The new folder contains every file from the original: `config.py`, `entrypoint.py`, `modal_app.py`, `modal_trainer.py`, `trainer_core.py`, `utils.py`, `prompts.py`, `lean_verification.py`, `sdpo_loss.py`, etc.

2. **Imports / package name**
   - In the new folder, update internal imports to use the new package name (e.g. `sdpo_modal_local_verify.config` instead of `sdpo_modal.config`). Any runner script that launches the local-verify pipeline will import from the new folder (e.g. `from sdpo_modal_local_verify.entrypoint import run_main`).

3. **Changes only in the new folder**
   - Add `local_lean_verifier.py` (or copy from repo root if it stays shared; see below).
   - In the new folder’s `modal_app.py`: remove Kimina image, `KiminaLeanServer`, `LeanVerifier`; keep only inference image and a trainer that exposes `generate_batch` and `run_sdpo_step` (no `run_sdpo` that calls a verifier).
   - In the new folder’s `modal_trainer.py`: implement `generate_batch` and `run_sdpo_step`; remove or do not use LeanVerifier.
   - In the new folder’s `trainer_core.py` / entrypoint: implement the **local-driven batched loop** (local calls Modal for generate_batch, local verifies, local calls Modal for run_sdpo_step).
   - In the new folder’s `config.py`: add `batch_size` (e.g. 4).
   - In the new folder’s `sdpo_loss.py`: add batched loss path (or average over N in the trainer).
   - New folder’s entrypoint (or single main entry) runs the batched local loop and uses the local verifier.

4. **Shared vs copied**
   - **Option A:** `local_lean_verifier.py` lives once at repo root (e.g. `sdpo_modal/local_lean_verifier.py` as today) and the **new folder** imports it via `from sdpo_modal.local_lean_verifier import verify` (so the new folder depends on the original package for that one module).  
   - **Option B:** Copy `local_lean_verifier.py` into the new folder so the new folder is **fully self-contained** and does not import from the original folder. Then the original folder can stay 100% untouched and the new folder can be renamed/moved independently.  
   - Plan assumes **Option B** for maximum isolation: copy `local_lean_verifier.py` into the new folder and update its imports (e.g. `from sdpo_modal_local_verify.lean_verification import ...`).

5. **Runner scripts**
   - Existing runner (e.g. `training/lean_sdpo_kimina_2b_modal.py`) continues to use the **original** folder (`sdpo_modal`); no changes.
   - New runner (e.g. `training/lean_sdpo_local_verify_modal.py`) uses the **new** folder (`sdpo_modal_local_verify`): imports the new app and entrypoint, runs the local-verify pipeline.

---

## 3. Target Architecture (MVP)

### 3.1 High-level flow (with batching for low latency)

**Per gradient-update step** (every N solutions, e.g. N=4):

```
[Local]
  load dataset, config, problem
  gradient_update_every_N = config.batch_size  # e.g. 4
  for update_step in 1..max_updates (or until success):
    1. prompts = [base_prompt] * N   # or N variants if using prior feedback
    2. batch_generations = Modal.SDPOTrainer.generate_batch.remote(prompts)
       → list of (raw_text, generated_ids) of length N
    3. For each of the N: parse → full_code → verification = local_verifier.verify(full_code)
    4. Build N iteration payloads (feedback_history, teacher_prompt, verification, etc.)
    5. step_logs = Modal.SDPOTrainer.run_sdpo_step.remote(config, problem, batch_payload)
       where batch_payload = list of N payloads; Modal computes loss = mean over N, one backward, one step
    6. If any solution succeeded or max_updates: break
  save local copy of logs
```

- **On Modal (local-compile app only):** A **separate Modal app** (separate image from the Kimina flow) runs `SDPOTrainer` with **`generate_batch(prompts: list[str])`** and **`run_sdpo_step(config, problem, batch_payload)`** only. This app has **no** `KiminaLeanServer` or `LeanVerifier`; verification is not on Modal for this path.
- **Existing Kimina app:** Unchanged. Keeps `KiminaLeanServer` image and `LeanVerifier`; full loop can still run on Modal with Kimina verification.
- **Locally (for local-compile path):** parsing (existing `sdpo_modal.utils`), plus a **local verifier** that wraps Goedel’s `lake exe repl` and returns a `VerifyResult`-shaped dict. Optional: verify N in parallel (e.g. Goedel’s `Lean4ServerScheduler` with N workers) to keep wall-clock time low.
</think>

### 3.2 Components (all in the new folder only; original folder unchanged)

| Component | Location (all under new folder, e.g. `sdpo_modal_local_verify/`) | Purpose |
|-----------|------------------------------------------------------------------|--------|
| **Local Goedel verifier adapter** | `local_lean_verifier.py` (copy from `sdpo_modal/`) | Same as Phase 1; update imports to use new package's `lean_verification`. |
| **Modal app** | `modal_app.py` (modified copy) | No Kimina. Inference image only; trainer with `generate_batch` and `run_sdpo_step`; distinct app name (e.g. `lean-sdpo-local-compile`). |
| **Modal trainer** | `modal_trainer.py` (modified copy) | Implement `generate_batch(prompts)` and `run_sdpo_step(config, problem, batch_payload)`; do not use LeanVerifier or Kimina. |
| **Batched SDPO loss** | `sdpo_loss.py` (modified copy) | Add batched path: average loss over N and one backward/step per batch. |
| **Config** | `config.py` (modified copy) | Add `batch_size` (e.g. 4). |
| **Loop runner** | `entrypoint.py` (modified copy) | Batched local loop: Modal `generate_batch` -> local parse + verify -> Modal `run_sdpo_step`. |
| **Trainer core** | `trainer_core.py` (copy; optionally refactor) | Reuse parsing/feedback logic for the local loop; batched step runs on Modal only. |
| **Original folder** | `sdpo_modal/` | **No changes.** Keeps Kimina image, `KiminaLeanServer`, `LeanVerifier`, full `run_sdpo`. |

---

## 4. Detailed Plan

### Phase 0: Copy folder (do first)

**Goal:** Create the new pipeline as a full copy of the original; all subsequent changes happen only in the copy.

1. **Copy the pipeline folder**
   - Copy `sdpo_modal/` in full to a new folder, e.g. `sdpo_modal_local_verify/`. Include every file: `config.py`, `entrypoint.py`, `modal_app.py`, `modal_trainer.py`, `trainer_core.py`, `utils.py`, `prompts.py`, `lean_verification.py`, `sdpo_loss.py`, and any `__init__.py`.
   - Copy `sdpo_modal/local_lean_verifier.py` into the new folder (it already exists in `sdpo_modal/` from Phase 1).
   - Update all **internal imports** in the new folder from `sdpo_modal.*` to the new package name (e.g. `sdpo_modal_local_verify.*`). Ensure the new folder is a valid Python package and can be imported (e.g. `from sdpo_modal_local_verify.entrypoint import run_main`).
   - **Do not modify any file in the original `sdpo_modal/`.**

### Phase 1: Local verification adapter (already done in sdpo_modal; copy into new folder)

**Goal:** The new folder needs a local verifier; use the one already implemented in `sdpo_modal/local_lean_verifier.py` (copy it in during Phase 0) and fix its imports to use the new package's `lean_verification`.

1. **In the new folder: `local_lean_verifier.py`**
   - Dependencies: use Goedel’s `verify_lean4_file` (either via `sys.path` / run from repo root, or copy minimal code with a clear comment).
   - Implement `verify(lean_code: str, timeout: int = 300, ...) -> dict` that:
     - Calls `verify_lean4_file(code=lean_code, ...)` with appropriate `lake_path` and `lean_workspace` (configurable, default to `mathlib4/` next to Goedel-Prover).
     - Maps Goedel’s keys to `VerifyResult`:  
       - `success` ← `pass`  
       - `complete` ← `complete`  
       - `has_sorry` ← `len(sorries) > 0` or sorry in warnings  
       - `feedback` ← `"\n".join(e['data'] for e in errors)` (or equivalent)  
       - `errors` ← list of error strings  
       - `source` ← `"local_lean"` or `"goedel"`  
       - `is_server_error` ← False for local; True only on unexpected exception if desired.
     - On exception: return `verification_error_result(lean_code, str(e), ...)` from `lean_verification` so the contract matches existing code.
   - Document: “TLDR: Local Lean 4 verification using Goedel-Prover’s lake exe repl. Produces VerifyResult-shaped dict for sdpo_modal.”
   - Optional: support a small pool (e.g. `Lean4ServerScheduler` with 1 worker) for reuse, or keep single-process for MVP.

2. **Contract test**
   - In a small script or test: run `local_verifier.verify(known_good_lean)` and `verify(known_bad_lean)` and assert `success`/`complete` and `errors`/feedback shape.

3. **Lean environment**
   - Document in README or devlog: run from repo root (or set `LEAN_WORKSPACE` / `LAKE_PATH`); `mathlib4` must be built (`lake build` in `mathlib4/`). No Docker required for this path.

### Phase 2: Split SDPO loop + batching (all changes in the new folder only)

**Goal:** In the **new folder**, implement batched generate/step and local-driven loop; pass verification results into Modal in batches (one gradient update per N solutions).

1. **Config (new folder's `config.py`)**
   - Add **`batch_size`** (or `gradient_update_every_N`) to `SDPOConfig`, e.g. default 4.

2. **Batched SDPO loss (new folder's `sdpo_loss.py`)**
   - In the new folder's `sdpo_loss.py`: add **`compute_sdpo_loss_batch`** that takes a list of (base_prompt, teacher_prompt, generated_ids) and returns (mean_loss, list of per_item rewards/kl for logging), or simply have the trainer call `compute_sdpo_loss` N times and average the losses before backward. Important: **one** backward and **one** optimizer.step() per batch so the update is over N solutions.

3. **Modal trainer (new folder's `modal_trainer.py`)**
   - **`generate_batch(self, prompts: list[str]) -> list[tuple[str, list[int]]]`**  
     - Run N generations (one per prompt) on the same container; return list of `(raw_text, generated_ids)` with serializable ids (e.g. `.tolist()`). Single round-trip for N solutions.
   - **`run_sdpo_step(self, config_dict, problem, batch_payload: list[dict]) -> dict`**  
     - `batch_payload`: list of N iteration payloads; each payload has base_prompt, teacher_prompt, raw_output, generated_ids, full_code, verification, feedback_history, etc.
     - For each payload: skip if server_error or success (no gradient); for the rest, compute SDPO loss and accumulate (or compute in one batched forward if implemented). **Loss = mean over the N items** that are used for the update.
     - One `loss.backward()`, one `optimizer.step()`.
     - Return aggregated step log (e.g. mean loss, list of per-item iter_logs for logging).
   - Ensure `run_sdpo_step` does **not** call any verifier; it only uses the `verification` dict from each payload.

4. **Refactor `trainer_core` (optional)**
   - Extract “one iteration” logic into a pure function so the same parsing/feedback logic can be reused locally for each of the N solutions. Keep `run_sdpo` for the all-on-Modal path if needed.

5. **Local loop (batched)**
   - In `entrypoint_local_verify.py` (or equivalent):
     - Load dataset, config, problem; set N = config.batch_size.
     - Local verifier (and optionally N parallel workers for verify to reduce wall time).
     - Each **update step**:  
       (1) `prompts = [base_prompt] * N` (or N variants if you maintain per-sample feedback).  
       (2) `batch_generations = trainer.generate_batch.remote(prompts)`.  
       (3) For each of the N: parse → full_code → verification (local). Build N iteration payloads.  
       (4) `step_result = trainer.run_sdpo_step.remote(config_dict, problem, batch_payload)`.  
       (5) If any solution succeeded or max update steps, break.  
     - Aggregate logs, save/print as in current `run_main`.

6. **Config / CLI**
   - Add `--local-verify` and `--batch-size` (or use config); local-verify path uses batched generate + batched step.

### Phase 3: Modal app and entrypoint in the new folder (no Kimina)

**Goal:** In the **new folder**, modify the copied `modal_app.py` to define one Modal app (e.g. `lean-sdpo-local-compile`) with inference image only and trainer with `generate_batch` and `run_sdpo_step` (no Kimina). Original `sdpo_modal/` is never touched.

1. **Original folder**
   - Leave **original** `sdpo_modal/` **unchanged**. Kimina app and full `run_sdpo` remain as-is.

2. **New Modal app for local-compile**
   - Create a **separate** Modal app (e.g. in `sdpo_modal/modal_app_local_compile.py` or a second app in the same file) with a **distinct app name** (e.g. `lean-sdpo-local-compile`).
   - This app uses only the **inference image** (no Kimina image). It defines a single class: `SDPOTrainer` (or `SDPOTrainerLocalCompile`) with `generate_batch` and `run_sdpo_step` only — no `KiminaLeanServer`, no `LeanVerifier`. Same volumes/secrets as needed for model and output.
   - Entrypoint for local-compile (e.g. `entrypoint_local_verify.py`) imports and uses **this** app’s trainer, not the Kimina app’s.

3. **Documentation**
   - Document two paths: (A) Kimina-on-Modal (existing); (B) Local compilation (new app + local verifier). No removal or deprecation of Kimina.

### Phase 4: Cleanup and robustness

1. **Error handling**
   - Local verifier: timeouts, missing `mathlib4`, missing `lake` → return `VerifyResult` with `success=False`, `is_server_error=True` and clear `feedback`.
   - Modal `run_sdpo_step`: if payload is malformed or missing keys, return a clear error dict instead of crashing.

2. **Serialization**
   - `generated_ids`: ensure list of ints (or numpy/torch serializable) so Modal → local → Modal round-trip works.

3. **State on Modal**
   - `run_sdpo_step` is stateless per call (all state in `iteration_payload`) so no need to keep iteration state on Modal. Optionally, add a “warm” method that keeps the model loaded and reuses it for multiple `generate_only` / `run_sdpo_step` calls in the same run.

4. **Modularity**
   - Keep `trainer_core` free of Modal and free of “where verification runs”; it only receives `verify_fn` and `generate_fn`.
   - Keep verification result format as `VerifyResult` everywhere so swapping Kimina vs local later stays easy.

---

## 5. File-Level Summary

**Rule: Original `sdpo_modal/` is never modified. Copy the folder, then change only files in the new folder.**

| What | Action |
|------|--------|
| **New folder** | Copy entire `sdpo_modal/` to e.g. `sdpo_modal_local_verify/` (all files). Copy `local_lean_verifier.py` in. Update all internal imports to new package name. |
| **New folder: `local_lean_verifier.py`** | Implement local `verify(lean_code) -> VerifyResult` using Goedel’s `verify_lean4_file`; map result; handle exceptions. Optional: batch or pool for verifying N solutions in parallel. |
| **New: `sdpo_modal/entrypoint_local_verify.py`** (or equivalent) | Batched loop: Modal `generate_batch` (N) → local parse + verify all N → Modal `run_sdpo_step(batch_payload)` once per update; save/print results. |
| **`sdpo_modal/lean_verification.py`** | No change; keep `VerifyResult` and `verification_error_result`; optional: add `parse_goedel_result()` if you want to centralize mapping there. |
| **`sdpo_modal/modal_trainer.py`** | Add **`generate_batch(prompts)`**; add **`run_sdpo_step(config, problem, batch_payload)`** (list of N payloads → mean loss, one backward/step). Keep or remove `run_sdpo` for non–local-verify mode. |
| **`sdpo_modal/sdpo_loss.py`** | Add batched path: either **`compute_sdpo_loss_batch`** or trainer calls `compute_sdpo_loss` N times and averages; one backward/step per batch. |
| **`sdpo_modal/config.py`** | Add **`batch_size`** (or `gradient_update_every_N`), e.g. default 4. |
| **`sdpo_modal/trainer_core.py`** | Optionally extract one-iteration logic for reuse; keep `run_sdpo` as the full loop when both fns are given. |
| **`sdpo_modal/modal_app.py`** | **No change.** Keep Kimina image, `KiminaLeanServer`, `LeanVerifier`, and existing `SDPOTrainer` + `run_sdpo`. |
| **New: `sdpo_modal/modal_app_local_compile.py`** (or second app in same file) | New Modal app (e.g. `lean-sdpo-local-compile`): inference image only; `SDPOTrainer` with `generate_batch` + `run_sdpo_step`; no Kimina. Used by `entrypoint_local_verify`. |
| **`sdpo_modal/entrypoint.py`** | Either add `--local-verify` and `--batch-size` branch that uses the batched loop, or keep as-is and use a separate script for local verify. |
| **Goedel-Prover-main** | No structural change; only used as library (verifier + mathlib4). Document run-from-root or path setup. |

---

## 6. Answers to Your Questions

1. **Identify components using Kimina lean server; add separate path with local verification**  
   - **Kimina path (unchanged):** `modal_app.KiminaLeanServer`, `modal_app.LeanVerifier`, and `modal_trainer` with `verify_fn = LeanVerifier().verify.remote(...)` stay as-is for the existing app.  
   - **Local-compile path (new):** A **copy** of the pipeline in a new folder (e.g. `sdpo_modal_local_verify/`). That copy has its own Modal app (no Kimina), its own `local_lean_verifier.py`, and entrypoint; original `sdpo_modal/` is never changed. The local-compile loop runs locally and uses the local verifier; it talks only to the new app’s `generate_batch` and `run_sdpo_step`.

2. **Adapt verification code (copy, minimal change) to sdpo_pipeline**  
   - **Adapter only:** Do not change Goedel’s verifier logic. Add a thin adapter that: (a) calls `verify_lean4_file(code=lean_code, ...)` with correct workspace/path, (b) maps the returned dict to `VerifyResult` (success, complete, has_sorry, feedback, errors, source), (c) on exception returns `verification_error_result(...)`. Optionally copy only `verify_lean4_file` (and its direct deps) into the repo with attribution if you want to avoid running from Goedel root.

3. **Pipeline: Modal generate → local compile → send compilation results back to Modal**  
   - **Yes, possible.** Implement it as: local driver loop; each iteration: (1) Modal `generate_only` → local gets `(raw_text, generated_ids)`; (2) local parses and builds `full_code`, then runs local verification; (3) local calls Modal `run_sdpo_step` with the verification result and other iteration data; Modal performs the training step and returns logs. So “compile” is done locally and “compilation results” are sent to Modal in the `run_sdpo_step` payload.

---

## 7. Execution Order (Recommended)

1. **Phase 0:** Copy `sdpo_modal/` to `sdpo_modal_local_verify/` (or chosen name). Copy `local_lean_verifier.py` into the new folder. Update all internal imports in the new folder to the new package name.  
2. **Phase 1:** Ensure local verifier in the new folder imports from the new package's `lean_verification`. (Already implemented in `sdpo_modal/`; just fix imports in the copy.)  
3. **Phase 2:** In the **new folder only**: add `batch_size` to config; batched loss; `generate_batch` and `run_sdpo_step` in the trainer; entrypoint implements batched local loop.  
4. **Phase 3:** In the **new folder only**: modify `modal_app.py` to remove Kimina and expose only the trainer with `generate_batch` and `run_sdpo_step`.  
5. **Phase 4:** Error handling, serialization, docs. Add a runner script that imports from the new folder and runs its entrypoint.

**Important:** The **original** `sdpo_modal/` is **never** modified. All changes are confined to the new folder. Kimina Lean Server image and the existing app stay intact.
