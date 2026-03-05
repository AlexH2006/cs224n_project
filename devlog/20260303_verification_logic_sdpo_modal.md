# Verification logic in SDPO Modal pipeline

**Date:** 2026-03-03  
**Topics:** verification, kimina, sdpo, modal, logic, success, complete, sorry

---

This document describes the **verification logic** used in the SDPO Modal scripts (e.g. `lean_sdpo_qwen_3b_modal.py`, `lean_sdpo_kimina_2b_modal.py`): what gets verified, how the Kimina server is called, how results are interpreted, and how the training loop uses them. For timeout and container layout, see [20260302_lean_verification_system_deep_dive.md](20260302_lean_verification_system_deep_dive.md).

---

## 1. What gets verified

Verification receives a **single string**: the full Lean 4 file to typecheck. That string is built as follows.

### 1.1 Building the full Lean code

1. **Theorem code**  
   From the dataset: `lean4_code` / `lean4_statement` / `statement` / etc. (configurable `theorem_fields`). Contains imports (optional), the formal statement, and a proof placeholder: `:= sorry` or `:= by sorry` or a bare `sorry`.

2. **Proof tactics**  
   Extracted from the model’s raw output by `_extract_proof_tactics`: from `\`\`\`lean4` blocks, or after `</think>`, or from a `:= by` line. If extraction fails or output is truncated (e.g. `<think>` without `</think>`), tactics are set to `"sorry"`.

3. **Header**  
   Either from the dataset (`header` / `imports` / etc.) or from model-generated `import`/`open`/`set_option` lines in the tactics, or the config’s `default_header`.

4. **Assembly**  
   `_create_full_lean_code(config, theorem_code, tactics, header)`:
   - Replaces the theorem’s proof placeholder with the extracted tactics (with proper `by` and indentation).
   - Prepends the chosen header.
   - Returns `header + "\n\n" + theorem_with_proof`.

So the **input to verification** is always one contiguous Lean 4 file (header + statement + proof).

---

## 2. Pre-verification guards

Two checks run **before** calling the verifier.

### 2.1 Commented-out formal statement

If **every** non-empty, non-import line in the theorem code is a comment (e.g. minif2f problems tagged `-- Error: Real.log`), then there is no real theorem to prove. Verifying such a file would only typecheck imports and comments and could succeed trivially, giving a wrong reward.

- **Check:** `_theorem_code_is_commented_out(lean4_code)`.
- **If true:** We do **not** call the verifier. We set a synthetic `verification` with `success=False`, `complete=False`, `source="skipped"`, and break the iteration loop.

### 2.2 No need to verify yet

We always build `full_code` and verify (except for the commented-out case above). The “extracted tactics is just `sorry`” case is handled **after** verification (see below).

---

## 3. Kimina Lean Server (Layer 3)

- **Image:** `projectnumina/kimina-lean-server:2.0.0` (pre-built Lean + Mathlib).
- **Process:** On container entry, `_start_lean_server()` starts a subprocess: `python -m server`, listening on `http://0.0.0.0:8000`.
- **Health check:** A minimal proof `example : True := trivial` is POSTed to `/verify`; we wait up to 60s for 200.
- **Liveness:** Before each verify, `_ensure_server_alive()` checks the process and health endpoint; if dead or unresponsive, the server is restarted.

### 3.1 Verify request

- **Endpoint:** `POST http://localhost:8000/verify` (inside the same container).
- **Body:**
  ```json
  {
    "codes": [{"custom_id": "<id>", "proof": "<full_lean_code_string>"}],
    "infotree_type": "original"
  }
  ```
- **HTTP timeout:** 60 seconds (fail fast for normal use).
- **Retries:** Up to 3 attempts. On `ConnectError` or `TimeoutException` or other exception, we restart the Lean server and retry; if all fail, we return `{"error": "...", "is_server_error": true}`.

### 3.2 Verify response (successful HTTP)

Kimina returns a JSON object. We care about:

- **`results`**  
  List of per-code results (we send one code, so `results[0]`).
- **`results[0].status`**  
  e.g. `"error"` when something went wrong.
- **`results[0].messages`**  
  List of Lean messages (each with `severity`, `data`, etc.). We collect **errors** from `severity == "error"` and also treat any message whose text contains `"unsolved goals"` (case-insensitive) as an error.
- **`results[0].sorries`**  
  List of reported `sorry` occurrences.

If the HTTP call fails (after retries), we return `{"error": "...", "is_server_error": true}`.

---

## 4. LeanVerifier (Layer 2): normalizing the result

`LeanVerifier` runs in a **separate** Modal CPU container. It calls `KiminaLeanServer().verify.remote(lean_code)` and then turns the raw Kimina response into a **normalized verification result** used by the SDPO loop.

### 4.1 Normalized result shape

Every verification attempt (whether it hit Kimina or not) returns a dict with at least:

| Field | Type | Meaning |
|-------|------|--------|
| `success` | bool | No Lean errors (and no “unsolved goals” treated as error). |
| `complete` | bool | `success` and no `sorry` left (no placeholder). |
| `has_sorry` | bool | Either Kimina reported sorries or the string `"sorry"` appears in the code. |
| `feedback` | str | Error text for the model (concatenated errors, or server/parse error message). |
| `errors` | list[str] | List of error strings. |
| `source` | str | `"kimina"` or `"skipped"` or from exception. |
| `is_server_error` | bool | True if the failure was due to server/HTTP/timeout, not a real proof failure. |
| `debug` | dict | Optional; e.g. `verifier_wall_s`. Removed from the dict before use in the loop so it isn’t stored in logs. |

### 4.2 Parsing Kimina response

- **Kimina returned an error (HTTP or after retries):**  
  We set `success=False`, `complete=False`, `has_sorry` from presence of `"sorry"` in the code, `feedback` to the server error message, `is_server_error=True`.

- **Kimina returned 200 with `results` and at least one result:**  
  - **errors:** All `messages` with `severity == "error"` plus any message whose `data` contains `"unsolved goals"`.  
  - **has_error:** `len(errors) > 0 or status == "error"`.  
  - **has_sorry:** Kimina’s `sorries` or `"sorry"` in the code.  
  - **success:** `not has_error`.  
  - **complete:** `not has_error and not has_sorry`.  
  - **feedback:** `"\n".join(errors)` (used as training signal).  
  - **is_server_error:** False.

- **Kimina returned 200 but unexpected shape (e.g. no `results`):**  
  We set `success=False`, `complete=False`, `is_server_error=True`, `feedback="Unexpected response format from Kimina server"`.

- **Any exception in LeanVerifier:**  
  We set `success=False`, `complete=False`, `feedback` to the exception message, `is_server_error=True`.

So: **success** = “Lean didn’t report errors”; **complete** = “success and no sorry”; **is_server_error** = “we couldn’t get a reliable proof result”.

---

## 5. SDPO loop: retries and use of verification

### 5.1 Calling the verifier and retries

- The trainer runs in a **GPU** container and calls `LeanVerifier().verify.remote(full_code)`.
- **Retries:** Up to 3 attempts. If `verification.get("is_server_error")` is true, we retry (with a 5s sleep between attempts). We stop as soon as we get a result with `is_server_error=False`, or after 3 attempts.

### 5.2 Success and overrides

- **Base:** `is_success = verification["success"] and verification["complete"]`.
- **Override — extracted tactics are just `"sorry"`:**  
  If `tactics.strip().lower() == "sorry"`, we **force** `is_success = False`, `verification["complete"] = False`, `verification["has_sorry"] = True`, and set `verification["feedback"]` to either:
  - “Reasoning was cut off…” (if output was truncated), or  
  - “No valid proof tactics found. Output actual Lean 4 tactics in a \`\`\`lean4 code block.”  
  So even if the Lean file contained `sorry` and Kimina reported no errors (e.g. only that one placeholder), we still treat it as failure and give the model a clear instruction.

### 5.3 What the loop does with the result

- **If `is_success`:**  
  Store `best_proof`, append iteration log, record metrics, **break** (problem solved).

- **If `is_server_error`:**  
  Append iteration log (no loss/reward), **do not** train on this attempt, **continue** to next iteration (retry generation/verification). This avoids training on unverified or bogus feedback.

- **Otherwise (proof failed but not server error):**  
  Append `(feedback, tactics)` to `feedback_history`, build the teacher prompt with that history, compute SDPO loss, step optimizer, then continue to the next iteration.

### 5.4 Debug logging

On the Modal volume, under `output_dir/debug/lean_server/`, we append a JSONL file `verify_<session_id>.jsonl` with:

- **verify_start:** `event`, `ts`, `iteration`, `attempt`, `code_len`.
- **verify_end:** `event`, `ts`, `iteration`, `attempt`, `duration_s`, `success`, `complete`, `is_server_error`, `error_snippet` (first 500 chars of feedback).

This allows offline inspection of verification timing and outcomes without changing the verification logic.

---

## 6. Summary

- **Input to verification:** One full Lean 4 file (header + theorem with proof placeholder replaced by extracted tactics).
- **Guards:** Skip verification entirely when the theorem code is fully commented out.
- **Kimina:** One HTTP POST per verify; we parse `results[0].messages` (errors + “unsolved goals”) and `sorries` to get success/complete/has_sorry and a single `feedback` string.
- **LeanVerifier:** Normalizes all outcomes (server errors, timeouts, parse errors) into one dict with `success`, `complete`, `has_sorry`, `feedback`, `is_server_error`.
- **SDPO loop:** Retries up to 3 times on server error; overrides success to false when extracted tactics are `"sorry"`; only trains when verification actually ran and failed (no training on server error); uses `feedback` for the teacher prompt and reward signal.
