# Modal Lean Verification Pipeline: Goedel and Kimina

**Date:** 2026-03-03  
**Purpose:** Single reference for a functioning verification pipeline on Modal—either with **Kimina Lean Server** (HTTP) or **Goedel-style** (subprocess REPL). Covers the verifier contract, pass/complete semantics, Kimina API, and how to map results so step2/step3 work unchanged.

---

## 1. Verifier contract (pipeline requirement)

**Input:** One Lean 4 source string (full file). Optional: `timeout`.

**Output dict (required for step2/step3 and compatibility):**

- `pass` (bool) — no Lean errors
- `complete` (bool) — pass and no sorries and no sorry/failed warnings
- `verify_time` (float)

**Optional for debugging:** `errors`, `warnings`, `sorries`, `system_errors`.

**On exception or server/REPL failure:** Return at least `pass=False`, `complete=False`, `verify_time` (e.g. 0), and `errors` or `system_errors` so callers do not crash on missing keys.

**Pipeline usage:** Step2 attaches this dict to each item as `compilation_result`; step3 uses `compilation_result["complete"]` or `compilation_result["pass"]` and groups by problem. Any verifier (local or Modal) must return this shape.

---

## 2. Pass / complete semantics (same for both backends)

Both Kimina and the Goedel REPL expose the same Lean REPL output: `messages` (list of `{severity, data}`) and `sorries`. Use the same logic everywhere:

```text
errors   = [m for m in messages if m.get("severity") == "error"]
warnings = [m for m in messages if m.get("severity") == "warning"]
pass     = len(errors) == 0
complete = pass and len(sorries) == 0 and not any(
    "declaration uses 'sorry'" in (w.get("data") or "") or "failed" in (w.get("data") or "")
    for w in warnings
)
```

Optionally treat messages whose `data` contains “unsolved goals” as errors. **verify_time:** from REPL/server response when available, else wall clock.

---

## 3. Backend options

| | Goedel-style (REPL) | Kimina Lean Server |
|---|---------------------|---------------------|
| **Where** | Subprocess `lake exe repl` in mathlib4 workspace | HTTP `POST /verify` to Kimina server (e.g. in Modal via `projectnumina/kimina-lean-server:2.0.0`) |
| **Input** | Stdin: one JSON line `{"cmd": "<lean_code>"}` then `"\r\n\r\n"`. | Body: `{"codes": [{"custom_id": "<id>", "proof": "<lean_code>"}], "infotree_type": "original", "timeout": 300}`. |
| **Output** | Stdout: one JSON with `messages`, `sorries`. | Response: `results` list; per item: `custom_id`, optional `error`, optional `response` (has `messages`, `sorries`, `time`). |
| **Result** | Parse stdout → errors/warnings/sorries → pass/complete/verify_time. | Parse `results[i].response` (or handle `results[i].error`) → same. |

Kimina uses the same Lean REPL under the hood, so **exact** Goedel semantics are possible if the parser uses `response.messages` and `response.sorries` and the complete formula above. The Kimina API does **not** include a `status` field on each result; use `error` and `response` only.

---

## 4. Kimina Lean Server API (condensed)

**Request:** `POST /verify`

- **Body:** `codes` (list of `{custom_id, proof}` or `{custom_id, code}`), optional `timeout` (default 300), optional `infotree_type` (e.g. `"original"`), optional `disable_cache`.
- **Single snippet:** `codes: [{"custom_id": "1", "proof": "<lean_code>"}]`.

**Response:** `{ "results": [ ... ] }` or top-level `error` on request failure.

- **Per result:** `custom_id`; optional `error` (string) — if set, treat as failure (timeout/server/REPL); optional `response` — when present, extended REPL output:
  - `messages`: list of `{severity, data, pos?, endPos?}` (severity = "error" | "warning" | "info" | "trace").
  - `sorries`: list of `{pos, endPos, goal, ...}`.
  - `time`: float (use as verify_time).
- If REPL crashes, `response` may be `{ "message": "<error string>", "time": ... }` — treat as failure.

---

## 5. Kimina → Goedel-shaped result (parser steps)

Use this in `sdpo_modal/lean_verification.py` (or equivalent) so the pipeline gets the contract in §1.

1. If top-level `error`: return `pass=False`, `complete=False`, `verify_time=wall_s`, `errors=[error]`.
2. If no `results` or empty: same with generic error.
3. `r = results[0]` (or match by `custom_id`). If `r.get("error")`: failure; set pass=False, complete=False, verify_time from `r.get("response", {}).get("time")` if present.
4. If `r.get("response")` is None: failure (same as above).
5. If `response` has key `"message"` (REPL error): pass=False, complete=False.
6. Else: `messages = response.get("messages") or []`, `sorries = response.get("sorries") or []`. Compute `errors`, `warnings`, `pass`, `complete` as in §2. `verify_time = response.get("time")` or wall time.
7. Return dict with at least `pass`, `complete`, `verify_time`; preferably also `errors`, `warnings`, `sorries`.

**Fixes for current parser:** (1) Treat result-level `error` as failure. (2) Treat `response` with only `"message"` as REPL error. (3) Build `warnings` and use full complete condition (no “declaration uses 'sorry'” / “failed” in warning data). (4) Do not use `r.get("status")`; API has no such field. (5) Expose `verify_time` and `pass` for step2/step3.

---

## 6. Goedel-style verification on Modal (optional)

If you run the REPL on Modal instead of Kimina:

- **Image:** Elan + mathlib4 (pinned commit), `lake build`, `lake exe repl` in workspace. No Kimina container.
- **Function:** Write `{"cmd": code}` + `"\r\n\r\n"` to stdin; run `subprocess.run([lake_path, "exe", "repl"], stdin=..., capture_output=True, text=True, cwd=mathlib4_root, timeout=timeout)`; parse stdout JSON; compute pass/complete/verify_time as in §2; return the same dict as §1.
- Reuse `prover/lean/verifier.py` logic or `prover/lean/ast_parser.py` if you need AST (optional).

---

## 7. Checklist for a functioning pipeline

- [ ] Verifier (Kimina or REPL) returns a dict with **pass**, **complete**, **verify_time** on every path (success and failure).
- [ ] Kimina parser: handle result **error** and **response.message**; derive **warnings** from messages; implement full **complete** condition; expose **verify_time** and **pass**.
- [ ] On exception in Modal verifier: return `pass=False`, `complete=False`, `verify_time`, and `errors` or `system_errors`.
- [ ] Step2: set `codes[i]["compilation_result"] = <verifier output dict>`; if using a remote verifier, replace `Lean4ServerScheduler` with a client that expects this dict.
- [ ] Step3: no change; it reads `compilation_result["complete"]` or `["pass"]`.

**References:** Goedel verifier: `Goedel-Prover-main/prover/lean/verifier.py`. Kimina: https://github.com/project-numina/kimina-lean-server (`client/kimina_client/models.py`, `server/routers/backward.py`).
