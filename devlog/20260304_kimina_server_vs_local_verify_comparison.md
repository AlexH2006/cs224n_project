# Kimina Lean Server vs Current Local-Verify Approach

**TLDR:** Both use the same Lean 4 REPL and JSON protocol. Kimina runs as an **HTTP server with a pool of long-lived REPL processes** (typically in Linux/Docker/Modal). The current approach runs **one subprocess per verification** on the driver machine (your Mac). Same verification semantics; different process model, location, and scalability.

---

## 1. Kimina Lean Server (`kimina-lean-server` codebase)

### What it is

- **FastAPI HTTP server** that exposes Lean 4 verification over the network.
- Runs in **Linux** (Docker image; e.g. `projectnumina/kimina-lean-server:2.0.0` or built from `kimina-lean-server/Dockerfile`).
- Uses a **pool of long-lived REPL processes** managed by a `Manager`.

### How verification works

1. **REPL process**  
   Each REPL is started once and kept alive:
   - Command: `lake env <repl_path>` (or `lake exe repl` in project dir), `cwd=project_dir` (mathlib4).
   - Same JSON protocol as Goedel: send `{"cmd": "<code>"}\n\n` on stdin, read one JSON object from stdout.

2. **Header/body split**  
   For each snippet, `split_snippet(code)` splits into:
   - **Header**: import lines (and blank lines before first non-import).
   - **Body**: rest of the code (theorem + tactics).

3. **Reuse**  
   - Manager assigns a REPL (by header, so same imports → same REPL when possible).
   - **First use of a REPL:** run header as first command (`is_header=True`), then run body.
   - **Later uses:** run only the body (with `env` / `gc` for state), so one REPL serves many snippets with the same header.

4. **API**
   - **POST /api/check**  
     Body: `{ "snippets": [ { "id": "...", "code": "..." } ], "timeout", "debug", "reuse", "infotree" }`.  
     Returns list of per-snippet responses (response, error, time, diagnostics).
   - **POST /verify** (backward-compat)  
     Body: `{ "codes": [ { "custom_id": "...", "proof": "..." } ], "timeout", "infotree_type", "disable_cache" }`.  
     Internally converts to snippets and calls the same `run_checks` as `/api/check`, then returns `VerifyResponse(results=...)`.

5. **Response shape**  
   Each result has `response` (messages, sorries, env, time, etc.) or `error`.  
   This is what `sdpo_modal`’s `parse_kimina_response()` expects when the Modal app calls `POST /verify`.

### Where it runs in your setup

- In **sdpo_modal** (Kimina pipeline): a Modal class runs the Kimina server **inside a Modal container** (Linux) as a subprocess (`python -m server`), then `LeanVerifier.verify.remote(lean_code)` POSTs to `http://localhost:8000/verify` inside that container.
- So verification runs **on Modal (Linux)**, not on your Mac → no macOS dyld issue.

---

## 2. Current approach (`sdpo_modal_local_verify` + `local_lean_verifier`)

### What it is

- **No HTTP server.**  
  Verification is a **direct Python call** `verify(lean_code)` on the **driver machine** (the machine that runs the Modal entrypoint and the SDPO loop).

### How verification works

1. **One subprocess per call**  
   Each `verify(lean_code)`:
   - Runs `subprocess.run([lake_path, "exe", "repl"], stdin=..., cwd=lean_workspace, timeout=...)`.
   - Writes **one** JSON message (the full `lean_code` as `cmd`) to stdin.
   - Reads **one** JSON response from stdout.
   - Process exits; no REPL reuse.

2. **No header/body split**  
   The **entire** Lean file (header + theorem + tactics) is sent in a single `cmd`.  
   No separate “run header once, then many bodies” optimization.

3. **Same REPL, same protocol**  
   Under the hood it’s the same REPL binary (from the repl package in mathlib4) and the same request/response format (messages, sorries, errors) as Kimina.  
   So verification **semantics** (pass/fail, sorry, compiler errors) are the same.

4. **Result**  
   `local_lean_verifier` maps REPL JSON to the same **VerifyResult** contract (success, complete, feedback, errors, etc.) used by the rest of the pipeline.

### Where it runs

- On the **driver** (e.g. your Mac).  
- So when you run the local-verify pipeline, the REPL binary runs **on macOS** → you hit the **dyld __DATA_CONST** issue there.

---

## 3. Similarities

| Aspect | Kimina server | Local verifier |
|--------|----------------|----------------|
| **Core engine** | Lean 4 REPL (repl package) | Same |
| **Protocol** | JSON over stdin/stdout (cmd → response) | Same |
| **Workspace** | mathlib4 (`lake build`) | Same (e.g. Goedel-Prover-main/mathlib4) |
| **Output shape** | messages, sorries, errors | Same; both mapped to VerifyResult |
| **Semantics** | pass/fail, sorry, compiler errors | Same |

So for a **given** Lean snippet and mathlib4 build, the **logical** result (valid / sorry / error and the error messages) is the same; only where and how the process runs differ.

---

## 4. Differences

| Aspect | Kimina Lean Server | Current local approach |
|--------|--------------------|------------------------|
| **Execution location** | Server (Linux container, e.g. Modal) | Driver machine (e.g. Mac) |
| **Process model** | Pool of **long-lived** REPLs; reuse by header | **One-shot** subprocess per `verify()` |
| **Header/body** | Splits code; runs header once per REPL, then many bodies | Sends full code in one shot; no split |
| **API** | HTTP: POST /verify, POST /api/check | Python: `verify(lean_code)` |
| **Concurrency** | Many requests; pool size and reuse | Sequential per driver (one process per call) |
| **Resource use** | More memory (several REPLs), less per-request startup | Low memory, high per-request startup (new process each time) |
| **macOS** | Runs in Linux on Modal → **no dyld issue** | REPL runs on Mac → **dyld __DATA_CONST** failure |

---

## 5. When to use which

- **Kimina Lean Server**  
  - When you want verification **on Linux** (Modal, Docker, VM) and/or **scalable** verification (many concurrent checks, REPL reuse).  
  - This is what the original **sdpo_modal** (Kimina) pipeline uses: verification runs in the Modal container.

- **Local verifier**  
  - When you want **no server**, **no Docker** on the driver, and are fine running verification **on the driver** (e.g. for dev or single-machine runs).  
  - On **Linux** drivers it works as-is; on **macOS** you need to avoid running the REPL there (e.g. run the pipeline on Linux, or use Docker/VM for the verification step).

---

## 6. References

- Kimina server: `kimina-lean-server/` (Dockerfile, `server/main.py`, `server/repl.py`, `server/manager.py`, `server/routers/check.py`, `server/routers/backward.py`, `server/split.py`).
- Modal Kimina usage: `sdpo_modal/modal_app.py` (KiminaLeanServer, LeanVerifier), `sdpo_modal/lean_verification.py` (parse_kimina_response).
- Local verifier: `sdpo_modal_local_verify/local_lean_verifier.py`, `sdpo_modal_local_verify/entrypoint.py`.
