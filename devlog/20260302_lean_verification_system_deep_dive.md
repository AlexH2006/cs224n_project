# Lean verification system and timeout mechanism — detailed analysis

**Date:** 2026-03-02  
**Topics:** verification, kimina, modal, timeouts, containers, debugging, sdpo, comparison, sandbox_fusion

---

This document explains how Lean verification works in our SDPO pipeline and how every timeout and lifecycle setting behaves. It is written for readers who are new to containers (Docker) and distributed systems (Modal).

---

## 1. High-level: what runs where

When you run `modal run training/lean_sdpo_kimina_2b_modal.py ...`, Modal starts a **distributed** job. Different pieces of code run in different **containers** (isolated environments, like lightweight virtual machines).

### 1.1 Containers in plain language

- A **container** is a run environment: fixed OS, CPU/memory, and your code. It is started from an **image** (a snapshot of files and dependencies). Think of the image as a template and the container as a running instance.
- Modal **spins up containers on demand**. When you call `SomeClass().method.remote()`, Modal may start a new container for `SomeClass`, run `method` there, and return the result to the caller. The caller might be your laptop (local entrypoint) or another container.
- Containers are **ephemeral**: they can be shut down when idle (see *scaledown* below) or when a run hits a *timeout* (see below). When a container is gone, any process running inside it (e.g. the Lean server) is gone too.

### 1.2 The three “layers” of verification

Verification is done by a chain of three layers:

```
  [1] SDPOTrainer (GPU container)     →  runs training, generates proof text
           │
           │  LeanVerifier().verify.remote(full_code)
           ▼
  [2] LeanVerifier (CPU container)   →  parses result, talks to Kimina
           │
           │  KiminaLeanServer().verify.remote(lean_code)
           ▼
  [3] KiminaLeanServer (CPU container)  →  runs Lean server process, HTTP /verify
           │
           │  HTTP POST localhost:8000/verify (inside this container)
           ▼
       Lean server subprocess (python -m server)  →  typechecks the proof
```

- **Layer 1 — SDPOTrainer**  
  Runs on a GPU container. It has the model, generates proofs, and for each attempt calls `LeanVerifier().verify.remote(full_code)`. It **blocks** until that call returns (or times out).

- **Layer 2 — LeanVerifier**  
  Runs in its **own** CPU container. When `verify.remote(full_code)` is invoked, Modal runs `LeanVerifier.verify(lean_code)` in that container. Inside that method we do `server = KiminaLeanServer(); result = server.verify.remote(lean_code)`. So the LeanVerifier container **blocks** until the KiminaLeanServer call returns.

- **Layer 3 — KiminaLeanServer**  
  Runs in **another** CPU container. That container’s startup (`@modal.enter()`) starts the actual Lean server as a **subprocess** (`python -m server`). The `verify` method then sends an HTTP POST to `http://localhost:8000/verify` (inside the same container). The Lean server process typechecks the code and replies over HTTP.

So: **two separate containers** are involved in verification (LeanVerifier and KiminaLeanServer). The “Lean engine” is the subprocess inside the **KiminaLeanServer** container.

---

## 2. Timeout mechanism — in detail

There are three different “timeout” concepts. It’s important not to confuse them.

### 2.1 Modal function timeout (`timeout=...`)

- **What it is:** A limit on how long **one execution** of a Modal function (e.g. `verify`) is allowed to run.
- **Where we set it:**
  - `KiminaLeanServer`: `timeout=1200` (20 minutes)
  - `LeanVerifier`: `timeout=900` (15 minutes)
  - `SDPOTrainer`: `timeout=3600` (1 hour)
- **What happens when it fires:** Modal **terminates that function run**. For a class method, that means the **container** running that method is stopped. So:
  - If `KiminaLeanServer.verify` runs for 1200 seconds, Modal kills the **KiminaLeanServer container**. The Lean server **subprocess** runs inside that container, so it is killed too. **The server goes down.**
  - If `LeanVerifier.verify` runs for 900 seconds, Modal kills the **LeanVerifier container**. The KiminaLeanServer container (and its Lean process) might still be running if it’s a different container, but the LeanVerifier process is gone.
- **Important:** This timeout is **per execution**, not “per container lifetime”. So a single verification that runs too long can cause the whole KiminaLeanServer container (and thus the Lean engine) to be killed. That is the “server goes down after timeout” behavior.

### 2.2 HTTP client timeout (httpx)

- **What it is:** Inside `KiminaLeanServer.verify` we use `httpx.Client(timeout=600.0)` and then `client.post(...)`. That limits how long we **wait for the HTTP response** from the Lean server (localhost:8000).
- **What happens when it fires:** The **client** (our Python code) gives up waiting. We get `httpx.TimeoutException`. The **server** (Lean process) is not necessarily killed; it may still be typechecking. So:
  - The request is abandoned from the client’s point of view.
  - We retry (currently up to 10 times with 3 s delay). So in the worst case we can wait 10 × (600 s + 3 s) ≈ 6030 s in one `KiminaLeanServer.verify` call. That can exceed the Modal timeout (1200 s), and then **Modal kills the container** → server goes down.
- **Design goal:** User requirement is that verification should take **no longer than a minute**. So the HTTP timeout should be **60 s**. That way we never hold the request (or the container) for 10 minutes, and we avoid triggering Modal’s function timeout under normal use.

### 2.3 Scaledown window (`scaledown_window=...`)

- **What it is:** How long a container is allowed to sit **idle** (no new requests) before Modal may **scale it down** (destroy it).
- **Where we set it:**
  - `KiminaLeanServer`: `scaledown_window=300` (5 minutes)
  - `LeanVerifier`: `scaledown_window=300`
  - `SDPOTrainer`: `scaledown_window=600`
- **What happens when it fires:** After 300 s (or 600 s) with no new work, Modal can shut down that container. The next time we call `KiminaLeanServer().verify.remote(...)`, Modal may start a **new** container. That new container runs `@modal.enter()` again, which starts the Lean server subprocess — so we pay **cold start** (~18 s) again.
- **Important:** This is **not** “server goes down because of a timeout on a request.” It’s “server (container) goes away because we didn’t send another request for a while.” So if between two verifications we have a long gap (e.g. long generation + long training), the Kimina container might be scaled down and the next verification gets a cold start.

---

## 3. Summary: when does the “server” go down?

| Cause | What actually happens |
|-------|------------------------|
| **Modal function timeout (1200 s) on KiminaLeanServer** | Modal kills the KiminaLeanServer **container**. The Lean server subprocess is inside that container, so it is killed. **This is the main “server goes down after timeout” behavior.** |
| **Modal function timeout (900 s) on LeanVerifier** | LeanVerifier container is killed. KiminaLeanServer container (and Lean process) may still be running. |
| **HTTP timeout (e.g. 600 s)** | Client stops waiting; server might still be running. But we then retry; if we retry many times, the **total** time can exceed Modal’s 1200 s and then Modal kills the Kimina container. |
| **Scaledown (300 s idle)** | No request for 300 s → Modal may destroy the Kimina container. Next request gets a new container (cold start). The “server” didn’t crash; it was shut down due to idleness. |

So the **unwanted** behavior (“server goes down after timeout”) is: a **single verification** running so long that Modal’s **function timeout** (1200 s for KiminaLeanServer) is hit, and Modal then **kills the container** (and thus the Lean engine).

---

## 4. Recommended behavior (aligned with “verification ≤ 1 minute”)

1. **HTTP timeout:** Set to **60 s** in `KiminaLeanServer.verify`. No single request should wait longer than a minute.
2. **Retries:** Use a small number of retries (e.g. 2–3) so that a few timeouts don’t add up to more than a couple of minutes. This avoids approaching the 1200 s Modal timeout and prevents the container (and server) from being killed.
3. **Modal timeouts:** Keep 900 s / 1200 s as a safety net, but in practice we should never hit them if each verification finishes (or times out at 60 s) quickly.
4. **Scaledown:** Optionally increase `scaledown_window` for KiminaLeanServer (e.g. to 600 s) so that short gaps between iterations don’t cause scale-down and cold starts.

---

## 5. Code locations (for reference)

| Setting | File | Approx. line | Symbol |
|--------|------|--------------|--------|
| KiminaLeanServer timeout | `lean_sdpo_kimina_2b_modal.py` | ~183 | `@app.cls(timeout=1200, ...)` |
| KiminaLeanServer scaledown | same | ~184 | `scaledown_window=300` |
| KiminaLeanServer HTTP timeout | same | ~248 | `httpx.Client(timeout=600.0)` |
| KiminaLeanServer retries | same | ~244, 246 | `max_retries = 10`, `retry_delay = 3` |
| LeanVerifier timeout | same | ~285 | `@app.cls(timeout=900, ...)` |
| LeanVerifier scaledown | same | ~286 | `scaledown_window=300` |
| Verification call site | same | ~646 | `LeanVerifier().verify.remote(full_code)` |

---

## 6. Debugging: Lean server monitoring

To observe what happens during verification (without guessing), we:

- Write **debug logs** into a dedicated folder: `debug/lean_server/` (under the run’s output area on the Modal volume and, when synced, locally).
- Each verification is logged with: start/end time, duration, iteration, attempt, success, server_error, and optional error snippet.

See the code changes that add this monitoring and the timeout/retry fixes described above.

---

## 7. Confirmation: server goes down after timeout (and fixes applied)

**Investigation result:** Yes. When the **Modal function timeout** (1200 s for `KiminaLeanServer.verify`) is hit, Modal **terminates that function’s execution**. The KiminaLeanServer container is running that function, so the container is stopped. The Lean server is a **subprocess** inside that container, so it is killed too. So the Lean server does go down after a timeout in the sense of “the container (and everything in it) is killed by Modal.”

**Why this is bad:** The next verification then runs in a **new** container (cold start ~18 s). Repeated timeouts can cause repeated cold starts and long waits.

**Fixes applied in code:**

1. **HTTP timeout reduced to 60 s** in `KiminaLeanServer.verify` (was 600 s). Verification is expected to complete in under a minute; if it doesn’t, we fail fast and avoid holding the request (and risking the 1200 s Modal timeout).
2. **Retries reduced to 3** (was 10) so we don’t pile up 10 × 60 s on timeouts.
3. **`scaledown_window` increased to 600 s** for KiminaLeanServer (was 300 s) so the container is less likely to be scaled down between iterations; this reduces cold starts from idleness rather than from timeout.
4. **Lean server debug logs** are written to `debug/lean_server/` (see below) so we can see verify_start/verify_end and duration for every verification.

---

## 8. Comparison: SDPO folder (verl) vs our approach — Lean verification and compilation

The **SDPO folder** in this repo (the verl-based codebase under `SDPO/`) was read to see how they use a “Lean” or verification server to compile/verify results. Below is how they do it and how it differs from our Kimina-based pipeline.

### 8.1 How the SDPO folder handles “Lean” and code verification

- **They do not use Kimina Lean Server.** They use **Sandbox Fusion** (Bytedance): a generic multi-language code execution API ([SandboxFusion](https://github.com/bytedance/SandboxFusion)).
- **Where it lives:**  
  - `SDPO/verl/utils/reward_score/sandbox_fusion/` — `utils.py` (API calls, `check_correctness`), `__init__.py` (`compute_score`).  
  - Reward wiring: `verl/trainer/ppo/reward.py` (builds `default_compute_score` with `sandbox_fusion_url` when configured), `verl/utils/reward_score/__init__.py` (data_source-based dispatch; sandbox_fusion used for codecontests, apps, codeforces, taco).
- **API contract:**
  - Single HTTP POST to a URL (e.g. `https://<endpoint>/run_code`).
  - Payload: `code`, `stdin`, `compile_timeout`, `run_timeout`, `memory_limit_MB`, **`language`** (e.g. `"python"`, `"lean"`), `files`, `fetch_files`.
  - Response: top-level status (`Success` / `Failed` / `SandboxError`), plus **`compile_result`** (status, execution_time, stderr, return_code) and **`run_result`** (stdout, stderr, return_code, execution_time).
- **Lean in SDPO:** In `sandbox_fusion/utils.py`, **`"lean"` is one of `SUPPORTED_LANGUAGES`**. So you *can* send Lean code with `language="lean"` and get back compile + run results. The reward path that uses sandbox_fusion is built for **generic code** (test-case style: inputs/outputs, pass/fail). There is **no** Lean-specific parsing (no sorries, no goal/message structure). You would treat success/failure from compile/run status and stdout/stderr.
- **Timeouts and retries:**  
  - `DEFAULT_TIMEOUT = 10` s (shared compile/run).  
  - Request timeout = `compile_timeout + run_timeout + API_TIMEOUT` (10 s).  
  - Retries: `MAX_RETRIES = 3`, only on **504 Gateway Timeout**; linear backoff.
- **Deployment:** External service (e.g. FaaS/volcengine) or self-hosted Sandbox Fusion. No Modal; no in-process Lean server.

### 8.2 Our approach (Kimina Lean Server)

- **Server:** **Kimina Lean Server** (Project Numina), image `projectnumina/kimina-lean-server:2.0.0`, run as a subprocess inside a **Modal** container (`KiminaLeanServer`).
- **API:** POST to `http://localhost:8000/verify` (inside the same container) with:
  - Body: `{"codes": [{"custom_id", "proof"}], "infotree_type": "original"}`.
  - Response: **Lean-specific** — `results[].messages` (severity, data), `sorries`, `status`. We derive `success`, `complete`, `has_sorry`, and feedback text for SDPO.
- **Purpose:** **Lean 4 proof verification** (typecheck + no sorries) and **structured error feedback** for test-time RL (SDPO), not generic code test cases.
- **Timeouts:** HTTP 60 s, Modal 900/1200 s, retries 3 (see §7). Debug logs in `debug/lean_server/`.

### 8.3 Side-by-side summary

| Aspect | SDPO folder (verl) | Our pipeline |
|--------|--------------------|--------------|
| **Service** | Sandbox Fusion (Bytedance) | Kimina Lean Server (Project Numina) |
| **Role of “Lean”** | One of many languages (`language="lean"`) | Dedicated Lean 4 verification server |
| **API** | `/run_code` with code + language + compile/run timeouts | `/verify` with `codes[].proof` + `infotree_type` |
| **Response** | compile_result + run_result (status, stdout, stderr, return_code) | results[].messages, sorries, status (Lean-specific) |
| **Reward/use** | Pass rate over test cases (inputs/outputs); no Lean semantics | Proof success + complete + no sorry; feedback for SDPO |
| **Hosting** | External or self-hosted Sandbox Fusion | Modal: KiminaLeanServer container + Lean subprocess |
| **Default timeouts** | 10 s compile/run, 3 retries on 504 | 60 s HTTP, 3 retries; Modal 900/1200 s |

### 8.4 Important findings

1. **No Kimina in SDPO folder.** The SDPO (verl) code does not call or configure Kimina. Any “Lean” support there is via Sandbox Fusion with `language="lean"`, which is generic compile/run, not theorem-proving verification.
2. **Different goals.** Sandbox Fusion in verl is for **code correctness** (run code, compare stdout to expected). Our pipeline is for **Lean 4 proof verification** (typecheck, no sorries, structured feedback). They are not interchangeable without adding a Lean-aware layer on top of Sandbox Fusion or switching to a Lean-specific backend (e.g. Kimina).
3. **If we wanted “Sandbox Fusion + Lean” in our repo:** We would still need to interpret Lean compile/run output (and possibly run a Lean-specific backend behind Sandbox Fusion). We would not get Kimina’s `messages`/`sorries` format unless that backend were Kimina or compatible.
4. **Reusing ideas from SDPO folder.** Useful patterns we could consider (without changing our Kimina choice): explicit **request timeout** = compile_timeout + run_timeout + buffer; **retry only on 504** (or specific server errors) to avoid retrying on client/timeout errors; **concurrency control** (e.g. semaphore) if we ever batch many verification requests.
5. **Our choice of Kimina is appropriate** for Lean theorem proving and SDPO feedback. The SDPO folder’s Sandbox Fusion integration does not replace it for that use case.
