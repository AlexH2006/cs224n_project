# Verification takes forever on iteration 3–4 (important issue)

**Date:** 2026-03-02  
**Topics:** verification, kimina, sdpo, performance, modal

---

## Observed behavior

During SDPO runs (e.g. PutnamBench problem 0, `modal run ... --max-iterations 5`), **verification appears to hang** on the third or fourth iteration:

- Iterations 1–2: "Generated N chars" → short pause → "Verification: FAILED" (or SUCCESS).
- Iteration 3 or 4: After "Generated N chars", the run sits with **no further output** for a long time (minutes). Users may assume a hang and interrupt (Ctrl+C). On interrupt, the traceback shows the process blocked in `LeanVerifier().verify.remote(full_code)`.

Example from a run that was interrupted during iteration 4 verification:

```
--- Iteration 4/5 ---
  Generated 16316 chars
Disconnecting from Modal - This will terminate your Modal app in a few seconds.
...
  File ".../lean_sdpo_kimina_2b_modal.py", line 646, in run_sdpo
    verification = LeanVerifier().verify.remote(full_code)
...
ERROR ... Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.
```

The "Engine core died" message is a consequence of the app shutting down (e.g. user interrupt), not necessarily the root cause of slowness.

---

## Root cause analysis

### 1. No progress indicator (confirmed)

There is **no log line between** "Generated N chars" and the verification result. The main process is blocked on `LeanVerifier().verify.remote(full_code)` for the full duration of the Kimina request. So from the user’s perspective the run appears to hang.

### 2. Verification can legitimately take a long time

- **LeanVerifier** Modal timeout: **900 s** (15 min).
- **KiminaLeanServer** HTTP client timeout: **600 s** (10 min) per request.
- For a single iteration, verification can therefore run up to ~10 minutes with no output, which feels like "forever."

### 3. Why later iterations might be slower

- **Problem difficulty**: Hard problems (e.g. Putnam) can require heavy typechecking. Each verification attempt sends `full_code` (header + theorem + tactics) to the Kimina Lean server; complex or large proofs take longer to typecheck.
- **Kimina server cold start**: `KiminaLeanServer` has `scaledown_window=300`. If 5+ minutes pass between iterations (e.g. long generation + long previous verification), the Kimina container may scale down. The next verification then starts a new container and pays the **~17.7 s** server startup before the first HTTP verify.
- **full_code size**: `full_code` is built from the same theorem each time but with new tactics each iteration. If later attempts produce longer or more complex tactic scripts, the blob sent to the verifier is larger and can take longer to process.

### 4. Summary of causes

| Cause | Effect |
|-------|--------|
| No "Verifying..." (or similar) log | Run looks like it hung after "Generated N chars". |
| Long Lean typecheck (hard problem / large proof) | Verification runs up to 600 s per attempt with no progress. |
| Kimina container scaled down | Extra ~18 s cold start before the first verify in a new container. |
| High Modal/HTTP timeouts | Verification is allowed to run 10–15 min, so "forever" is possible. |

---

## Recommendations

1. **Add a progress message** before calling `LeanVerifier().verify.remote(...)` so users see that verification is in progress (e.g. `"  Verifying with Kimina Lean Server..."`). Optionally log again when verification returns and report elapsed time.
2. **Consider a lower effective timeout** for verification (e.g. 120–180 s) with a clear "Verification timeout" result and retry, so hard problems fail fast instead of blocking for 10+ minutes.
3. **Optional**: Emit a heartbeat or periodic log from the verifier for very long runs (e.g. every 60 s) so users know the process is still working.
4. **Optional**: Cap or trim `full_code` size sent to the verifier (e.g. max chars) to avoid sending huge tactic blocks that Lean will struggle with.

---

## Status

**Recorded as an important issue.** No code change in this devlog beyond recommending the progress message; a follow-up change can add that and/or the timeout/heartbeat improvements above.

---

## Fix applied: Lean server health check and auto-restart

### Updated root cause

The `KiminaLeanServer` Modal container has `scaledown_window=600`, so it stays alive between iterations. However the Lean server **subprocess** (`python -m server`) inside that container can crash or become unresponsive after several `import Mathlib` verifications (REPL pool exhaustion / memory accumulation). The `@modal.enter()` startup only runs once when the container first starts — there was no mechanism to detect or restart a dead subprocess. On iteration 3–4 the container is reused but the subprocess is dead, causing every HTTP request to time out at 60 s × 3 retries, cascading through the outer retry loop.

### Changes (all four training files: `lean_sdpo_kimina_2b_modal.py`, `lean_sdpo_kimina_distill_1_7b_modal.py`, `lean_sdpo_goedel_8b_modal.py`, `lean_sdpo_deepseek_7b_modal.py`)

1. **Extracted `_start_lean_server()`**: Reusable method that kills any existing subprocess, spawns a new one, and blocks until the health-check endpoint responds (up to 60 s). Called from `@modal.enter()` on first boot and from any restart path.

2. **Added `_ensure_server_alive()`**: Called at the top of every `verify()` invocation. Checks two things:
   - `self.server_proc.poll() is not None` → subprocess exited → restart.
   - Quick 5 s health-check HTTP POST → if it fails → restart.
   
   This catches both "process crashed" and "process alive but hung" before we attempt the real verification.

3. **Retry-on-failure restarts the server**: On `ConnectError`, `TimeoutException`, or any other exception during a verify attempt, the retry loop now calls `_start_lean_server()` instead of just sleeping. This ensures the next attempt talks to a fresh Lean server rather than retrying against a dead one.

**Backport:** The same KiminaLeanServer health-check and restart logic was applied to `lean_sdpo_goedel_8b_modal.py` and `lean_sdpo_deepseek_7b_modal.py`. All four training files now use identical verification server behavior. See `20260302_bugfixes_verification.md` for the full parity summary (unsolved goals, truncated → sorry, and server restart).
