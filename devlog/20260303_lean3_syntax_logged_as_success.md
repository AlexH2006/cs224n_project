# Lean 3 syntax in stored full_code logged as verification success

**Date:** 2026-03-03  
**Topics:** verification, kimina, lean3, lean4, sdpo, logs, bugfixes

---

## Summary

A run’s iteration log can record `verification.success: true` and `verification.complete: true` while the **stored** `full_code` for that iteration is written in **Lean 3** syntax. When that same `full_code` is re-submitted to the Kimina Lean Server (Lean 4), verification **fails**. This note documents the finding and how to re-verify stored proofs locally.

---

## Observed case

- **Log:** `sdpo_results/qwen_3b/minif2f-lean4/run_0_20260303_022443/logs.json`
- **Iteration:** 2, `full_code` at line 87 (mathd_algebra_478).

The log shows:

```json
"verification": {
  "success": true,
  "complete": true,
  "has_sorry": false,
  "feedback": "",
  "errors": [],
  "source": "kimina"
}
```

Re-verifying the **exact** `full_code` string from the log with the Kimina server (via `debug/verify_one_lean.py`) yields **FAIL**: multiple “expected token” errors and “invalid 'end', insufficient scopes”.

---

## Why the stored proof fails (Lean 3 vs Lean 4)

The stored proof uses Lean 3 conventions; Kimina uses Lean 4 + Mathlib 4.

| Issue | In stored `full_code` | In Lean 4 |
|--------|------------------------|-----------|
| Import | `import data.real.basic` | `import Mathlib` (or equivalent) |
| Tactic block | `begin` … `end` inside `by` | Tactics directly under `by`, no `begin`/`end` |
| Variables | `variable (...)` inside the proof | Not allowed; theorem already binds them |
| Substitution | `h₁.subst h₃` | e.g. `rw [h₃] at h₁` or `h₁ ▸ h₃` |
| `show` | `show v = 65, from h₇` | `show v = 65 from h₇` (no comma) |
| Block end | `end` | No `end` for a single theorem |

So the **content** of `full_code` in the log is **not** valid Lean 4; the verifier correctly rejects it when run again.

---

## Implications

1. **Logged success vs re-verification:** A log entry with `success: true` for a given `full_code` does not guarantee that the **same** string would pass Kimina today; in this case it fails.
2. **Possible causes for the original “success”:**  
   - The code actually sent to Kimina in that run may have differed (e.g. different assembly or normalization).  
   - Or a different verifier/version was used.  
   - Or a bug in the pipeline (e.g. wrong code attached to the result).  
   The exact cause is not determined here.
3. **Model output:** The model (Qwen 3B) produced Lean 3–style tactics (`begin`/`end`, `variable`, `.subst`). The pipeline should either normalize to Lean 4 before verification or treat Lean 3 syntax as invalid and not mark the run as a complete success.

---

## How to re-verify stored proofs locally

1. **Kimina server** (e.g. in Docker):
   ```bash
   docker run --rm -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
   ```
2. **Save the `full_code`** from the log to a `.lean` file (e.g. `debug/mathd_algebra_478_verify.lean`).
3. **Run the verifier script:**
   ```bash
   .venv/bin/python debug/verify_one_lean.py --file debug/mathd_algebra_478_verify.lean -o debug/mathd_algebra_478_verify_result.json
   ```
4. Inspect exit code and `debug/*_result.json` for errors.

Raw Kimina output for the mathd_algebra_478 re-verification is in `debug/mathd_algebra_478_verify_result.json`.

---

## Recommendations

- **Reproducibility:** When logging “success”, consider re-verifying the stored `full_code` in a separate step (or in CI) to ensure the logged string actually passes the current Lean 4 verifier.
- **Tactic extraction / assembly:** If the model can output Lean 3 syntax, add either (a) a Lean 3→4 normalization step before verification, or (b) detection of Lean 3 markers (`begin`/`end`, `import data.`, `.subst`, etc.) and treat as failure or “wrong dialect” so success is not reported for invalid Lean 4.
- **Documentation:** Point pipeline docs to this devlog for the distinction between “verification reported success in the run” and “stored full_code verifies under current Kimina”.
