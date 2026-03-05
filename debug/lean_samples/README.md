# Lean samples

Sample Lean files for manual inspection or for use with `debug/verify_one_lean.py`.

| File | Notes |
|------|-------|
| `mathd_algebra_478.lean` | Lean 4 / Mathlib4 — cone volume problem (known good) |
| `mathd_algebra_478_verify.lean` | Lean 3 syntax — same problem; verifies as "success" on Kimina despite wrong syntax (see devlog/20260303_lean3_syntax_logged_as_success.md) |
| `mathd_numbertheory_3_from_run5.lean` | Lean 4 — number theory problem extracted from Qwen 3B run 5 |
| `qwen_3b_run5_mathd_numbertheory_3.lean` | Lean 4 — Qwen 3B run 5 full output for mathd_numbertheory_3 |
| `amc12_2001_p5.lean` | Lean 4 — AMC 2001 problem 5 attempt |
| `imosl_2007_algebra_p6.lean` | Lean 4 — IMO SL 2007 algebra P6 attempt (may use invalid APIs) |
| `known_bad/mathd_algebra_478_from_log.lean` | Minimal failing example (undefined identifier) for testing error paths |
