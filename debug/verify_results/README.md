# Verify results

Saved JSON responses from the Kimina Lean Server for sample `.lean` files. For reference only; no code depends on these.

| File | Source |
|------|--------|
| `mathd_algebra_478_verify_result.json` | `lean_samples/mathd_algebra_478.lean` |
| `mathd_numbertheory_3_verify_result.json` | `lean_samples/mathd_numbertheory_3_from_run5.lean` |
| `qwen_3b_run5_mathd_numbertheory_3_verify_result.json` | `lean_samples/qwen_3b_run5_mathd_numbertheory_3.lean` |
| `imosl_2007_algebra_p6_verify_result.json` | `lean_samples/imosl_2007_algebra_p6.lean` |

Regenerate with:

```bash
python debug/verify_one_lean.py -f debug/lean_samples/<file>.lean -o debug/verify_results/<name>_verify_result.json
```
