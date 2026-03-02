# Baseline Inspection — Goedel 8B & Kimina 1.7B

Centralized notes for the minif2f-lean4 baseline (first 40 problems, pass@4). Both models achieve **35/40 (87.5%)** and fail on the **same 5 problems** (confirmed from both run summaries in the repo).

**References:** [baseline/README.md](../baseline/README.md) | Run: `modal run baseline/lean_baseline_eval_modal.py --n-problems 40 --pass-k 4`

---

## 1. Summary

| Model | Accuracy | Failed count | Summary / proofs |
|-------|----------|--------------|-------------------|
| **Goedel-Prover-V2-8B** | 35/40 (87.5%) | 5 | `results/goedel_8b_baseline/summary.json`, `proofs.jsonl` |
| **Kimina-Prover-RL-1.7B** | 35/40 (87.5%) | 5 | `results/kimina_1.7b_baseline/summary.json`, `proofs.jsonl` |

Failure criterion: no attempt (out of 4) verifies **without** `sorry` (Kimina verification: `complete: true`, `has_sorry: false`).

---

## 2. Goedel-Prover-V2-8B — Failed proof generations

Source: **`results/goedel_8b_baseline/summary.json`**  
Generated from run output path `results/run_20260226_213048/proofs.jsonl` (copied to `goedel_8b_baseline`).

### 2.1 Aggregate

- **n_problems:** 40  
- **n_success:** 35  
- **accuracy:** 0.875  
- **Verification source:** kimina  

### 2.2 Failed problems (5)

All five have **best chosen attempt** still **incomplete** (`verification.complete: false`, `verification.has_sorry: true`). No verification errors; feedback string empty in summary.

| problem_idx | problem_id | chosen_attempt | verification |
|-------------|------------|----------------|---------------|
| 2 | **aime_1983_p1** | 3 | complete: false, has_sorry: true |
| 6 | **imo_1969_p2** | 3 | complete: false, has_sorry: true |
| 19 | **amc12a_2020_p10** | 3 | complete: false, has_sorry: true |
| 28 | **mathd_algebra_320** | 3 | complete: false, has_sorry: true |
| 30 | **imo_1997_p5** | 3 | complete: false, has_sorry: true |

**Observation:** For each failed problem, the best attempt was **attempt 3** (last of 4). So the model never produced a fully verified proof in 4 tries for these IDs.

### 2.3 How to re-inspect (Goedel 8B)

```bash
# List failed IDs and first line of feedback
python baseline/scripts/inspect_baseline_summary.py results/goedel_8b_baseline/summary.json
```

Proof attempts (by problem_idx and attempt) are in `results/goedel_8b_baseline/proofs.jsonl` (sorted by problem_idx, then attempt).

---

## 3. Kimina-Prover-RL-1.7B — Failed proof generations

Source: **`results/kimina_1.7b_baseline/summary.json`**  
Generated from run output path `results/run_20260226_222315/proofs.jsonl` (copied to `kimina_1.7b_baseline`).

### 3.1 Aggregate

- **n_problems:** 40  
- **n_success:** 35  
- **accuracy:** 0.875  
- **Verification source:** kimina  

### 3.2 Failed problems (5) — same as Goedel 8B

Kimina fails on the **same 5 problem IDs** as Goedel. Same pattern: best chosen attempt = 3, `verification.complete: false`, `verification.has_sorry: true`.

| problem_idx | problem_id | chosen_attempt | verification |
|-------------|------------|----------------|---------------|
| 2 | **aime_1983_p1** | 3 | complete: false, has_sorry: true |
| 6 | **imo_1969_p2** | 3 | complete: false, has_sorry: true |
| 19 | **amc12a_2020_p10** | 3 | complete: false, has_sorry: true |
| 28 | **mathd_algebra_320** | 3 | complete: false, has_sorry: true |
| 30 | **imo_1997_p5** | 3 | complete: false, has_sorry: true |

### 3.3 How to re-inspect (Kimina 1.7B)

```bash
# List failed IDs and first line of feedback
python baseline/scripts/inspect_baseline_summary.py results/kimina_1.7b_baseline/summary.json
```

Proof attempts are in `results/kimina_1.7b_baseline/proofs.jsonl` (160 lines: 40 problems × 4 attempts; fields: `problem_idx`, `problem_id`, `attempt`, `full_code`).

To produce a **new** run (e.g. after changing the pipeline):  
`modal run baseline/lean_baseline_eval_modal.py --model AI-MO/Kimina-Prover-RL-1.7B --n-problems 40 --pass-k 4`  
Outputs go to `results/run_<timestamp>/` unless overridden with `--out-dir` / `--out-name`.

---

## 4. Other baseline-related artifacts

| Artifact | Model | Scope | Notes |
|----------|--------|--------|--------|
| [baseline/README.md](../baseline/README.md) | Both | 40 problems | How to run; 35/40 for both models |
| `results/goedel_8b_baseline/summary.json`, `proofs.jsonl` | Goedel-Prover-V2-8B | 40 problems, 40×4 attempts | Per-problem success, chosen_attempt, verification; proofs sorted by problem_idx, attempt |
| `results/kimina_1.7b_baseline/summary.json`, `proofs.jsonl` | Kimina-Prover-RL-1.7B | 40 problems, 40×4 attempts | Same structure; same 5 failed IDs as Goedel |
| [baseline/scripts/inspect_baseline_summary.py](../baseline/scripts/inspect_baseline_summary.py) | Any | — | Print failed problem_id and feedback from any summary.json |
| results/misc (minif2f_kimina_prover_*, etc.) | Kimina-Prover-Preview-Distill-1.5B | 5 problems | Different model; verification logs with errors |

---

## 5. Summary

- **Goedel-Prover-V2-8B** and **Kimina-Prover-RL-1.7B** both score **35/40** on the first 40 minif2f-lean4 test problems (pass@4) and fail on the **same 5 problems** (confirmed from both summaries in the repo).
- **Failed problem IDs (both models):** aime_1983_p1, imo_1969_p2, amc12a_2020_p10, mathd_algebra_320, imo_1997_p5. For each, best chosen attempt = 3 and verification is incomplete (has_sorry).
- **Goedel 8B:** `results/goedel_8b_baseline/summary.json`, `proofs.jsonl`. **Kimina 1.7B:** `results/kimina_1.7b_baseline/summary.json`, `proofs.jsonl`. Use `inspect_baseline_summary.py <path/to/summary.json>` for a quick list of failed IDs and feedback.
