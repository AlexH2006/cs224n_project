# Baseline evaluation

Lean proof baseline evaluation pipeline: generate proof attempts with vLLM on Modal, verify with the Kimina Lean Server, report pass@k on [HaimingW/minif2f-lean4](https://huggingface.co/datasets/HaimingW/minif2f-lean4) (MiniF2F-test, paper-style).

## Results

| Model | Problems | Pass@k | Accuracy |
|-------|----------|--------|----------|
| **Goedel-Prover-V2-8B** | 40 (first in order) | 4 | 35/40 (87.5%) |
| **Kimina-Prover-RL-1.7B** | 100 (random sample) | 1 | 78/100 (78.0%) |

- **Goedel 87.5%:** First 40 problems of minif2f-lean4, Pass@4. Full run in `run_goedel_8b_40_87p5/`.
- **Kimina 78%:** 100 randomly sampled problems from test split, Pass@1, 30s verifier timeout. Full run in `run_kimina_1_7b_100_78/`.

## How to run

From the repo root:

```bash
# Default: 244 problems (full test set), Pass@1, Goedel-Prover-V2-8B
modal run baseline/lean_baseline_eval_modal.py

# Kimina 1.7B, 100 problems, Pass@1
modal run baseline/lean_baseline_eval_modal.py --model AI-MO/Kimina-Prover-RL-1.7B --n-problems 100

# Kimina 1.7B, full 244 test problems, Pass@4
modal run baseline/lean_baseline_eval_modal.py --model AI-MO/Kimina-Prover-RL-1.7B --n-problems 244 --pass-k 4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-problems` | 244 | Number of problems. If >= 244, uses all test problems in order. Else random sample. |
| `--pass-k` | 1 | Attempts per problem. Pass if any attempt succeeds. |
| `--model` | Goedel-Prover-V2-8B | HuggingFace model for generation |
| `--dataset` | HaimingW/minif2f-lean4 | Dataset |
| `--split` | test | Split (test only for paper-style eval) |
| `--seed` | 42 | Random seed for problem sampling |
| `--verify-only` | False | Skip generation; verify existing proofs JSONL |
| `--generate-only` | False | Generate only; skip verification |
| `--out-name` | (auto) | Path to proofs JSONL (for verify-only or custom output) |

### Outputs

- `results/run_<model>_<timestamp>/proofs.jsonl` — Lean proofs (problem_idx, problem_id, attempt, full_code)
- `results/run_<model>_<timestamp>/full_outputs.jsonl` — Full model outputs including raw text
- `results/run_<model>_<timestamp>/summary.json` — Pass@k accuracy and per-problem results

### Requirements

- Modal installed (`pip install modal`)
- HuggingFace token for gated models: `modal secret create huggingface HF_TOKEN=...`

## Contents

- **lean_baseline_eval_modal.py** — Modal app: parallel proof generation (vLLM), serial verification (Kimina), 30s timeout, retry/reset on server errors.
- **run_goedel_8b_40_87p5/** — Goedel-Prover-V2-8B run: 40 problems, 87.5%, Pass@4 (proofs.jsonl, summary.json).
- **run_kimina_1_7b_100_78/** — Kimina-Prover-RL-1.7B run: 100 problems, 78%, Pass@1 (proofs.jsonl, full_outputs.jsonl, summary.json).
- **VERIFICATION_REPORT.md** — Verification checklist for MiniF2F-test alignment.
