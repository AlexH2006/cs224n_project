# Baseline evaluation

Lean proof baseline evaluation pipeline: generate proof attempts with vLLM on Modal, verify with the Kimina Lean Server, report pass@k on [minif2f-lean4](https://huggingface.co/datasets/cat-searcher/minif2f-lean4).

## Results (first 40 problems of minif2f, pass@4)

Both models achieve **35/40 (87.5%)** on the first 40 problems of the minif2f-lean4 test split:

| Model | Accuracy |
|-------|----------|
| **Goedel-Prover-V2-8B** | 35/40 (87.5%) |
| **Kimina-Prover-RL-1.7B** | 35/40 (87.5%) |

## How to run

From the repo root:

```bash
# Default model (Goedel-Prover-V2-8B)
modal run baseline/lean_baseline_eval_modal.py --n-problems 40 --pass-k 4

# Kimina 2B
modal run baseline/lean_baseline_eval_modal.py --model AI-MO/Kimina-Prover-RL-1.7B --n-problems 40 --pass-k 4
```

Outputs (proofs JSONL and summary) are written under `results/run_<timestamp>/` by default. Use `--out-dir` to change the base directory and `--out-name` to specify an explicit output path.

## Contents

- **lean_baseline_eval_modal.py** — Modal app: parallel proof generation (vLLM) and serial verification (Kimina), with timeout/retry/reset logic for robust runs.
