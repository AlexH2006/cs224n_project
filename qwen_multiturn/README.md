# qwen_multiturn

Evaluation pipeline for Qwen 3.5 on MiniF2F (Lean 4), with Modal H100 for generation and Kimina for verification.

## Pipeline overview

1. **Generate** — Batched vLLM on Modal H100
2. **Parse** — Extract lean4 code block from raw output
3. **Verify** — Kimina Lean Server on localhost
4. **Save** — `logs.json`, `summary.json`, etc. under `test_results_multiturn/`

## Prerequisites

- Python 3.10+ with `modal`, `datasets`, `transformers` (or install from project `requirements.txt`)
- [Modal](https://modal.com) account and CLI (`modal token new`)
- Kimina Lean Server (Docker) — must be running locally before eval:
  ```bash
  docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
  ```

## Quick start

```bash
# Start Kimina first (in another terminal)
docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0

# Default: 20 problems, pass@4, no correction
python3 -m modal run qwen_multiturn/modal_app.py
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--n-problems` | 20 | Number of problems to evaluate |
| `--problem-idx` | -1 | Single problem index (overrides n-problems when >= 0) |
| `--parallelizations` | 4 | Parallel attempts per problem |
| `--model` | Qwen/Qwen3.5-4B | HuggingFace model |
| `--correction-rounds` | 0 | Self-correction rounds per sample (0 = off) |
| `--think-mode` | False | Enable <think>...</think> reasoning |
| `--generate-only` | False | Skip verification |
| `--kimina-url` | http://localhost:8000 | Kimina server URL |
| `--generation-save-batch-size` | None | Save after each N problems |
| `--inference-batch-size` | None | Max prompts per vLLM call |
| `--temperature` | 0.6 | Sampling temperature |
| `--top-p` | 0.95 | Top-p sampling |
| `--seed` | 42 | Random seed |

## Output layout

`test_results_multiturn/run_{model}_{timestamp}/`

- `logs.json` — Per-problem detail, attempts, verification
- `summary.json` — Pass@k, round accuracies, metadata
- `success_rate_summary.json` — Per-problem success rates
- `correction_rounds_data.json`, `correction_rounds_performance.png` — When `correction_rounds > 0`

## Correction rounds

When `--correction-rounds N`, each failed sample receives verifier feedback and is re-generated up to N times. The model sees only its own previous attempt and Kimina errors — not the reference solution.

## Utilities

- **check_lean.py** — Paste Lean code, verify via Kimina:
  ```bash
  python -m qwen_multiturn.check_lean
  ```

- **plot_correction_rounds_subset.py** — Cumulative solved-by-round for a subset:
  ```bash
  python -m qwen_multiturn.utils.plot_correction_rounds_subset RUN_DIR [PROBLEM_INDICES_JSON]
  ```

- **plot_pass_at_k.py** — Pass@k curve from logs:
  ```bash
  python -m qwen_multiturn.utils.plot_pass_at_k /path/to/logs.json
  ```

- **sample_and_plot_pass_at_k.py** — Stratified sample and plot:
  ```bash
  python -m qwen_multiturn.utils.sample_and_plot_pass_at_k RUN_DIR [RESULTS_DIR]
  ```

## Tests

```bash
pytest qwen_multiturn/tests/ -v
```

## Package layout

- `modal_app.py` — Modal app, generation, orchestration
- `config.py` — EvalConfig and defaults
- `dataset.py` — Load problems from HuggingFace
- `prompts.py` — Round 0 and correction prompts
- `parsing.py` — Extract lean4 blocks
- `batch_generation.py` — Build flat prompts for vLLM
- `local_lean_verifier.py`, `kimina_transport.py` — Verification
- `results.py` — Save logs, summary, success rates
- `utils/` — Plotting, subset analysis
