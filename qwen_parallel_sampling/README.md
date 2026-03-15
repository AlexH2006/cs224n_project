# qwen_eval — Qwen MiniF2F-Lean4 pass@k evaluation

Pass@k evaluation for Qwen 3.5 on MiniF2F (Lean 4). Generation runs on Modal (H100, vLLM); parsing and verification run locally.

## Prerequisites

1. **Modal** — `pip install modal`, then `modal token new` (or set `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`).
2. **Lean verification server** — Must be running locally before you run eval (except with `--generate-only`). Example (Docker, port 8000):

   ```bash
   docker run -d -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
   ```

3. **Project root** — Run all commands from the **project root** (parent of `qwen_eval/`).

## How to run

Invoke the Modal app; the local entrypoint `run_eval` orchestrates generate → parse → verify → save.

### Default (20 problems, pass@4)

```bash
python3 -m modal run qwen_eval/modal_app.py
```

### Common options

```bash
# Model and scale
python3 -m modal run qwen_eval/modal_app.py --model "Qwen/Qwen3.5-9B" --n-problems 20 --pass-k 4

# Single problem by index
python3 -m modal run qwen_eval/modal_app.py --problem-idx 0 --pass-k 4

# Subset of problems from a JSON file (e.g. qwen_eval/problem_idx.json)
python3 -m modal run qwen_eval/modal_app.py --problem-index-file qwen_eval/problem_idx.json

# Disable <think> reasoning (non-thinking mode)
python3 -m modal run qwen_eval/modal_app.py --no-think-mode

# Save results after every N problems (incremental)
python3 -m modal run qwen_eval/modal_app.py --generation-save-batch-size 50

# Generation only (no verification; for testing)
python3 -m modal run qwen_eval/modal_app.py --generate-only
```

### CLI options (summary)

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen3.5-4B` | HuggingFace model. |
| `--n-problems` | `20` | Number of problems (ignored if `--problem-idx` or `--problem-index-file` set). |
| `--problem-idx` | `-1` | Single problem at this dataset index (`-1` = use `--n-problems`). |
| `--problem-index-file` | — | JSON file with problem indices (e.g. `qwen_eval/problem_idx.json`). |
| `--pass-k` | `4` | Samples per problem for pass@k. |
| `--no-think-mode` | `False` | Disable `<think>` blocks (tokenizer `enable_thinking=False`). |
| `--temperature` | `0.6` | Sampling temperature. |
| `--max-new-tokens` | `8192` | Max tokens per generation. |
| `--kimina-url` | `http://localhost:8000` | Verification server URL. |
| `--generate-only` | `False` | Skip verification. |
| `--inference-batch-size` | — | Max prompts per vLLM batch (default from config). |
| `--generation-save-batch-size` | — | Save after each N problems. |
| `--results-base-dir` | — | Override output base dir (default: `baseline`). |

## Output

Results go to `{results_base_dir}/run_{model_safe}_{timestamp}/` (default base: `baseline/`), with:

- `logs.json` — per-problem attempt logs (prompt, raw output, extracted block, full code, verification).
- `summary.json` — run summary (e.g. success rate, timings).
- `success_rate_summary.json` — pass@k success-rate summary.

For pass@k plots and sampling utilities, see `qwen_eval/utils/` (e.g. `plot_pass_at_k.py`, `sample_and_plot_pass_at_k.py`).
