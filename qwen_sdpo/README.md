# qwen_sdpo — Qwen 3.5 self-distillation on Modal

Self-distillation (SDPO) for a single Lean 4 problem: the model improves by distilling from itself using compiler feedback. Generation and gradient steps run on Modal (H100); parsing and verification run locally.

## Prerequisites

1. **Modal** — `pip install modal`, then `modal token new` (or set `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`).
2. **Lean verification server** — Must be running locally before you start. Example (Docker, port 8000):

   ```bash
   docker run -d -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
   ```

3. **Project root** — Run all commands from the **project root** (parent of `qwen_sdpo/`).

## How to run

### Single problem (default: problem index 0)

```bash
python3 -m modal run qwen_sdpo/modal_app.py
```

Override model, problem index, and common options:

```bash
python3 -m modal run qwen_sdpo/modal_app.py --model "Qwen/Qwen3.5-4B" --problem-idx 4
python3 -m modal run qwen_sdpo/modal_app.py --model "Qwen/Qwen3.5-9B" --problem-idx 0 --max-iterations 5
```

### Batch (multiple problems)

Each problem is trained from the base HuggingFace model (reset between problems). By default uses `qwen_sdpo/problem_idx.json`; override with `--problems-json` (path relative to project root).

```bash
python3 -m modal run qwen_sdpo/modal_app.py::run_sdpo_batch --model "Qwen/Qwen3.5-4B"
python3 -m modal run qwen_sdpo/modal_app.py::run_sdpo_batch --model "Qwen/Qwen3.5-4B" --problems-json results/sampled_problems.json
```

### Common CLI options (both entrypoints)

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `Qwen/Qwen3.5-4B` | HuggingFace model (e.g. 4B or 9B). |
| `--problem-idx` | `0` | Dataset index for single-problem run. |
| `--max-iterations` | `5` | Max SDPO iterations per problem. |
| `--dataset` | `cat-searcher/minif2f-lean4` | HuggingFace dataset. |
| `--dataset-split` | `test` | Dataset split. |
| `--learning-rate` | (from config) | AdamW learning rate. |
| `--temperature` | `0.6` | Sampling temperature. |
| `--feedback-errors-only` | `True` | Teacher prompt: errors only (vs errors + failed proof). |
| `--use-think-mode` | `True` | Allow `<think>` reasoning; `false` = non-thinking mode. |
| `--teacher-mode` | `full_output` | KL target: `full_output`, `answer_only`, or `code_only`. |
| `--minibatch-size` | `1` | Samples per iteration (vLLM batch size). |
| `--kl-mask-final-code-only` | `False` | Restrict KL loss to final code block. |
| `--kimina-url` | `http://localhost:8000` | Verification server URL. |
| `--results-base-dir` | `sdpo_results` | Local output base (single-problem only). |

## Output

- **Single problem:** `{results_base_dir}/{model_tag}/{teacher_mode}/run_{model_tag}_{problem_idx}_{timestamp}/` — logs, metrics, training curves, local copy of run.
- **Batch:** `sdpo_results/{model_tag}/run_{model_tag}_{timestamp}/` with `runs/problem_*/` per problem and `manifest/checkpoint_manifest.json`.

Algorithm and workflow details: [docs/README_SDPO.md](../docs/README_SDPO.md), [docs/QWEN_SDPO_WORKFLOW.md](../docs/QWEN_SDPO_WORKFLOW.md), [docs/SDPO_TRAINER_DEEP_DIVE.md](../docs/SDPO_TRAINER_DEEP_DIVE.md).
