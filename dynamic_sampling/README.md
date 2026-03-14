# Dynamic Sampling

Multi-round evaluation on MiniF2F with a **total attempt budget**. Runs pass@1 for all problems first, then repeatedly on remaining problems with pass@floor(remaining_budget/n) until the budget is exhausted or no problems remain.

## Input

- **Budget**: Maximum number of attempts (generations) across all problems (e.g. 256 or 244×8).
- Optional: model, output dir, Kimina URL, etc.

## Output

All results are written under a run directory `dynamic_sampling_results/run_<Model>_<timestamp>/` (or a custom `--output-dir`):

1. **summary.json**: Per-problem pass summary and aggregate stats.
   - `aggregate`: `n_problems_total`, `n_problems_attempted`, `n_passed`, `total_generations`, `pass_rate`.
   - `problems`: list of `{problem_idx, problem_id, passed, attempts_used, round_finished}`.

2. **raw_logs.json**: Full raw results per round.
   - `rounds`: list of `{round_index, pass_k, generations_this_round, logs_path, logs}` (logs = full qwen_eval logs.json content).

Round subdirs `round_0/`, `round_1/`, … contain the qwen_eval run dirs and problem_indices.json for each round.

## Usage

1. Start Kimina (for verification):
   ```bash
   docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
   ```

2. Run dynamic sampling (from repo root):
   ```bash
   python -m dynamic_sampling.entrypoint --budget 256
   python -m dynamic_sampling.entrypoint --budget 512 --model Qwen/Qwen3.5-9B --output-dir my_results
   python -m dynamic_sampling.entrypoint --budget 244 --n-problems 50 --use-thinking
   python -m dynamic_sampling.entrypoint --budget 100 --problem-index-file dynamic_sampling/problem_idx.json
   ```
   Use `--problem-index-file` to run only on a JSON file of indices (e.g. `{"problem_indices": [2, 3, 6, ...]}`) for the first round.

3. By default Qwen `<think>...</think>` is disabled (qwen_eval `--no-think-mode`). Use `--use-thinking` to enable it.

4. Optional: `--repo-root PATH` to set the working directory for the qwen_eval subprocess (default: current directory).

## Dependencies

Uses the existing **qwen_eval** Modal pipeline. No new pip dependencies. Requires Modal and the qwen_eval package (and `--results-base-dir` support in `qwen_eval/modal_app.py`).

## Tests

From repo root:

```bash
python -m pytest dynamic_sampling/tests/ -v
```
