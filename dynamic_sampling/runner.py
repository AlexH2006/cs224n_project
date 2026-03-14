"""
TLDR: Orchestrator for dynamic_sampling: loop rounds (invoke qwen_eval, update state) until budget or no problems.

Uses invoker to run each round, state to update from logs, output to write final summary and raw_logs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from dynamic_sampling.config import DynamicSamplingConfig
from dynamic_sampling.invoker import run_round, RoundResult
from dynamic_sampling.output import RoundRecord, write_outputs
from dynamic_sampling.state import initial_state, initial_state_from_indices, RoundState


def _load_problem_indices(path: Path) -> list[int]:
    """Load problem_indices from a JSON file. Expects {"problem_indices": [int, ...]}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    indices = data.get("problem_indices", [])
    if not isinstance(indices, list):
        raise ValueError(f"problem_index_file must contain 'problem_indices' array, got {type(indices)}")
    return [int(x) for x in indices]


def run(
    config: DynamicSamplingConfig,
    *,
    repo_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Run dynamic_sampling until budget exhausted or no problems remain.

    For each round: compute pass_k = max(1, floor(remaining_budget / n)), invoke qwen_eval,
    load logs and update state, then repeat. Writes summary.json and raw_logs.json to output_dir.

    Args:
        config: DynamicSamplingConfig (budget, model, n_problems, etc.).
        repo_root: Working directory for subprocess (default: current directory).
        output_dir: Where to write summary and raw_logs (default: config.output_dir with run subdir).

    Returns:
        Path to the directory containing summary.json and raw_logs.json.
    """
    base = Path(output_dir or config.output_dir)
    if config.problem_index_file:
        path = Path(config.problem_index_file)
        if not path.is_absolute() and repo_root is not None:
            path = Path(repo_root) / path
        if not path.is_file():
            raise FileNotFoundError(f"problem_index_file not found: {path}")
        initial_indices = _load_problem_indices(path)
        state = initial_state_from_indices(initial_indices)
        n_problems_total = len(state.remaining)
    else:
        n_problems = min(max(0, config.n_problems), 244)  # cap at MiniF2F test size
        state = initial_state(n_problems)
        n_problems_total = n_problems
    base.mkdir(parents=True, exist_ok=True)
    round_records: list[RoundRecord] = []
    round_index = 0

    while state.remaining and state.total_generations < config.budget:
        n = len(state.remaining)
        pass_k = max(1, int(config.budget // n))

        round_base = base / f"round_{round_index}"
        result: RoundResult = run_round(
            config,
            state.remaining,
            pass_k,
            round_base,
            repo_root=repo_root,
        )
        logs_data = json.loads(result.logs_path.read_text(encoding="utf-8"))
        state.update_from_round_logs(logs_data, pass_k=pass_k, round_index=round_index)
        round_records.append(
            RoundRecord(
                round_index=round_index,
                pass_k=pass_k,
                generations_this_round=result.generations_this_round,
                logs_path=result.logs_path,
            )
        )
        round_index += 1

    write_outputs(base, state, round_records, n_problems_total=n_problems_total)
    return base
