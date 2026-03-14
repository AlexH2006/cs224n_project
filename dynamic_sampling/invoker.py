"""
TLDR: Invoke qwen_eval via subprocess and discover the run dir and logs.

Writes problem_indices JSON, runs `python -m modal run qwen_eval/modal_app.py ...`,
discovers the run_* subdir under base_dir, returns path to logs.json and generation count.
Does not parse log contents; state module does.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dynamic_sampling.config import DynamicSamplingConfig


@dataclass
class RoundResult:
    """Result of one round: path to logs.json and number of generations this round."""

    logs_path: Path
    generations_this_round: int


def _model_safe(model_name: str) -> str:
    """'Qwen/Qwen3.5-4B' -> 'Qwen3.5-4B' for matching run dir names."""
    return model_name.split("/")[-1]


def discover_run_dir(base_dir: Path, model_name: str) -> Optional[Path]:
    """
    Under base_dir, find the run subdir (run_<model>_<timestamp>).
    Returns the single subdir if exactly one, else the newest by mtime.
    Returns None if no subdir or base_dir does not exist.
    """
    if not base_dir.is_dir():
        return None
    prefix = f"run_{_model_safe(model_name)}_"
    subdirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not subdirs:
        return None
    if len(subdirs) == 1:
        return subdirs[0]
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def run_round(
    config: DynamicSamplingConfig,
    remaining_indices: list[int],
    pass_k: int,
    base_dir: Path,
    *,
    index_filename: str = "problem_indices.json",
    repo_root: Optional[Path] = None,
) -> RoundResult:
    """
    Write problem_indices JSON, run qwen_eval for this round, discover run dir, return logs path and generation count.

    Args:
        config: DynamicSamplingConfig (model, kimina_url, etc.).
        remaining_indices: Problem indices to evaluate this round.
        pass_k: Number of attempts per problem.
        base_dir: Directory to pass as --results-base-dir; run dir will be base_dir/run_Model_timestamp/.
        index_filename: Name of the indices JSON file under base_dir (default problem_indices.json).
        repo_root: Working directory for subprocess (default: current working directory).

    Returns:
        RoundResult with logs_path and generations_this_round.

    Raises:
        FileNotFoundError: if run dir or logs.json not found after subprocess completes.
        subprocess.CalledProcessError: if modal run fails.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    index_path = base_dir / index_filename
    index_path.write_text(
        json.dumps({"problem_indices": remaining_indices}, indent=2),
        encoding="utf-8",
    )
    cwd = Path(repo_root) if repo_root is not None else Path.cwd()
    index_path_abs = index_path.resolve()
    # Pass absolute path so modal run can find it regardless of cwd
    index_path_for_cmd = str(index_path_abs)

    # Modal run expects a file path (e.g. qwen_eval/modal_app.py); module form uses -m and dots.
    module_arg = config.qwen_eval_module
    if not module_arg.endswith(".py"):
        module_arg = f"{module_arg}.py"
    cmd = [
        sys.executable,
        "-m",
        "modal",
        "run",
        module_arg,
        "--problem-index-file",
        index_path_for_cmd,
        "--pass-k",
        str(pass_k),
        "--results-base-dir",
        str(base_dir.resolve()),
        "--model",
        config.model,
        "--kimina-url",
        config.kimina_url,
    ]
    if config.no_think_mode:
        cmd.append("--no-think-mode")

    subprocess.run(cmd, cwd=str(cwd), check=True)

    run_dir = discover_run_dir(base_dir, config.model)
    if run_dir is None:
        raise FileNotFoundError(f"No run dir found under {base_dir}")
    logs_path = run_dir / "logs.json"
    if not logs_path.is_file():
        raise FileNotFoundError(f"logs.json not found at {logs_path}")
    summary_path = run_dir / "summary.json"
    generations_this_round = len(remaining_indices) * pass_k
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        gen_metrics = summary.get("generation_metrics") or {}
        if gen_metrics.get("n_requests") is not None:
            generations_this_round = gen_metrics["n_requests"]

    return RoundResult(logs_path=logs_path, generations_this_round=generations_this_round)
