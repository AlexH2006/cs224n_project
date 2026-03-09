"""
TLDR: Persist eval results to disk in a structured, human-readable format.

Output layout:
    baseline/run_{model_safe}_{YYYYMMDD_HHMMSS}/
        logs.json     — per-problem detail (problem metadata, all attempts, verification)
        summary.json  — top-level accuracy stats and run metadata

The logs.json format matches the sdpo_results/local_verify convention so results
are comparable across pipelines.

Used by: modal_app.py (after verification phase completes).
"""

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from qwen_eval.config import EvalConfig


def _model_safe(model_name: str) -> str:
    """Convert 'Qwen/Qwen3.5-4B' → 'Qwen3.5-4B' for use in directory names."""
    return model_name.split("/")[-1]


def make_run_dir(cfg: EvalConfig) -> Path:
    """
    Create and return the run output directory.

    Path: {cfg.results_base_dir}/run_{model_safe}_{YYYYMMDD_HHMMSS}/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _model_safe(cfg.model_name)
    run_dir = Path(cfg.results_base_dir) / f"run_{model_tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_results(
    run_dir: Path,
    cfg: EvalConfig,
    problem_logs: list[dict[str, Any]],
) -> None:
    """
    Write logs.json and summary.json to run_dir.

    Args:
        run_dir:      Directory created by make_run_dir().
        cfg:          EvalConfig snapshot for this run.
        problem_logs: List of per-problem dicts, each with shape:
            {
              "problem":      {id, split, formal_statement, header, informal_stmt},
              "config":       EvalConfig as dict,
              "attempts":     [{attempt, prompt, raw_output, extracted_block,
                                full_code, verification, success, num_tokens}, ...],
              "success":      bool,
              "best_attempt": int | null,
              "best_proof":   str | null,
            }
    """
    cfg_dict = dataclasses.asdict(cfg)

    # Attach config snapshot to each problem log (for self-contained records).
    for log in problem_logs:
        log["config"] = cfg_dict

    # -------------------------------------------------------------------------
    # logs.json — full detail per problem
    # -------------------------------------------------------------------------
    logs_path = run_dir / "logs.json"
    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(problem_logs, f, indent=2, ensure_ascii=False)
    print(f"Saved logs    → {logs_path}")

    # -------------------------------------------------------------------------
    # summary.json — accuracy stats and run metadata
    # -------------------------------------------------------------------------
    n_problems = len(problem_logs)
    n_success = sum(1 for log in problem_logs if log.get("success"))
    pass_at_k = round(n_success / n_problems, 4) if n_problems > 0 else 0.0

    summary = {
        "model": cfg.model_name,
        "dataset": cfg.dataset_name,
        "split": cfg.dataset_split,
        "n_problems": n_problems,
        "pass_k": cfg.pass_k,
        "n_success": n_success,
        "pass_at_k": pass_at_k,
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        # Sampling params for reproducibility
        "sampling": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "top_k": cfg.top_k,
            "min_p": cfg.min_p,
            "presence_penalty": cfg.presence_penalty,
            "repetition_penalty": cfg.repetition_penalty,
            "max_new_tokens": cfg.max_new_tokens,
            "seed": cfg.seed,
        },
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary → {summary_path}")
    print(f"\npass@{cfg.pass_k}: {n_success}/{n_problems} = {pass_at_k:.1%}")


def build_problem_log(
    problem: dict[str, Any],
    attempts: list[dict[str, Any]],
    cfg: EvalConfig,
) -> dict[str, Any]:
    """
    Build a single problem log entry from generation + verification results.

    Args:
        problem:  Problem dict from dataset.py (problem_id, formal_statement, etc.)
        attempts: List of attempt dicts, each with keys:
                    attempt, prompt, raw_output, extracted_block,
                    full_code, verification, num_tokens
        cfg:      EvalConfig (used for dataset split label).

    Returns:
        A problem log dict ready to be included in problem_logs for save_results().
    """
    # Find the first successful attempt (success=True, complete=True, no sorry).
    best_attempt = None
    best_proof = None
    for att in attempts:
        v = att.get("verification", {})
        if v.get("success") and v.get("complete") and not v.get("has_sorry"):
            best_attempt = att["attempt"]
            best_proof = att["extracted_block"]
            break

    # Attach success flag to each attempt entry.
    for att in attempts:
        v = att.get("verification", {})
        att["success"] = bool(v.get("success") and v.get("complete") and not v.get("has_sorry"))

    return {
        "problem": {
            "id": problem["problem_id"],
            "problem_idx": problem["problem_idx"],
            "dataset": cfg.dataset_name,
            "split": cfg.dataset_split,
            "formal_statement": problem["formal_statement"],
            "header": problem["header"],
            "informal_stmt": problem["informal_stmt"],
        },
        "attempts": attempts,
        "success": best_attempt is not None,
        "best_attempt": best_attempt,
        "best_proof": best_proof,
    }
