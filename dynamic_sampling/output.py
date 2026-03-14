"""
TLDR: Write summary.json and raw_logs.json from final RoundState and round records.

Given aggregated state and per-round metadata (logs path, pass_k, generations), writes
output_dir/summary.json and output_dir/raw_logs.json. No parsing of logs beyond reading JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynamic_sampling.state import ProblemResult, RoundState


@dataclass
class RoundRecord:
    """Metadata for one round: index, pass_k, generations, path to logs.json."""

    round_index: int
    pass_k: int
    generations_this_round: int
    logs_path: Path


def write_outputs(
    output_dir: Path,
    state: RoundState,
    round_records: list[RoundRecord],
    *,
    n_problems_total: int,
) -> None:
    """
    Write summary.json and raw_logs.json under output_dir.

    Args:
        output_dir: Directory to write into (created if needed).
        state: Final round state (results, total_generations).
        round_records: Per-round metadata and logs path (order preserved).
        n_problems_total: Total number of problems in the dataset (for aggregate stats).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build summary: per-problem list + aggregate stats
    problems_summary = []
    for idx in sorted(state.results.keys()):
        r = state.results[idx]
        problems_summary.append({
            "problem_idx": r.problem_idx,
            "problem_id": r.problem_id,
            "passed": r.passed,
            "attempts_used": r.attempts_used,
            "round_finished": r.round_finished,
        })
    n_passed = sum(1 for r in state.results.values() if r.passed)
    aggregate = {
        "n_problems_total": n_problems_total,
        "n_problems_attempted": len(state.results),
        "n_passed": n_passed,
        "total_generations": state.total_generations,
        "pass_rate": round(n_passed / len(state.results), 4) if state.results else 0.0,
    }
    summary = {
        "aggregate": aggregate,
        "problems": problems_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Build raw_logs: list of rounds with metadata and full logs content
    raw_rounds = []
    for rec in round_records:
        logs_path = Path(rec.logs_path)
        logs_content = []
        if logs_path.is_file():
            logs_content = json.loads(logs_path.read_text(encoding="utf-8"))
        raw_rounds.append({
            "round_index": rec.round_index,
            "pass_k": rec.pass_k,
            "generations_this_round": rec.generations_this_round,
            "logs_path": str(logs_path),
            "logs": logs_content,
        })
    raw_logs = {"rounds": raw_rounds}
    raw_path = output_dir / "raw_logs.json"
    raw_path.write_text(json.dumps(raw_logs, indent=2, ensure_ascii=False), encoding="utf-8")
