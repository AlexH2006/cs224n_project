#!/usr/bin/env python3
"""
Generate a discovery-time summary for each run in results/Qwen3.5_4B_discovery_time/KL_sum_loss.
Each run has SDPO logs (iteration_logs with samples). Flatten all samples in order; discovery time
= first generation (1-based) at which a sample passes; estimated = total_generations / proofs_passed.
Output: results/Qwen3.5_4B_discovery_time/KL_sum_loss/discovery_time_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path


def is_pass(sample: dict) -> bool:
    v = sample.get("verification") or {}
    if not v:
        return bool(sample.get("success"))
    return (
        v.get("success", False)
        and v.get("complete", False)
        and not v.get("has_sorry", True)
    )


def discovery_time_for_run(logs_path: Path) -> dict | None:
    """Load SDPO logs.json; return per-run discovery time dict or None if invalid."""
    if not logs_path.is_file():
        return None
    with open(logs_path, encoding="utf-8") as f:
        data = json.load(f)
    problem_id = data.get("problem_id") or data.get("problem", {}).get("problem_id") or ""
    problem_idx = data.get("problem", {}).get("problem_idx")
    # Flatten: all samples in order (generation 1, 2, 3, ...)
    samples: list[dict] = []
    for it_log in data.get("iteration_logs", []):
        samples.extend(it_log.get("samples", []))
    total_generations = len(samples)
    proofs_passed = sum(1 for s in samples if is_pass(s))
    first_pass_generation = None
    for i, s in enumerate(samples):
        if is_pass(s):
            first_pass_generation = i + 1
            break
    estimated = round(total_generations / proofs_passed, 2) if proofs_passed else None
    return {
        "run_dir": logs_path.parent.name,
        "problem_idx": problem_idx,
        "problem_id": problem_id,
        "total_generations": total_generations,
        "proofs_passed": proofs_passed,
        "first_pass_generation": first_pass_generation,
        "estimated_discovery_time": estimated,
    }


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    base = repo / "results" / "Qwen3.5_4B_discovery_time" / "KL_sum_loss"
    out_path = base / "discovery_time_summary.json"

    run_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
    runs: list[dict] = []
    for run_dir in run_dirs:
        logs_path = run_dir / "logs.json"
        row = discovery_time_for_run(logs_path)
        if row:
            runs.append(row)

    # Optional overall stats
    solved = [r for r in runs if r["proofs_passed"] and r["estimated_discovery_time"]]
    summary = {
        "description": "Discovery time per run (SDPO logs): first_pass_generation = 1-based generation of first passing sample; estimated_discovery_time = total_generations / proofs_passed.",
        "n_runs": len(runs),
        "n_solved": len(solved),
        "runs": runs,
    }
    if solved:
        summary["mean_estimated_discovery_time"] = round(
            sum(r["estimated_discovery_time"] for r in solved) / len(solved), 2
        )
        summary["mean_first_pass_generation"] = round(
            sum(r["first_pass_generation"] for r in solved) / len(solved), 2
        )

    base.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path} (n_runs={len(runs)}, n_solved={len(solved)})", flush=True)


if __name__ == "__main__":
    main()
