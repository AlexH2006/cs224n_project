"""
Summary for ALL problems that appear in the four round_3 run folders:
  181436, 182852, 195008, 195015.
Merge attempts across these four runs; rank by success_rate descending.
"""

from __future__ import annotations

import json
from pathlib import Path


def is_pass(attempt: dict) -> bool:
    v = attempt.get("verification") or {}
    if not v:
        return bool(attempt.get("success"))
    return (
        v.get("success", False)
        and v.get("complete", False)
        and not v.get("has_sorry", True)
    )


def n_errors(attempt: dict) -> int:
    v = attempt.get("verification") or {}
    return len(v.get("errors") or [])


def load_attempts_by_problem(logs_path: Path) -> dict[int, list[dict]]:
    with open(logs_path, encoding="utf-8") as f:
        logs = json.load(f)
    out: dict[int, list[dict]] = {}
    for log in logs:
        problem = log.get("problem", {}) or {}
        idx = problem.get("problem_idx")
        if idx is None:
            continue
        attempts = log.get("attempts", [])
        out.setdefault(idx, []).extend(attempts)
    return out


def load_problem_id_by_idx(logs_path: Path) -> dict[int, str]:
    """problem_idx -> problem_id from first occurrence in logs."""
    with open(logs_path, encoding="utf-8") as f:
        logs = json.load(f)
    out: dict[int, str] = {}
    for log in logs:
        problem = log.get("problem", {}) or {}
        idx = problem.get("problem_idx")
        if idx is not None and idx not in out:
            out[idx] = problem.get("id", "")
    return out


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    round3_dir = repo / "dynamic_sampling_results" / "round_3"
    run_names = [
        "run_Qwen3.5-4B_20260312_181436",
        "run_Qwen3.5-4B_20260312_182852",
        "run_Qwen3.5-4B_20260312_195008",
        "run_Qwen3.5-4B_20260312_195015",
    ]
    out_path = repo / "dynamic_sampling_results" / "round_3_all_problems_summary.json"

    run_dirs = [round3_dir / name for name in run_names]
    sources = []
    merged: dict[int, list[dict]] = {}
    problem_id_by_idx: dict[int, str] = {}

    for run_dir in run_dirs:
        logs_path = run_dir / "logs.json"
        if not logs_path.is_file():
            continue
        sources.append(str(logs_path.resolve()))
        by_problem = load_attempts_by_problem(logs_path)
        ids = load_problem_id_by_idx(logs_path)
        for idx, attempts in by_problem.items():
            merged.setdefault(idx, []).extend(attempts)
            if idx not in problem_id_by_idx and idx in ids:
                problem_id_by_idx[idx] = ids[idx]

    all_indices = set(merged.keys())

    problems_out = []
    for idx in all_indices:
        attempts = merged[idx]
        total = len(attempts)
        n_truncated = sum(1 for a in attempts if a.get("truncated") is True)
        n_non_truncated = total - n_truncated
        truncation_rate = round(n_truncated / total, 4) if total else 0.0
        total_errors = sum(n_errors(a) for a in attempts)
        errors_per_non_truncated_proof = round(total_errors / n_non_truncated, 4) if n_non_truncated else None
        n_successful = sum(1 for a in attempts if is_pass(a))
        success_rate = round(n_successful / total, 4) if total else 0.0
        problems_out.append({
            "problem_idx": idx,
            "problem_id": problem_id_by_idx.get(idx, ""),
            "total_attempts": total,
            "n_successful": n_successful,
            "success_rate": success_rate,
            "n_truncated": n_truncated,
            "n_non_truncated": n_non_truncated,
            "truncation_rate": truncation_rate,
            "total_error_messages": total_errors,
            "errors_per_non_truncated_proof": errors_per_non_truncated_proof,
        })

    # Rank by success_rate descending (highest first)
    problems_out.sort(key=lambda p: (-p["success_rate"], p["problem_idx"]))

    summary = {
        "description": "All problems appearing in round_3 runs 181436, 182852, 195008, 195015; ranked by success_rate descending.",
        "sources": sources,
        "n_problems": len(problems_out),
        "problems": problems_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path} (n={len(problems_out)} problems, ranked by success_rate desc)", flush=True)


if __name__ == "__main__":
    main()
