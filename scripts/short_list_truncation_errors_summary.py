"""
Summary of short_list (interesting) problems from round_3 runs, same format as unsolved_truncation_errors_summary.json.
Short list = problem indices from dynamic_sampling_results/interesting_problem_idx.json.
Merge attempts from round_3 run dirs (181436, 182852, 195008) for each short_list problem.
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


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    interesting_path = repo / "dynamic_sampling_results" / "interesting_problem_idx.json"
    round3_dir = repo / "dynamic_sampling_results" / "round_3"
    run_names = [
        "run_Qwen3.5-4B_20260312_181436",
        "run_Qwen3.5-4B_20260312_182852",
        "run_Qwen3.5-4B_20260312_195008",
        "run_Qwen3.5-4B_20260312_195015",
    ]
    out_path = repo / "dynamic_sampling_results" / "short_list_truncation_errors_summary.json"

    with open(interesting_path, encoding="utf-8") as f:
        interesting = json.load(f)
    # short_list: list of { problem_idx, problem_id, success_rate }
    problems_list = interesting.get("problems", interesting) if isinstance(interesting.get("problems"), list) else interesting
    if isinstance(problems_list, dict):
        problems_list = [{"problem_idx": int(k), "problem_id": v.get("problem_id", "")} for k, v in problems_list.items()]
    else:
        problems_list = problems_list if isinstance(problems_list, list) else []
    short_list_indices = {p["problem_idx"] for p in problems_list if "problem_idx" in p}
    problem_id_by_idx = {p["problem_idx"]: p.get("problem_id", "") for p in problems_list if "problem_idx" in p}

    run_dirs = [round3_dir / name for name in run_names]
    sources = []
    merged: dict[int, list[dict]] = {idx: [] for idx in short_list_indices}
    for run_dir in run_dirs:
        logs_path = run_dir / "logs.json"
        if not logs_path.is_file():
            continue
        sources.append(str(logs_path.resolve()))
        by_problem = load_attempts_by_problem(logs_path)
        for idx in short_list_indices:
            merged[idx].extend(by_problem.get(idx, []))

    problems_out = []
    for idx in sorted(short_list_indices):
        attempts = merged[idx]
        total = len(attempts)
        n_truncated = sum(1 for a in attempts if a.get("truncated") is True)
        n_non_truncated = total - n_truncated
        truncation_rate = round(n_truncated / total, 4) if total else 0.0
        total_errors = sum(n_errors(a) for a in attempts)
        errors_per_non_truncated_proof = round(total_errors / n_non_truncated, 4) if n_non_truncated else None
        n_successful = sum(1 for a in attempts if is_pass(a))
        n_unsuccessful = total - n_successful
        success_rate = round(n_successful / total, 4) if total else 0.0
        problems_out.append({
            "problem_idx": idx,
            "problem_id": problem_id_by_idx.get(idx, ""),
            "total_attempts": total,
            "n_successful": n_successful,
            "n_unsuccessful": n_unsuccessful,
            "success_rate": success_rate,
            "n_truncated": n_truncated,
            "n_non_truncated": n_non_truncated,
            "truncation_rate": truncation_rate,
            "total_error_messages": total_errors,
            "errors_per_non_truncated_proof": errors_per_non_truncated_proof,
        })

    def sort_key(p: dict) -> tuple[float, int]:
        e = p.get("errors_per_non_truncated_proof")
        return (e if e is not None else float("inf"), p["n_truncated"])
    problems_out.sort(key=sort_key)

    summary = {
        "description": "Short-list (interesting) problems: truncation and error stats from round_3 runs (181436, 182852, 195008, 195015).",
        "sources": sources,
        "n_problems": len(problems_out),
        "problems": problems_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path} (n={len(problems_out)} short_list problems)", flush=True)


if __name__ == "__main__":
    main()
