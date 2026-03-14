#!/usr/bin/env python3
"""
Summarize pass@k for each problem in an SDPO batch run (e.g. discovery_time/KL_sum_loss+1e-5lr/run_XXX).
Each problem has runs/problem_{idx}/logs.json with iteration_logs and samples. Flatten samples
in order; first_success_attempt = 1-based generation of first pass. Output pass@k curve and
per-problem summary.

If run_dir is a parent folder containing multiple run_* subdirs (each with runs/problem_*/),
all problems from all runs are merged into one combined pass@k summary.
"""

from __future__ import annotations

import argparse
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


def first_success_from_sdpo_logs(logs_path: Path) -> tuple[int | None, int, str, int]:
    """Return (first_success_1based, total_generations, problem_id, problem_idx)."""
    with open(logs_path, encoding="utf-8") as f:
        data = json.load(f)
    problem_id = data.get("problem_id") or data.get("problem", {}).get("problem_id") or ""
    problem_idx = data.get("problem", {}).get("problem_idx")
    samples = []
    for it_log in data.get("iteration_logs", []):
        samples.extend(it_log.get("samples", []))
    total = len(samples)
    first = None
    for i, s in enumerate(samples):
        if is_pass(s):
            first = i + 1
            break
    return (first, total, problem_id, problem_idx)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize pass@k per problem for an SDPO batch run"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory with runs/problem_*/logs.json, or parent dir with run_*/runs/ (merge all runs)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: run_dir/pass_at_k_summary.json)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=128,
        help="Max k for pass@k curve (default: 128)",
    )
    args = parser.parse_args()

    base = Path(args.run_dir).resolve()
    # Single run: base/runs/problem_*
    # Parent with multiple runs: base/run_*/runs/problem_*
    if (base / "runs").is_dir():
        run_dirs = [base]
    else:
        run_dirs = sorted(
            [d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_") and (d / "runs").is_dir()],
            key=lambda d: d.name,
        )
        if not run_dirs:
            raise SystemExit(f"No runs dir at {base / 'runs'} and no run_*/runs under {base}")

    problems = []
    first_success_list = []
    for run_dir in run_dirs:
        runs_dir = run_dir / "runs"
        problem_dirs = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")],
            key=lambda d: int(d.name.replace("problem_", "")) if d.name.replace("problem_", "").isdigit() else 0,
        )
        for pdir in problem_dirs:
            logs_path = pdir / "logs.json"
            if not logs_path.is_file():
                continue
            first_success, total, problem_id, problem_idx = first_success_from_sdpo_logs(logs_path)
            problems.append({
                "problem_idx": problem_idx,
                "problem_id": problem_id,
                "total_generations": total,
                "first_success_attempt": first_success,
                "solved": first_success is not None,
                "run_dir": run_dir.name,
            })
            first_success_list.append(first_success)

    # Sort by problem_idx for stable output
    problems.sort(key=lambda p: (p["problem_idx"] if p["problem_idx"] is not None else -1))
    first_success_list = [p["first_success_attempt"] for p in problems]

    n = len(problems)
    max_k = min(args.max_k, max((p["total_generations"] for p in problems), default=0))
    pass_at_k = []
    for k in range(1, max_k + 1):
        count = sum(1 for fs in first_success_list if fs is not None and fs <= k)
        pass_at_k.append({"k": k, "pass_at_k": round(100.0 * count / n, 2) if n else 0, "n_solved": count})

    summary = {
        "description": "Pass@k from SDPO run(s); each problem has one logs.json with iteration_logs/samples. first_success_attempt = 1-based generation of first passing sample.",
        "run_dirs": [d.name for d in run_dirs],
        "n_problems": n,
        "n_solved": sum(1 for p in problems if p["solved"]),
        "problems": problems,
        "pass_at_k_curve": pass_at_k,
    }

    out_path = Path(args.output).resolve() if args.output else base / "pass_at_k_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path} (n_problems={n}, n_solved={summary['n_solved']})", flush=True)


if __name__ == "__main__":
    main()
