"""
Compute average pass@k (k=1..32) of round1 (4 runs merged) and round2 (1 run),
plot the average curve in the parent folder, and write a JSON of pass@k for each k.

Usage:
  python scripts/plot_pass_at_k_round1_round2_avg.py \\
    results/Qwen3.5_4B_sampled_no_thinking_parallel_samplin/pass@32_round1 \\
    results/Qwen3.5_4B_sampled_no_thinking_parallel_samplin/pass@32_round2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def is_pass(attempt: dict[str, Any]) -> bool:
    verification = attempt.get("verification") or {}
    if not verification:
        return bool(attempt.get("success"))
    return (
        verification.get("success", False)
        and verification.get("complete", False)
        and not verification.get("has_sorry", True)
    )


def first_success_attempt_in_run(attempts: list[dict[str, Any]]) -> int | None:
    for i, a in enumerate(attempts):
        if is_pass(a):
            return i + 1
    return None


def load_run_logs(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "logs.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def collect_run_dirs(parent_dir: Path) -> list[Path]:
    run_dirs = []
    for child in sorted(parent_dir.iterdir()):
        if child.is_dir() and (child / "logs.json").exists():
            run_dirs.append(child)
    return run_dirs


ATTEMPTS_PER_RUN = 8


def merged_first_success_for_problem(problem_logs_across_runs: list[dict[str, Any]]) -> int | None:
    best: int | None = None
    for run_index, log in enumerate(problem_logs_across_runs):
        attempts = log.get("attempts") or []
        first_in_run = first_success_attempt_in_run(attempts)
        if first_in_run is not None:
            global_attempt = run_index * ATTEMPTS_PER_RUN + first_in_run
            if best is None or global_attempt < best:
                best = global_attempt
    return best


def compute_round1_first_success(run_dirs: list[Path]) -> list[int | None]:
    runs_data = [load_run_logs(d) for d in run_dirs]
    n = len(runs_data[0])
    for i, data in enumerate(runs_data):
        if len(data) != n:
            raise ValueError(f"Run {run_dirs[i].name} has {len(data)} problems, expected {n}")
    first_success = []
    for p in range(n):
        problem_logs = [run_data[p] for run_data in runs_data]
        first_success.append(merged_first_success_for_problem(problem_logs))
    return first_success


def compute_round2_first_success(run_dir: Path) -> list[int | None]:
    problem_logs = load_run_logs(run_dir)
    return [
        first_success_attempt_in_run(log.get("attempts") or [])
        for log in problem_logs
    ]


def pass_at_k_pcts(first_success: list[int | None], max_k: int = 32) -> list[float]:
    n = len(first_success)
    return [100.0 * sum(1 for a in first_success if a is not None and a <= k) / n for k in range(1, max_k + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average pass@k of round1 and round2, plot and write JSON.",
    )
    parser.add_argument("round1_dir", type=Path, help="pass@32_round1 directory (4 run subdirs)")
    parser.add_argument("round2_dir", type=Path, help="pass@32_round2 directory (1 run subdir)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plot and JSON (default: parent of round1_dir)",
    )
    parser.add_argument("--max-k", type=int, default=32)
    args = parser.parse_args()

    round1_dir = Path(args.round1_dir).resolve()
    round2_dir = Path(args.round2_dir).resolve()
    if args.output_dir is not None:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = round1_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    max_k = args.max_k

    run_dirs_r1 = collect_run_dirs(round1_dir)
    if len(run_dirs_r1) != 4:
        raise SystemExit(f"Round1: expected 4 run dirs, found {len(run_dirs_r1)} in {round1_dir}")

    run_dirs_r2 = collect_run_dirs(round2_dir)
    if len(run_dirs_r2) != 1:
        raise SystemExit(f"Round2: expected 1 run dir, found {len(run_dirs_r2)} in {round2_dir}")
    run_dir_r2 = run_dirs_r2[0]

    first_r1 = compute_round1_first_success(run_dirs_r1)
    first_r2 = compute_round2_first_success(run_dir_r2)
    n1, n2 = len(first_r1), len(first_r2)

    pcts_r1 = pass_at_k_pcts(first_r1, max_k)
    pcts_r2 = pass_at_k_pcts(first_r2, max_k)
    pcts_avg = [(a + b) / 2.0 for a, b in zip(pcts_r1, pcts_r2)]

    ks = list(range(1, max_k + 1))
    pass_at_k_data = {
        "max_k": max_k,
        "n_problems_round1": n1,
        "n_problems_round2": n2,
        "round1_dir": str(round1_dir),
        "round2_dir": str(round2_dir),
        "pass_at_k": [
            {
                "k": k,
                "round1_pct": round(pcts_r1[i], 4),
                "round2_pct": round(pcts_r2[i], 4),
                "average_pct": round(pcts_avg[i], 4),
            }
            for i, k in enumerate(ks)
        ],
    }

    json_path = out_dir / "pass_at_k_round1_round2_avg.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pass_at_k_data, f, indent=2)
    print(f"Wrote {json_path}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(ks, pcts_avg, marker="o", linestyle="-", markersize=6, color="C0")
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title(f"Pass@k (n={n1} problems, average of round1 & round2)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plot_path = out_dir / "pass_at_k_32_round1_round2_avg.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
