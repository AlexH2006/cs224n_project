"""
Plot pass@8 (k=1..8) comparing 50 samples vs minif2f-test with error bars.

- 50 samples: Use all generated data. Round1 has 4 runs of 8 attempts each; round2 has
  1 run of 32 attempts — split into 4 blocks of 8. So we have 8 "runs" of 8 attempts.
  At each k=1..8 compute mean and std across these 8 runs → curve with error bars.
- minif2f-test: Single run of 8 attempts per problem. Pass@k for k=1..8. Bootstrap
  over problems (sample with replacement, repeat 500 times) to get std at each k.

Format follows pass_at_k_32_parallel_vs_sdpo.png (axis labels, grid, legend); x-axis is k=1..8.
"""

from __future__ import annotations

import json
import random
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


def pass_at_k_pcts_from_first_success(first_success: list[int | None], max_k: int) -> list[float]:
    """Given first success attempt index per problem (1-based or None), return [pct at k=1, ..., k=max_k]."""
    n = len(first_success)
    if n == 0:
        return [0.0] * max_k
    return [
        100.0 * sum(1 for a in first_success if a is not None and a <= k) / n
        for k in range(1, max_k + 1)
    ]


def first_success_attempt(attempts: list[dict], max_attempts: int | None = None) -> int | None:
    """First 1-based attempt index that passed, or None. Only considers attempts[:max_attempts] if set."""
    if max_attempts is not None:
        attempts = attempts[:max_attempts]
    for i, a in enumerate(attempts):
        if is_pass(a):
            return i + 1
    return None


def load_run_logs(run_dir: Path) -> list[dict]:
    with open(run_dir / "logs.json", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected list")
    return data


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    base = repo / "results" / "Qwen3.5_4B_50samples_no_think" / "Qwen3.5_4B_50samples_no_think_parallel_samplin"
    round1_dir = base / "pass@32_round1"
    round2_dir = base / "pass@32_round2"
    all_logs_path = repo / "results" / "Qwen3.5-4B_all_no_thinking" / "logs.json"
    out_path = repo / "results" / "Qwen3.5_4B_50samples_no_think" / "pass_at_k_8_50samples_vs_minif2f_test.png"

    max_k = 8

    # ---- 50 samples: 8 runs of 8 attempts ----
    # Round1: 4 run dirs, each 50 problems × 8 attempts
    run_dirs_r1 = sorted([d for d in round1_dir.iterdir() if d.is_dir() and (d / "logs.json").exists()])
    if len(run_dirs_r1) != 4:
        raise SystemExit(f"Round1: expected 4 run dirs, got {len(run_dirs_r1)}")
    curves_50 = []
    for run_dir in run_dirs_r1:
        logs = load_run_logs(run_dir)
        first_success = [
            first_success_attempt(entry.get("attempts") or [], max_attempts=8)
            for entry in logs
        ]
        pcts = pass_at_k_pcts_from_first_success(first_success, max_k)
        curves_50.append(pcts)

    # Round2: 1 run, 50 problems × 32 attempts → 4 blocks of 8
    run_dirs_r2 = sorted([d for d in round2_dir.iterdir() if d.is_dir() and (d / "logs.json").exists()])
    if len(run_dirs_r2) != 1:
        raise SystemExit(f"Round2: expected 1 run dir, got {len(run_dirs_r2)}")
    logs_r2 = load_run_logs(run_dirs_r2[0])
    for block_start in [0, 8, 16, 24]:
        first_success = [
            first_success_attempt((entry.get("attempts") or [])[block_start : block_start + 8], max_attempts=None)
            for entry in logs_r2
        ]
        pcts = pass_at_k_pcts_from_first_success(first_success, max_k)
        curves_50.append(pcts)

    # Mean and std across 8 curves at each k
    import statistics
    ks = list(range(1, max_k + 1))
    mean_50 = [statistics.mean(c[k - 1] for c in curves_50) for k in ks]
    std_50 = [statistics.stdev(c[k - 1] for c in curves_50) if len(curves_50) > 1 else 0.0 for k in ks]

    # ---- minif2f-test: pass@8 from logs, bootstrap for std ----
    with open(all_logs_path, encoding="utf-8") as f:
        minif2f_logs = json.load(f)
    n_minif2f = len(minif2f_logs)
    rng = random.Random(42)
    n_bootstrap = 500
    bootstrap_curves = []
    for _ in range(n_bootstrap):
        indices = rng.choices(range(n_minif2f), k=n_minif2f)
        sampled = [minif2f_logs[i] for i in indices]
        first_success = [
            first_success_attempt(entry.get("attempts") or [], max_attempts=8)
            for entry in sampled
        ]
        pcts = pass_at_k_pcts_from_first_success(first_success, max_k)
        bootstrap_curves.append(pcts)
    mean_minif2f = [statistics.mean(c[k - 1] for c in bootstrap_curves) for k in ks]
    std_minif2f = [statistics.stdev(c[k - 1] for c in bootstrap_curves) for k in ks]

    # ---- Plot ----
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.errorbar(ks, mean_50, yerr=std_50, marker="o", linestyle="-", markersize=6, capsize=3, label="50 samples")
    ax.errorbar(ks, mean_minif2f, yerr=std_minif2f, marker="s", linestyle="-", markersize=6, capsize=3, label="minif2f-test")
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title("Pass@8: 50 samples vs minif2f-test (parallel sampling)")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
