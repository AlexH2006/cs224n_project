"""
Plot pass@k comparing:
  - 50 samples: parallel sampling (pass_at_k_round1_round2_avg.json from 50samples_no_think_parallel_samplin)
  - minif2f-test: full test set (pass@k computed from Qwen3.5-4B_all_no_thinking/logs.json, pass_k=8)

Format follows pass_at_k_32_parallel_vs_sdpo.png (axis labels, ticks 1,5,9,...,32, grid, legend).
minif2f-test has only k=1..8; we extend the curve flat to k=32 for a continuous line.
"""

from __future__ import annotations

import json
from pathlib import Path


def compute_pass_at_k_from_logs(problem_logs: list[dict], max_k: int) -> list[float]:
    """
    For each k in 1..max_k, compute % of problems with at least one success in first k attempts.
    Returns list of percentages (length max_k).
    """
    if not problem_logs:
        return [0.0] * max_k
    n = len(problem_logs)
    pcts = []
    for k in range(1, max_k + 1):
        count = 0
        for log in problem_logs:
            attempts = log.get("attempts", [])[:k]
            if any(a.get("success") for a in attempts):
                count += 1
        pcts.append(round(100.0 * count / n, 1))
    return pcts


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parallel_path = (
        repo
        / "results"
        / "Qwen3.5_4B_50samples_no_think"
        / "Qwen3.5_4B_50samples_no_think_parallel_samplin"
        / "pass_at_k_round1_round2_avg.json"
    )
    all_logs_path = repo / "results" / "Qwen3.5-4B_all_no_thinking" / "logs.json"
    out_path = repo / "results" / "Qwen3.5_4B_50samples_no_think" / "pass_at_k_50samples_vs_minif2f_test.png"

    with open(parallel_path, encoding="utf-8") as f:
        parallel_data = json.load(f)

    with open(all_logs_path, encoding="utf-8") as f:
        minif2f_logs = json.load(f)

    # 50 samples: k=1..32, average_pct
    ks_32 = [e["k"] for e in parallel_data["pass_at_k"]]
    pcts_50 = [e["average_pct"] for e in parallel_data["pass_at_k"]]
    assert ks_32 == list(range(1, 33)), "Expected k=1..32"

    # minif2f-test: compute pass@k for k=1..8 from logs
    max_k_minif2f = 8
    pcts_minif2f_raw = compute_pass_at_k_from_logs(minif2f_logs, max_k=max_k_minif2f)
    # Extend flat to k=32 (pass@8 = pass@9 = ... = pass@32)
    pcts_minif2f = pcts_minif2f_raw + [pcts_minif2f_raw[-1]] * (32 - max_k_minif2f)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(ks_32, pcts_50, marker="o", linestyle="-", markersize=6, label="50 samples")
    ax.plot(ks_32, pcts_minif2f, marker="s", linestyle="-", markersize=6, label="minif2f-test")
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title("Pass@k: 50 samples vs minif2f-test (parallel sampling)")
    tick_vals = list(range(1, 33, 4))
    if 32 not in tick_vals:
        tick_vals.append(32)
    ax.set_xticks(tick_vals)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
