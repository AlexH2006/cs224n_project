"""
Overlay pass@32 curves for parallel sampling (avg round1+round2) and SDPO on one graph.
Uses pass_at_k_round1_round2_avg.json (average_pct) and pass_at_k_summary.json (SDPO).
Saves to results/pass_at_k_32_parallel_vs_sdpo.png (does not overwrite pass_at_k_32_avg_parallel_sdpo.png).
X-axis ticks reduced to 1/4: e.g. 1, 5, 9, 13, 17, 21, 25, 29, 32.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parallel_path = repo / "results" / "Qwen3.5_4B_sampled_no_thinking_parallel_samplin" / "pass_at_k_round1_round2_avg.json"
    sdpo_path = repo / "results" / "Qwen3.5_4B_sampled_no_thinking_sdpo" / "pass_at_k_summary.json"
    out_path = repo / "results" / "pass_at_k_32_parallel_vs_sdpo.png"

    with open(parallel_path, encoding="utf-8") as f:
        parallel_data = json.load(f)
    with open(sdpo_path, encoding="utf-8") as f:
        sdpo_data = json.load(f)

    ks = [e["k"] for e in parallel_data["pass_at_k"]]
    parallel_pcts = [e["average_pct"] for e in parallel_data["pass_at_k"]]
    sdpo_pcts = [e["problems_solved_pct"] for e in sdpo_data["pass_at_k"]]
    assert len(ks) == len(parallel_pcts) == len(sdpo_pcts)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(ks, parallel_pcts, marker="o", linestyle="-", markersize=6, label="Parallel (avg round1+round2)")
    ax.plot(ks, sdpo_pcts, marker="s", linestyle="-", markersize=6, label="SDPO")
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title("Pass@k (n=50 problems, parallel vs SDPO pass@32)")
    # Reduce horizontal ticks by 4x: 32/4 = 8 ticks
    tick_vals = list(range(1, 33, 4))  # 1, 5, 9, 13, 17, 21, 25, 29
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
