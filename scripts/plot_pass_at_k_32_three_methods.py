"""
Plot pass@32 (or pass@k for k=1..32) comparing three methods:
  - Multi-turn correction (from multi_turn logs.json)
  - SDPO
  - Parallel sampling (avg round1+round2)

Data under results/Qwen3.5_4B_50samples_no_think. Styling for SDPO and Parallel
matches pass_at_k_32_parallel_vs_sdpo.png; multi-turn is added with a distinct style.
"""

from __future__ import annotations

import json
from pathlib import Path


def compute_multiturn_pass_at_k_from_logs(logs: list[dict], max_k: int = 32) -> list[dict]:
    """
    From multi-turn logs.json: each problem has one attempt with rounds 0..R.
    For each k in 1..max_k, compute % of problems that succeeded within the first k rounds
    (rounds 0..k-1). Returns list of {"k": k, "problems_solved_pct": pct} for k=1..max_k.
    """
    n = len(logs)
    if n == 0:
        return [{"k": k, "problems_solved_pct": 0.0} for k in range(1, max_k + 1)]

    # First success round per problem (0-indexed). 33 = never solved.
    first_success_round = []
    for log in logs:
        attempts = log.get("attempts") or []
        if not attempts:
            first_success_round.append(max_k + 1)
            continue
        rounds_list = attempts[0].get("rounds") or []
        found = max_k + 1
        for r in rounds_list:
            if r.get("success"):
                found = r.get("round", max_k + 1)
                break
        first_success_round.append(found)

    pass_at_k = []
    for k in range(1, max_k + 1):
        # Succeeded within first k rounds => first_success_round < k (0-indexed: rounds 0..k-1)
        count = sum(1 for r in first_success_round if r < k)
        pct = round(100.0 * count / n, 1)
        pass_at_k.append({"k": k, "problems_solved_pct": pct})
    return pass_at_k


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    base = repo / "results" / "Qwen3.5_4B_50samples_no_think"

    parallel_path = base / "Qwen3.5_4B_50samples_no_think_parallel_samplin" / "pass_at_k_round1_round2_avg.json"
    sdpo_path = base / "Qwen3.5_4B_50samples_no_think_sdpo" / "pass_at_k_summary.json"
    multiturn_logs_path = base / "Qwen3.5_4B_50samples_no_think_multi_turn" / "logs.json"
    out_path = base / "pass_at_k_32_three_methods.png"

    with open(parallel_path, encoding="utf-8") as f:
        parallel_data = json.load(f)
    with open(sdpo_path, encoding="utf-8") as f:
        sdpo_data = json.load(f)
    with open(multiturn_logs_path, encoding="utf-8") as f:
        multiturn_logs = json.load(f)

    ks = [e["k"] for e in parallel_data["pass_at_k"]]
    parallel_pcts = [e["average_pct"] for e in parallel_data["pass_at_k"]]
    sdpo_pcts = [e["problems_solved_pct"] for e in sdpo_data["pass_at_k"]]

    multiturn_pass_at_k = compute_multiturn_pass_at_k_from_logs(multiturn_logs, max_k=32)
    multiturn_ks = [e["k"] for e in multiturn_pass_at_k]
    multiturn_pcts = [e["problems_solved_pct"] for e in multiturn_pass_at_k]

    assert len(ks) == len(parallel_pcts) == len(sdpo_pcts) == 32
    assert len(multiturn_ks) == 32 and multiturn_ks == ks

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # Same style as pass_at_k_32_parallel_vs_sdpo.png for parallel and SDPO
    ax.plot(ks, parallel_pcts, marker="o", linestyle="-", markersize=6, label="Parallel sampling")
    ax.plot(ks, sdpo_pcts, marker="s", linestyle="-", markersize=6, label="Self-distillation")
    # Multi-turn: distinct marker/line so all three are distinguishable
    ax.plot(ks, multiturn_pcts, marker="^", linestyle="-", markersize=6, label="Multi-turn correction")
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title("Parallel Sampling vs Self-distillation vs Multi-turn Correction on 50 minif2f Problems")
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
