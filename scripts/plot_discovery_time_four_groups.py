"""
Plot percent problems passed vs number of events for four discovery-time groups:
- Multi-turn correction (QWen3.5_4B_discovery_time_multi_turn)
- Parallel sampling (Qwen3.5_4B_discovery_time_parallel_sampling), excluding problem 20
- SDPO KL_avg_loss+5e-6lr+code_mask
- SDPO KL_sum_loss+1e-5lr

Style similar to pass_at_k_32_three_methods.png: Problems solved (%) vs k (number of attempts).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def pass_at_k_pcts_from_first_success(
    first_success: list[int | None], max_k: int, n_problems: int | None = None
) -> list[float]:
    """Given first success attempt index per problem (1-based or None), return [pct at k=1, ..., k=max_k]."""
    n = n_problems if n_problems is not None else len(first_success)
    if n == 0:
        return [0.0] * max_k
    return [
        100.0 * sum(1 for a in first_success if a is not None and a <= k) / n
        for k in range(1, max_k + 1)
    ]


def load_multiturn_first_success(repo: Path) -> list[int | None]:
    base = repo / "results" / "Qwen3.5_4B_discovery_time" / "QWen3.5_4B_discovery_time_multi_turn"
    run_dir = next(base.iterdir())  # single run
    path = run_dir / "discovery_time_summary.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # best_round is 0-based; event = round + 1 (1-based)
    out = []
    for p in data["problems"]:
        r = p.get("best_round")
        out.append((r + 1) if r is not None and p.get("solved") else None)
    return out


def load_parallel_avg_pass_at_k(repo: Path, max_k: int) -> list[float]:
    """Pass@k for parallel sampling = average of round_1, round_2, and round_3 pass@k at each k.
    Loads from discovery_time_summary.json (10 problems, has round_1/round_2/round_3 discovery per problem).
    """
    path = (
        repo
        / "results"
        / "Qwen3.5_4B_discovery_time"
        / "Qwen3.5_4B_discovery_time_parallel_sampling"
        / "discovery_time_summary.json"
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = 10  # 10 problems common across rounds
    first_success_r1 = [r["round_1_discovery_generations"] for r in data["runs"]]
    first_success_r2 = [r["round_2_discovery_generations"] for r in data["runs"]]
    first_success_r3 = [r["round_3_discovery_generations"] for r in data["runs"]]
    pcts_r1 = pass_at_k_pcts_from_first_success(first_success_r1, max_k, n)
    pcts_r2 = pass_at_k_pcts_from_first_success(first_success_r2, max_k, n)
    pcts_r3 = pass_at_k_pcts_from_first_success(first_success_r3, max_k, n)
    return [(p1 + p2 + p3) / 3 for p1, p2, p3 in zip(pcts_r1, pcts_r2, pcts_r3)]


def load_sdpo_first_success(repo: Path, config: str) -> list[int | None]:
    """Load discovery_generations per run; for KL_sum we have multiple runs per problem, take best (min) per problem_idx."""
    if "KL_avg" in config:
        path = (
            repo
            / "results"
            / "Qwen3.5_4B_discovery_time"
            / "Qwen3.5_4B_discovery_time_sdpo"
            / "KL_avg_loss+5e-6lr+code_mask"
            / "discovery_time_summary.json"
        )
    else:
        path = (
            repo
            / "results"
            / "Qwen3.5_4B_discovery_time"
            / "Qwen3.5_4B_discovery_time_sdpo"
            / "KL_sum_loss+1e-5lr"
            / "discovery_time_summary.json"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    runs = data["runs"]
    # One entry per problem_idx: take best (min discovery_generations) if multiple runs
    by_idx: dict[int, int | None] = {}
    for r in runs:
        idx = r["problem_idx"]
        d = r.get("discovery_generations")
        if idx not in by_idx:
            by_idx[idx] = d
        else:
            if d is not None and (by_idx[idx] is None or d < by_idx[idx]):
                by_idx[idx] = d
    # Preserve order by problem_idx (same 10 as other configs)
    order = sorted(by_idx.keys())
    return [by_idx[i] for i in order]


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    max_k = 100  # events; reference used 32

    # Multi-turn: 10 problems, event = best_round + 1
    multi_first = load_multiturn_first_success(repo)
    n_multi = len(multi_first)
    multi_pcts = pass_at_k_pcts_from_first_success(multi_first, max_k, n_multi)

    # Parallel: avg pass@k from round_1, round_2, and round_3 (10 problems)
    parallel_pcts = load_parallel_avg_pass_at_k(repo, max_k)

    # SDPO KL_avg: 10 problems
    sdpo_avg_first = load_sdpo_first_success(repo, "KL_avg")
    n_avg = len(sdpo_avg_first)
    sdpo_avg_pcts = pass_at_k_pcts_from_first_success(sdpo_avg_first, max_k, n_avg)

    # SDPO KL_sum: 10 problems (deduped by problem_idx)
    sdpo_sum_first = load_sdpo_first_success(repo, "KL_sum")
    n_sum = len(sdpo_sum_first)
    sdpo_sum_pcts = pass_at_k_pcts_from_first_success(sdpo_sum_first, max_k, n_sum)

    # Plot (style similar to pass_at_k_32_three_methods.png)
    import matplotlib.pyplot as plt

    ks = list(range(1, max_k + 1))
    xticks = [1, 5, 9, 13, 17, 21, 25, 29, 32]
    if max_k > 32:
        xticks.extend([40, 60, 80, 100])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ks, parallel_pcts, "C0", linestyle="-", label="Parallel sampling (avg)")
    ax.plot(ks, sdpo_avg_pcts, "C1", linestyle="-", label="SDPO (KL avg, 5e-6 lr)")
    ax.plot(ks, multi_pcts, "C2", linestyle="-", label="Multi-turn correction")
    ax.plot(ks, sdpo_sum_pcts, "C3", linestyle="-", label="SDPO (KL sum, 1e-5 lr)")

    ax.set_xlabel("Number of generations")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title("Discovery time: Percent problems passed vs number of generations")
    ax.set_xticks(xticks)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xlim(1, max_k)

    out_path = (
        repo
        / "results"
        / "Qwen3.5_4B_discovery_time"
        / "pass_at_k_discovery_time_four_groups.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
