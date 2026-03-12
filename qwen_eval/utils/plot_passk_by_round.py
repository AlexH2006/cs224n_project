"""
Plot pass@k vs k with separate curves per correction round.

Uses the Codex/AlphaCode unbiased estimator:
  pass@k = 1 - C(n - c, k) / C(n, k)
where n = total attempts per problem, c = number of successes (by round r).

Used by: modal_app.run_eval() at end of pipeline.
"""

from __future__ import annotations

from math import comb
from pathlib import Path

from qwen_eval.config import EvalConfig


def _attempt_succeeded_by_round(attempt: dict, round_idx: int) -> bool:
    """
    True if this attempt succeeded at or before the given round.

    One-shot mode: use attempt["verification"] (no rounds key).
    Correction mode: check rounds with round_idx <= target for success.
    """
    rounds = attempt.get("rounds")
    if rounds is not None:
        for r in rounds:
            if r.get("round_idx", -1) > round_idx:
                continue
            v = r.get("verification") or {}
            if v.get("success") and v.get("complete") and not v.get("has_sorry"):
                return True
        return False
    v = attempt.get("verification") or {}
    return bool(v.get("success") and v.get("complete") and not v.get("has_sorry"))


def _compute_pass_at_k_for_round(
    problem_logs: list[dict],
    round_idx: int,
    pass_k: int,
) -> tuple[list[int], list[float]]:
    """
    Compute pass@k for a given round using Codex/AlphaCode formula.

    pass@k = 1 - C(n - c, k) / C(n, k)
    where n = pass_k, c = number of attempts that succeeded by round r.
    Averaged over all problems.

    Returns:
        ks: [1, 2, ..., pass_k]
        pct_solved: pass@k percentages (0-100) for each k
    """
    if not problem_logs or pass_k < 1:
        return [], []

    valid_logs = [log for log in problem_logs if log.get("attempts")]
    n_problems = len(valid_logs)
    if n_problems == 0:
        return [], []

    ks = list(range(1, pass_k + 1))
    pct_solved = []

    for k in ks:
        total_pass = 0.0
        for log in valid_logs:
            attempts = log.get("attempts", [])
            n = len(attempts)
            c = sum(1 for a in attempts if _attempt_succeeded_by_round(a, round_idx))
            if k > n:
                total_pass += 1.0 if c > 0 else 0.0
            else:
                numer = comb(n - c, k)
                denom = comb(n, k)
                pass_prob = 1.0 - (numer / denom) if denom > 0 else (1.0 if c > 0 else 0.0)
                total_pass += pass_prob
        pct = 100.0 * total_pass / n_problems
        pct_solved.append(pct)

    return ks, pct_solved


def plot_passk_by_round(
    problem_logs: list[dict],
    cfg: EvalConfig,
    run_dir: Path,
) -> Path:
    """
    Plot pass@k vs k with one line per correction round.

    Saves to run_dir/passk_by_round.png and prints values to console.
    """
    import matplotlib.pyplot as plt

    if not problem_logs:
        return run_dir / "passk_by_round.png"

    pass_k = cfg.pass_k
    num_rounds = cfg.num_correction_rounds
    rounds_to_plot = list(range(0, num_rounds + 1))

    fig, ax = plt.subplots(figsize=(8, 6))

    for round_idx in rounds_to_plot:
        ks, pct_solved = _compute_pass_at_k_for_round(problem_logs, round_idx, pass_k)
        if not ks:
            continue
        ax.plot(ks, pct_solved, marker="o", linestyle="-", markersize=6, label=f"round {round_idx}")
        print(f"  pass@k (round {round_idx}): " + ", ".join(f"k{k}={p:.1f}%" for k, p in zip(ks, pct_solved)))

    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_xticks(list(range(1, pass_k + 1)))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.legend()

    out_path = run_dir / "passk_by_round.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out_path}")
    return out_path
