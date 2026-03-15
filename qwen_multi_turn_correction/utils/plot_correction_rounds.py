"""
Plot cumulative percentage of problems solved by correction round.

X-axis: correction round index (0 = before correction, 1 = after one correction, etc.).
Y-axis: cumulative percentage of problems with at least one sample solved by that round.

Saves correction_rounds_performance.png and correction_rounds_data.json in run_dir.
Used by modal_app.run_eval when correction_rounds > 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_eval.results import compute_cumulative_solved_by_round


def plot_correction_rounds_performance(
    problem_logs: list[dict[str, Any]],
    correction_rounds: int,
    run_dir: Path,
) -> None:
    """
    Compute cumulative solved-by-round metrics, save JSON and PNG to run_dir.
    """
    if not problem_logs or correction_rounds < 0:
        return
    round_indices, accuracy_percent = compute_cumulative_solved_by_round(
        problem_logs, correction_rounds
    )
    if not round_indices:
        return

    data = {"rounds": round_indices, "accuracy_percent": accuracy_percent}
    data_path = run_dir / "correction_rounds_data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved correction rounds data → {data_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(round_indices, accuracy_percent, marker="o", linestyle="-", markersize=6)
    ax.set_xlabel("Correction round (pass@number)")
    ax.set_ylabel("Solved by round r (%)")
    ax.set_title("Cumulative problems solved by correction round")
    ax.set_xticks(round_indices)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    png_path = run_dir / "correction_rounds_performance.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved correction rounds plot → {png_path}")
