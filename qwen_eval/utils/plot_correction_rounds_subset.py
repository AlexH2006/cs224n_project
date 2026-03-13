"""
Plot cumulative problems solved by correction round for a subset of problem indices.

Usage:
  python -m qwen_eval.utils.plot_correction_rounds_subset RUN_DIR [PROBLEM_INDICES_JSON]

If PROBLEM_INDICES_JSON is omitted, uses the default 50-problem subset (see SUBSET_INDICES below).
Output: correction_rounds_data_subset.json, correction_rounds_performance_subset.png (in RUN_DIR).
Does not modify or delete any existing files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Default subset of 50 problem indices (for run_Qwen3.5-4B_20260312_004723)
SUBSET_INDICES = [
    7, 9, 10, 18, 19, 20, 21, 26, 32, 34, 43, 46, 48, 52, 53, 55, 56, 57,
    65, 68, 87, 89, 95, 97, 100, 108, 110, 115, 116, 117, 119, 122, 127,
    128, 132, 140, 146, 147, 155, 157, 159, 174, 183, 187, 202, 227, 228,
    231, 239, 241,
]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m qwen_eval.utils.plot_correction_rounds_subset RUN_DIR [PROBLEM_INDICES_JSON]")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        with open(sys.argv[2], encoding="utf-8") as f:
            problem_indices = json.load(f)
    else:
        problem_indices = SUBSET_INDICES

    logs_path = run_dir / "logs.json"
    summary_path = run_dir / "summary.json"
    if not logs_path.exists():
        print(f"Missing {logs_path}")
        sys.exit(1)

    with open(logs_path, encoding="utf-8") as f:
        problem_logs = json.load(f)
    subset_set = set(problem_indices)
    filtered = [log for log in problem_logs if (log.get("problem") or {}).get("problem_idx") in subset_set]

    if not filtered:
        print("No matching problems in logs for the given indices.")
        sys.exit(1)

    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        correction_rounds = summary.get("correction_rounds", 0)
    else:
        correction_rounds = 0
        for log in filtered:
            for att in log.get("attempts", []):
                for r in (att.get("rounds") or []):
                    correction_rounds = max(correction_rounds, r.get("round", 0))

    from qwen_eval.results import compute_cumulative_solved_by_round

    round_indices, accuracy_percent = compute_cumulative_solved_by_round(filtered, correction_rounds)
    if not round_indices:
        print("No rounds to plot.")
        sys.exit(0)

    data = {"rounds": round_indices, "accuracy_percent": accuracy_percent, "n_problems": len(filtered)}
    data_path = run_dir / "correction_rounds_data_subset.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved subset correction rounds data → {data_path} (n={len(filtered)})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(round_indices, accuracy_percent, marker="o", linestyle="-", markersize=6)
    ax.set_xlabel("Correction round (pass@number)")
    ax.set_ylabel("Solved by round r (%)")
    ax.set_title(f"Cumulative problems solved by correction round (subset, n={len(filtered)})")
    ax.set_xticks(round_indices)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    png_path = run_dir / "correction_rounds_performance_subset.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved subset correction rounds plot → {png_path}")


if __name__ == "__main__":
    main()
