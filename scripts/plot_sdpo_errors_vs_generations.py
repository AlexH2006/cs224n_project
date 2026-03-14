#!/usr/bin/env python3
"""
Plot number of verification error codes vs generation index from an SDPO run logs.json.
Smooths the curve (rolling average). Saves plot next to logs.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_error_counts(logs_path: Path) -> list[int]:
    """
    Extract error count per generation (in order) from SDPO logs.
    One generation = one entry in iteration_logs[*].samples (same as discovery_time_summary).
    Does NOT count the top-level response per iteration; only the flattened samples.
    """
    with open(logs_path, encoding="utf-8") as f:
        data = json.load(f)
    counts = []
    for it_log in data.get("iteration_logs", []):
        for s in it_log.get("samples", []):
            v = s.get("verification") or {}
            counts.append(len(v.get("errors") or []))
    return counts


def smooth(y: list[float], window: int) -> np.ndarray:
    """Rolling mean; same length as y, valid only in the middle."""
    if window < 1 or len(y) < window:
        return np.asarray(y, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    out = np.full_like(y_arr, np.nan)
    half = window // 2
    for i in range(half, len(y_arr) - (window - half - 1)):
        out[i] = np.mean(y_arr[i - half : i + (window - half)])
    # Fill edges: use partial windows
    for i in range(half):
        out[i] = np.mean(y_arr[: i + half + 1])
    for i in range(len(y_arr) - (window - half - 1), len(y_arr)):
        out[i] = np.mean(y_arr[i - half :])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot # errors vs # generations from SDPO logs.json (smoothed)"
    )
    parser.add_argument(
        "logs_json",
        type=Path,
        help="Path to logs.json (e.g. .../run_Qwen3.5-4B_105_.../logs.json)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output figure path (default: same dir as logs, errors_vs_generations.png)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Smoothing window size (default: 10)",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not plot raw error counts, only smoothed",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="If set, only plot the first N generations (e.g. 84 if run stopped or succeeded at 84)",
    )
    args = parser.parse_args()

    logs_path = Path(args.logs_json).resolve()
    if not logs_path.is_file():
        raise SystemExit(f"Not a file: {logs_path}")

    counts = load_error_counts(logs_path)
    if not counts:
        raise SystemExit("No generations found in logs.")
    if args.max_generations is not None:
        counts = counts[: args.max_generations]
        if not counts:
            raise SystemExit("No generations left after --max-generations.")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required: pip install matplotlib")

    x = np.arange(1, len(counts) + 1, dtype=float)
    y_raw = np.asarray(counts, dtype=float)
    y_smooth = smooth(counts, max(1, args.window))

    fig, ax = plt.subplots(figsize=(10, 5))
    if not args.no_raw:
        ax.plot(x, y_raw, alpha=0.35, color="tab:blue", linewidth=0.8, label="Raw")
    ax.plot(x, y_smooth, color="tab:blue", linewidth=1.5, label=f"Smoothed (window={args.window})")
    ax.set_xlabel("Generation (#)")
    ax.set_ylabel("Number of error codes")
    title = f"Verification errors vs generation\n{logs_path.parent.name}"
    if args.max_generations is not None:
        title += f" (first {len(counts)} generations)"
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=1)
    fig.tight_layout()

    out_path = Path(args.output).resolve() if args.output else logs_path.parent / "errors_vs_generations.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path} ({len(counts)} generations)", flush=True)


if __name__ == "__main__":
    main()
