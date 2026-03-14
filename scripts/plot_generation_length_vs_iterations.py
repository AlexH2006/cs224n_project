#!/usr/bin/env python3
"""
From SDPO logs under results/Qwen3.5_4B_discovery_time/Qwen3.5-4B (full_output runs),
plot token length vs generation index (1, 2, 3, ...) for failed generations only.
X-axis = number of generations (not iteration number). When multiple runs have the
same generation index, average the token length. Output: results/Qwen3.5_4B_discovery_time/generation_length_vs_iterations.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def is_failed(sample: dict) -> bool:
    """True if this generation did not succeed (verification failed or truncated)."""
    if sample.get("success") is True:
        return False
    v = sample.get("verification") or {}
    return not v.get("success", False)


def get_problem_id(logs_path: Path) -> str:
    """Read problem_id from logs.json; return empty string if missing."""
    with open(logs_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("problem_id") or data.get("problem", {}).get("problem_id") or ""


def collect_generation_lengths(logs_path: Path) -> list[int]:
    """From one logs.json, return token lengths for failed generations only (1st, 2nd, ...)."""
    with open(logs_path, encoding="utf-8") as f:
        data = json.load(f)
    lengths: list[int] = []
    for it_log in data.get("iteration_logs", []):
        for sample in it_log.get("samples", []):
            if not is_failed(sample):
                continue
            n = sample.get("num_tokens")
            if n is not None:
                lengths.append(int(n))
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot token length vs generation index (failed only).")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="If set, use only this run's logs.json and save plot in this run directory.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=15,
        metavar="WINDOW",
        help="Rolling average window size for smoothing (0 = no smoothing). Default: 15.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, save the plot to this path (overrides default output path).",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    base = repo / "results" / "Qwen3.5_4B_discovery_time" / "Qwen3.5-4B"
    out_dir = repo / "results" / "Qwen3.5_4B_discovery_time"
    out_path = out_dir / "generation_length_vs_iterations.png"

    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        logs_path = run_dir / "logs.json" if run_dir.is_dir() else run_dir
        if not logs_path.is_file():
            raise SystemExit(f"Logs not found: {logs_path}")
        logs_files = [logs_path]
        out_dir = run_dir if run_dir.is_dir() else run_dir.parent
        out_path = out_dir / "generation_length_vs_iterations.png"
    if args.output is not None:
        out_path = Path(args.output).resolve()
        out_dir = out_path.parent
    if args.run_dir is None:
        logs_files = list(base.rglob("logs.json"))
        if not logs_files:
            raise SystemExit(f"No logs.json found under {base}")

    # per run: list of token lengths by generation index (0 = 1st generation)
    all_lengths: list[list[int]] = []
    for p in logs_files:
        lengths = collect_generation_lengths(p)
        if lengths:
            all_lengths.append(lengths)

    if not all_lengths:
        raise SystemExit("No generation token lengths found in any log.")

    max_gen = max(len(r) for r in all_lengths)
    # For each generation index k (1-based), average token length over runs that have that generation
    gen_indices: list[int] = []
    avg_lengths: list[float] = []
    for k in range(1, max_gen + 1):
        vals = [r[k - 1] for r in all_lengths if len(r) >= k]
        if not vals:
            continue
        gen_indices.append(k)
        avg_lengths.append(sum(vals) / len(vals))

    # Optional smoothing: centered rolling mean
    plot_lengths = avg_lengths
    if args.smooth > 0 and len(avg_lengths) >= args.smooth:
        kernel = np.ones(args.smooth) / args.smooth
        plot_lengths = np.convolve(avg_lengths, kernel, mode="same").tolist()

    fig, ax = plt.subplots()
    ax.plot(gen_indices, plot_lengths, marker="o", markersize=3, linestyle="-")
    ax.set_xlabel("Number of generations")
    ylabel = "Token length" if len(all_lengths) == 1 else "Average token length"
    ax.set_ylabel(ylabel)
    if len(all_lengths) == 1 and logs_files:
        problem_id = get_problem_id(logs_files[0])
        title = f"Generation Length vs Number of Generations ({problem_id})" if problem_id else "Generation Length vs Number of Generations"
    else:
        title = "Generation Length vs Number of Generations"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path} (n_runs={len(all_lengths)}, max_generations={max_gen})", flush=True)


if __name__ == "__main__":
    main()
