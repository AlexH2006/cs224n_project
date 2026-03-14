"""
Pass@k analysis and plotting for SDPO pass@32 results.

TL;DR:
  - Discovers all run dirs under a pass@32 folder (e.g. results/.../pass@32).
  - For each run, loads runs/problem_*/logs.json (SDPO format: iteration_logs with
    samples per iteration). Computes the 1-based attempt index at which each problem
    first gets a successful verification.
  - Writes a summary JSON (attempts to first solution per problem) and a problem_idx
    JSON (indices of problems solved at least once) under the parent results folder.
  - Plots Pass@k vs k (with optional bootstrap error bars) matching the reference
    style (title "Pass@k (n=50 problems)", blue line + markers + vertical error bars).

Usage:
  python scripts/pass_at_k_sdpo_pass32.py results/Qwen3.5_4B_sampled_no_thinking_sdpo/pass@32
  python scripts/pass_at_k_sdpo_pass32.py results/Qwen3.5_4B_sampled_no_thinking_sdpo/pass@32 -o results/Qwen3.5_4B_sampled_no_thinking_sdpo/pass_at_k_50sample.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# Pass criterion (single attempt)
# -----------------------------------------------------------------------------


def is_pass(sample: dict[str, Any]) -> bool:
    """
    True iff this attempt counts as a pass: verification success, complete, no sorry.

    Uses verification.success, verification.complete, verification.has_sorry when
    present; otherwise falls back to top-level "success".
    """
    verification = sample.get("verification") or {}
    if not verification:
        return bool(sample.get("success"))
    return (
        verification.get("success", False)
        and verification.get("complete", False)
        and not verification.get("has_sorry", True)
    )


# -----------------------------------------------------------------------------
# SDPO log structure: iteration_logs[*].samples[*]
# -----------------------------------------------------------------------------


def get_attempts_from_problem_log(log_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Flatten SDPO logs into a single chronological list of attempts (one per sample).

    Each element is a sample dict (with verification/success). Order: iteration 1
    samples, then iteration 2 samples, etc.
    """
    attempts: list[dict[str, Any]] = []
    for entry in log_data.get("iteration_logs") or []:
        for sample in entry.get("samples") or []:
            attempts.append(sample)
    return attempts


def first_success_attempt(log_data: dict[str, Any]) -> int | None:
    """
    Return the 1-based attempt index at which the first pass occurs, or None if never.

    Attempts are ordered chronologically (iteration_logs order, then samples within each).
    """
    attempts = get_attempts_from_problem_log(log_data)
    for i, sample in enumerate(attempts):
        if is_pass(sample):
            return i + 1
    return None


# -----------------------------------------------------------------------------
# Discovery and loading
# -----------------------------------------------------------------------------


def discover_run_dirs(pass_at_32_dir: Path) -> list[Path]:
    """Return run directories (immediate children) that contain a 'runs' subdir."""
    if not pass_at_32_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {pass_at_32_dir}")
    run_dirs = []
    for child in sorted(pass_at_32_dir.iterdir()):
        if child.is_dir() and (child / "runs").is_dir():
            run_dirs.append(child)
    return run_dirs


def discover_problem_logs(run_dir: Path) -> list[Path]:
    """Return paths to runs/problem_*/logs.json in this run, sorted by problem number."""
    runs = run_dir / "runs"
    logs = []
    for sub in sorted(runs.iterdir(), key=lambda p: _problem_number(p.name)):
        if sub.is_dir():
            p = sub / "logs.json"
            if p.exists():
                logs.append(p)
    return logs


def _problem_number(problem_dir_name: str) -> int:
    """Extract numeric part from 'problem_26' -> 26."""
    try:
        return int(problem_dir_name.replace("problem_", ""))
    except ValueError:
        return -1


def load_problem_log(path: Path) -> dict[str, Any]:
    """Load a single problem logs.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_all_problems(pass_at_32_dir: Path) -> list[tuple[Path, dict[str, Any], int]]:
    """
    Collect (log_path, log_data, problem_idx) for every problem across all runs.

    problem_idx comes from log_data["problem"]["problem_idx"] when present,
    else from the directory name (problem_N).
    """
    run_dirs = discover_run_dirs(pass_at_32_dir)
    if not run_dirs:
        raise ValueError(f"No run directories with 'runs/' found under {pass_at_32_dir}")

    collected: list[tuple[Path, dict[str, Any], int]] = []
    for run_dir in run_dirs:
        for log_path in discover_problem_logs(run_dir):
            data = load_problem_log(log_path)
            idx = data.get("problem", {}).get("problem_idx")
            if idx is None:
                idx = _problem_number(log_path.parent.name)
            collected.append((log_path, data, idx))
    return collected


# -----------------------------------------------------------------------------
# Summary and problem_idx outputs
# -----------------------------------------------------------------------------


def build_summary_records(
    collected: list[tuple[Path, dict[str, Any], int]],
) -> list[dict[str, Any]]:
    """
    Build one record per problem: problem_idx, problem_id, attempts_to_first_success.
    """
    records = []
    for _path, data, idx in collected:
        problem_id = data.get("problem_id") or data.get("problem", {}).get("problem_id") or f"problem_{idx}"
        att = first_success_attempt(data)
        records.append({
            "problem_idx": idx,
            "problem_id": problem_id,
            "attempts_to_first_success": att,
        })
    return records


def build_summary_json(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Single summary object for writing to summary JSON."""
    by_idx = {r["problem_idx"]: r["attempts_to_first_success"] for r in records}
    by_id = {r["problem_id"]: r["attempts_to_first_success"] for r in records}
    n_solved = sum(1 for r in records if r["attempts_to_first_success"] is not None)
    return {
        "n_problems": len(records),
        "n_solved": n_solved,
        "by_problem_idx": by_idx,
        "by_problem_id": by_id,
        "records": records,
    }


def build_solved_problem_indices(records: list[dict[str, Any]]) -> list[int]:
    """Sorted list of problem indices that were solved at least once."""
    indices = [r["problem_idx"] for r in records if r["attempts_to_first_success"] is not None]
    return sorted(indices)


# -----------------------------------------------------------------------------
# Pass@k
# -----------------------------------------------------------------------------


def compute_pass_at_k(
    records: list[dict[str, Any]],
    max_k: int = 32,
) -> tuple[list[int], list[float]]:
    """
    For each k in 1..max_k, fraction (0–100) of problems with at least one pass
    in the first k attempts.

    Returns (ks, pct_list) where pct_list[i] = percentage at k = ks[i].
    """
    first_success = [r["attempts_to_first_success"] for r in records]
    n = len(first_success)
    ks = list(range(1, max_k + 1))
    pcts = []
    for k in ks:
        count = sum(1 for a in first_success if a is not None and a <= k)
        pcts.append(100.0 * count / n if n else 0.0)
    return ks, pcts


def bootstrap_pass_at_k(
    records: list[dict[str, Any]],
    max_k: int,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Bootstrap over problems to get mean and std of pass@k for each k.

    Returns (ks, mean, std) with arrays of length max_k.
    """
    rng = rng or np.random.default_rng()
    n = len(records)
    first_success = [r["attempts_to_first_success"] for r in records]
    ks = list(range(1, max_k + 1))
    samples = np.zeros((n_bootstrap, max_k))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        for i, k in enumerate(ks):
            count = sum(1 for j in idx if first_success[j] is not None and first_success[j] <= k)
            samples[b, i] = 100.0 * count / n
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    return ks, mean, std


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_pass_at_k(
    pass_at_32_dir: Path,
    output_path: Path | None = None,
    max_k: int = 32,
    n_problems: int = 50,
    use_bootstrap_errors: bool = False,
    n_bootstrap: int = 1000,
) -> Path:
    """
    Load data from pass_at_32_dir, compute pass@k, and save plot.

    If use_bootstrap_errors, error bars are bootstrap std over the 50 problems;
    otherwise no error bars. Output path defaults to parent of pass@32 / pass_at_k_50sample.png.
    """
    import matplotlib.pyplot as plt

    collected = collect_all_problems(pass_at_32_dir)
    records = build_summary_records(collected)
    n_actual = len(records)
    if n_problems is None:
        n_problems = n_actual

    if use_bootstrap_errors and n_actual >= 2:
        ks, mean, std = bootstrap_pass_at_k(records, max_k=max_k, n_bootstrap=n_bootstrap)
        fig, ax = plt.subplots()
        ax.errorbar(
            ks,
            mean,
            yerr=std,
            marker="o",
            linestyle="-",
            markersize=6,
            capsize=4,
            capthick=1,
            color="C0",
        )
    else:
        ks, pcts = compute_pass_at_k(records, max_k=max_k)
        fig, ax = plt.subplots()
        ax.plot(ks, pcts, marker="o", linestyle="-", markersize=6, color="C0")

    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title(f"Pass@k (n={n_problems} problems)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    if output_path is None:
        output_path = pass_at_32_dir.parent / "pass_at_k_50sample.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pass@k analysis and plot for SDPO pass@32 results (two runs, 50 problems combined).",
    )
    parser.add_argument(
        "pass_at_32_dir",
        type=Path,
        help="Directory containing run_*/runs/problem_*/logs.json (e.g. results/.../pass@32)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <parent of pass@32>/pass_at_k_50sample.png)",
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=32,
        help="Maximum k for pass@k (default: 32)",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=50,
        help="Number of problems for plot title (default: 50)",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Add bootstrap error bars on the plot",
    )
    args = parser.parse_args()

    pass_at_32_dir = args.pass_at_32_dir.resolve()
    if not pass_at_32_dir.is_dir():
        raise SystemExit(f"Not a directory: {pass_at_32_dir}")

    # Collect and write summary + problem_idx under parent of pass@32
    parent = pass_at_32_dir.parent
    collected = collect_all_problems(pass_at_32_dir)
    records = build_summary_records(collected)
    summary = build_summary_json(records)
    solved_indices = build_solved_problem_indices(records)

    summary_path = parent / "pass32_attempts_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary → {summary_path} (n_problems={summary['n_problems']}, n_solved={summary['n_solved']})")

    problem_idx_path = parent / "pass32_solved_problem_idx.json"
    with open(problem_idx_path, "w", encoding="utf-8") as f:
        json.dump(solved_indices, f, indent=2)
    print(f"Wrote solved problem indices → {problem_idx_path} (count={len(solved_indices)})")

    out = plot_pass_at_k(
        pass_at_32_dir,
        output_path=args.output,
        max_k=args.max_k,
        n_problems=args.n_problems,
        use_bootstrap_errors=args.bootstrap,
    )
    print(f"Saved plot → {out}")


if __name__ == "__main__":
    main()
