#!/usr/bin/env python3
"""
Compare attempts used per problem between an SDPO run and the baseline (no-thinking)
run, matched by problem_idx. Outputs a table and a pass@k vs k plot (baseline vs SDPO).

Definitions:
- Baseline: N attempts per problem (N = --max-baseline-attempts, default 8); "attempt at first success"
  = 1..N or N+1 if never solved within first N attempts.
- SDPO: 1--8 iterations, each with --sdpo-attempts-per-iteration attempts (default 4), so up to 32 attempts.
  "First success attempt" = (iteration - 1) * attempts_per_iteration + 1, or N+1 if never.

Pass@k: fraction of problems solved within the first k attempts (same k for both; use N for pass@N).

SDPO-only wins: problems that SDPO solved but baseline did not within the first N attempts.

Baseline can be a single logs.json (list of problems with attempts) or a directory of subruns:
  - Single file: one logs.json with N attempts per problem.
  - Directory (e.g. results/Qwen3.5_4B_sampled_no_thinking): each subrun has logs.json with 8
    attempts per problem; 4 subruns combined = 32 attempts (pass@32).

Usage (from project root):
    python scripts/compare_sdpo_baseline_attempts.py \\
        --sdpo-runs-dir sdpo_results/Qwen3.5-4B/run_Qwen3.5-4B_20260311_111404/runs \\
        --baseline-logs results/run_Qwen3.5-4B_no_think_mode/logs.json

    # Pass@32: baseline from directory of 4 subruns (8 attempts each = 32), SDPO 4 attempts/iter
    ... --baseline-logs results/Qwen3.5_4B_sampled_no_thinking --max-baseline-attempts 32 --sdpo-attempts-per-iteration 4

    # Optional: explicit output paths
    ... --output results/compare_table.json --plot results/pass_at_k_sdpo_vs_baseline.png

    # Write SDPO-only wins (problem_idx, problem_id) to JSON
    ... --output-wins results/sdpo_only_wins.json

    # Or: output directory + prefix (table and plot written there)
    ... --output-dir results/sdpo_vs_baseline_comparison --output-prefix compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_sdpo_run(runs_dir: Path) -> dict[int, dict]:
    """
    Load per-problem data from SDPO runs/problem_*/logs.json.
    Returns dict: problem_idx -> { problem_id, first_success_at (1-8 or 9), solved, attempts_used }
    """
    runs_dir = Path(runs_dir)
    out = {}
    for path in sorted(runs_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("problem_"):
            continue
        logs_path = path / "logs.json"
        if not logs_path.exists():
            continue
        with open(logs_path, encoding="utf-8") as f:
            logs = json.load(f)
        problem = logs.get("problem", {}) or {}
        problem_idx = problem.get("problem_idx")
        if problem_idx is None:
            try:
                problem_idx = int(path.name.replace("problem_", ""))
            except ValueError:
                continue
        iteration_logs = logs.get("iteration_logs", [])
        attempts_used = len(iteration_logs)
        first_success_at = 9
        for i, it in enumerate(iteration_logs):
            if it.get("success"):
                first_success_at = i + 1
                break
        solved = logs.get("success", False)
        out[problem_idx] = {
            "problem_id": problem.get("problem_id", ""),
            "first_success_at": first_success_at,
            "solved": solved,
            "attempts_used": attempts_used,
        }
    return out


def is_pass(attempt: dict) -> bool:
    """
    True iff this attempt counts as a pass: success, complete, and no sorry.
    Matches scripts/plot_pass_at_k_multi_run.py.
    """
    verification = attempt.get("verification") or {}
    if not verification:
        return bool(attempt.get("success"))
    return (
        verification.get("success", False)
        and verification.get("complete", False)
        and not verification.get("has_sorry", True)
    )


def load_baseline_by_problem_indices(
    logs_path: Path,
    problem_indices: set[int],
    max_attempts: int | None = None,
) -> dict[int, dict]:
    """
    Load baseline logs.json and filter by problem_idx.
    If max_attempts is set, only the first max_attempts attempts are considered;
    first_success_attempt is then 1..max_attempts or max_attempts+1 if never.
    Returns dict: problem_idx -> { problem_id, first_success_attempt, n_success, success_rate }
    """
    with open(logs_path, encoding="utf-8") as f:
        all_logs = json.load(f)
    out = {}
    for log in all_logs:
        problem = log.get("problem", {}) or {}
        problem_idx = problem.get("problem_idx")
        if problem_idx is None:
            continue
        if problem_idx not in problem_indices:
            continue
        attempts = log.get("attempts", [])
        if max_attempts is not None:
            attempts = attempts[:max_attempts]
        sentinel = (max_attempts + 1) if max_attempts is not None else 9
        first_success_attempt = sentinel
        for i, a in enumerate(attempts):
            if is_pass(a):
                first_success_attempt = i + 1
                break
        n_success = sum(1 for a in attempts if is_pass(a))
        n_attempts = len(attempts)
        success_rate = round(n_success / n_attempts, 4) if n_attempts else 0.0
        out[problem_idx] = {
            "problem_id": problem.get("id", problem.get("problem_id", "")),
            "first_success_attempt": first_success_attempt,
            "n_success": n_success,
            "success_rate": success_rate,
        }
    return out


def load_baseline_from_multi_run_dir(
    parent_dir: Path,
    problem_indices: set[int],
    max_attempts: int = 32,
) -> dict[int, dict]:
    """
    Load baseline from a directory of subruns (each subrun has logs.json with 8 attempts per problem).
    Combine attempts in subrun order: run1 attempts + run2 attempts + ... then take first max_attempts.
    Returns dict: problem_idx -> { problem_id, first_success_attempt, n_success, success_rate }
    """
    run_dirs = sorted(
        d for d in parent_dir.iterdir()
        if d.is_dir() and (d / "logs.json").exists()
    )
    if not run_dirs:
        return {}

    # Build per-problem combined attempts across runs (in run order)
    # problem_idx -> list of attempt dicts (up to max_attempts)
    combined: dict[int, list[dict]] = {idx: [] for idx in problem_indices}
    problem_id_by_idx: dict[int, str] = {}

    for run_dir in run_dirs:
        with open(run_dir / "logs.json", encoding="utf-8") as f:
            all_logs = json.load(f)
        for log in all_logs:
            problem = log.get("problem", {}) or {}
            problem_idx = problem.get("problem_idx")
            if problem_idx is None or problem_idx not in problem_indices:
                continue
            problem_id_by_idx[problem_idx] = problem.get("id", problem.get("problem_id", ""))
            attempts = log.get("attempts", [])
            combined[problem_idx].extend(attempts)

    out = {}
    sentinel = max_attempts + 1
    for problem_idx in problem_indices:
        attempts = combined.get(problem_idx, [])[:max_attempts]
        first_success_attempt = sentinel
        for i, a in enumerate(attempts):
            if is_pass(a):
                first_success_attempt = i + 1
                break
        n_success = sum(1 for a in attempts if is_pass(a))
        n_attempts = len(attempts)
        success_rate = round(n_success / n_attempts, 4) if n_attempts else 0.0
        out[problem_idx] = {
            "problem_id": problem_id_by_idx.get(problem_idx, ""),
            "first_success_attempt": first_success_attempt,
            "n_success": n_success,
            "success_rate": success_rate,
        }
    return out


def compute_pass_at_k(first_success_values: list[int], max_k: int = 8) -> tuple[list[int], list[float]]:
    """
    first_success_values: per-problem attempt at first success (1-8 or 9 if never).
    Returns (ks, pct_solved): k in 1..max_k, and fraction (0-100) of problems solved within k.
    """
    n = len(first_success_values)
    if n == 0:
        return [], []
    ks = list(range(1, max_k + 1))
    pct = [100.0 * sum(1 for v in first_success_values if v <= k) / n for k in ks]
    return ks, pct


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SDPO vs baseline attempts by problem_idx, output table and pass@k plot"
    )
    parser.add_argument(
        "--sdpo-runs-dir",
        type=Path,
        required=True,
        help="Path to SDPO run's runs/ directory (e.g. sdpo_results/.../run_*/runs)",
    )
    parser.add_argument(
        "--baseline-logs",
        type=Path,
        required=True,
        help="Path to baseline: a logs.json file, or a directory of subruns (each with logs.json, 8 attempts; 4 subruns = pass@32).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for comparison table (JSON) and pass_at_k plot (PNG). Default: same as sdpo-runs-dir parent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for table output (JSON or CSV if path ends with .csv). Overrides output-dir/prefix for table.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path for pass@k vs k figure (PNG). Overrides output-dir/prefix for plot.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="compare",
        help="Prefix for output files when using --output-dir (default: compare)",
    )
    parser.add_argument(
        "--max-baseline-attempts",
        type=int,
        default=8,
        help="Number of baseline attempts to consider per problem (default: 8). Use 32 for 32-attempt baseline.",
    )
    parser.add_argument(
        "--output-wins",
        type=Path,
        default=None,
        help="Optional path to write SDPO-only wins (problem_idx, problem_id) as JSON.",
    )
    parser.add_argument(
        "--sdpo-attempts-per-iteration",
        type=int,
        default=4,
        help="Attempts per SDPO iteration (minibatch size). Used to convert iterations to attempt index for pass@k (default: 4).",
    )
    args = parser.parse_args()

    sdpo_dir = args.sdpo_runs_dir
    baseline_path = args.baseline_logs
    if not sdpo_dir.is_dir():
        raise SystemExit(f"SDPO runs dir not found: {sdpo_dir}")
    if not baseline_path.exists():
        raise SystemExit(f"Baseline path not found: {baseline_path}")
    baseline_is_dir = baseline_path.is_dir()

    output_dir = args.output_dir or (sdpo_dir.parent if sdpo_dir.name == "runs" else sdpo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix

    max_attempts = args.max_baseline_attempts
    attempts_per_iter = args.sdpo_attempts_per_iteration

    # Load both sources
    sdpo = load_sdpo_run(sdpo_dir)
    if not sdpo:
        raise SystemExit(f"No SDPO problem logs found under {sdpo_dir}")
    if baseline_is_dir:
        baseline = load_baseline_from_multi_run_dir(
            baseline_path, set(sdpo.keys()), max_attempts=max_attempts
        )
    else:
        baseline = load_baseline_by_problem_indices(
            baseline_path, set(sdpo.keys()), max_attempts=max_attempts
        )

    # Build merged rows (only problem_indices present in SDPO)
    rows = []
    for idx in sorted(sdpo.keys()):
        s = sdpo[idx]
        b = baseline.get(idx)
        row = {
            "problem_idx": idx,
            "problem_id": s["problem_id"],
            "baseline_first_success": b["first_success_attempt"] if b else None,
            "baseline_success_rate": b["success_rate"] if b else None,
            "baseline_n_success": b["n_success"] if b else None,
            "sdpo_first_success": s["first_success_at"] if s["solved"] else None,
            "sdpo_attempts_used": s["attempts_used"],
            "sdpo_solved": s["solved"],
        }
        rows.append(row)

    # SDPO-only wins: solved by SDPO but not by baseline within first max_attempts
    sdpo_only_wins = [
        r
        for r in rows
        if r["sdpo_solved"]
        and (
            r["baseline_first_success"] is None
            or r["baseline_first_success"] > max_attempts
        )
    ]
    print(f"SDPO-only wins (solved by SDPO, not by baseline in first {max_attempts} attempts): {len(sdpo_only_wins)}")
    if sdpo_only_wins:
        for r in sdpo_only_wins:
            print(f"  {r['problem_idx']}  {r['problem_id']}")
    else:
        print("  (none)")
    if args.output_wins is not None:
        out_wins = Path(args.output_wins)
        out_wins.parent.mkdir(parents=True, exist_ok=True)
        wins_data = {
            "max_baseline_attempts": max_attempts,
            "n_sdpo_only_wins": len(sdpo_only_wins),
            "problems": [
                {"problem_idx": r["problem_idx"], "problem_id": r["problem_id"]}
                for r in sdpo_only_wins
            ],
        }
        with open(out_wins, "w", encoding="utf-8") as f:
            json.dump(wins_data, f, indent=2)
        print(f"Saved SDPO-only wins → {out_wins}")
    print()

    # Table output
    print(f"Comparison (n={len(rows)} problems, matched by problem_idx)\n")
    print(f"{'idx':<5} {'problem_id':<35} {'base_1st':<9} {'base_rate':<10} {'sdpo_1st':<9} {'sdpo_used':<10} {'sdpo_ok'}")
    print("-" * 95)
    for r in rows:
        base_1st = r["baseline_first_success"] if r["baseline_first_success"] is not None else "—"
        base_rate = f"{r['baseline_success_rate']:.2f}" if r["baseline_success_rate"] is not None else "—"
        sdpo_1st = r["sdpo_first_success"] if r["sdpo_first_success"] is not None else "—"
        print(
            f"{r['problem_idx']:<5} {r['problem_id']:<35} {str(base_1st):<9} {base_rate:<10} "
            f"{str(sdpo_1st):<9} {r['sdpo_attempts_used']:<10} {r['sdpo_solved']}"
        )

    table_out = args.output
    if table_out is not None:
        table_out = Path(table_out)
        table_out.parent.mkdir(parents=True, exist_ok=True)
        if str(table_out).lower().endswith(".csv"):
            import csv
            with open(table_out, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["problem_idx", "problem_id", "baseline_first_success", "baseline_success_rate", "sdpo_attempts_used", "sdpo_solved"])
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k) for k in w.fieldnames})
            print(f"\nSaved table (CSV) → {table_out}")
        else:
            with open(table_out, "w", encoding="utf-8") as f:
                json.dump({"n_problems": len(rows), "rows": rows}, f, indent=2)
            print(f"\nSaved table (JSON) → {table_out}")
    else:
        out_json = output_dir / f"{prefix}_table.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"n_problems": len(rows), "rows": rows}, f, indent=2)
        print(f"\nSaved table → {out_json}")

    # Pass@k: use only problems that appear in both (for fair comparison)
    # max_k = max_attempts (e.g. 32 for pass@32). SDPO iteration -> attempt: (iter-1)*attempts_per_iter+1
    common_idx = [r["problem_idx"] for r in rows if r["baseline_first_success"] is not None]
    if not common_idx:
        print("No problems in both baseline and SDPO; skipping pass@k.")
        return
    max_k = max_attempts
    baseline_first = [
        min(baseline[i]["first_success_attempt"], max_k + 1) for i in common_idx
    ]
    # Convert SDPO first_success_at (iteration 1..8) to attempt index (1..32)
    sdpo_first_attempt = []
    for i in common_idx:
        s = sdpo[i]
        if s["solved"]:
            # first success at iteration first_success_at -> attempt (iter-1)*4+1
            first_iter = s["first_success_at"]
            sdpo_first_attempt.append((first_iter - 1) * attempts_per_iter + 1)
        else:
            sdpo_first_attempt.append(max_k + 1)

    ks, baseline_pct = compute_pass_at_k(baseline_first, max_k=max_k)
    _, sdpo_pct = compute_pass_at_k(sdpo_first_attempt, max_k=max_k)

    # Plot pass@k vs k
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(ks, baseline_pct, marker="o", linestyle="-", markersize=6, label="Baseline (no thinking)")
    ax.plot(ks, sdpo_pct, marker="s", linestyle="-", markersize=6, label="SDPO")
    ax.set_xlabel("k (number of attempts)")
    ax.set_ylabel("Problems solved (%)")
    ax.set_title(f"Pass@{max_k} (n={len(common_idx)} problems, {attempts_per_iter} attempts/iter for SDPO)")
    if max_k > 12:
        tick_vals = list(range(1, max_k + 1, 4))
        if max_k not in tick_vals:
            tick_vals.append(max_k)
        ax.set_xticks(tick_vals)
    else:
        ax.set_xticks(ks)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    plot_path = args.plot
    if plot_path is None:
        plot_path = output_dir / f"{prefix}_pass_at_k.png"
    else:
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {plot_path}")


if __name__ == "__main__":
    main()
