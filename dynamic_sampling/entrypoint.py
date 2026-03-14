"""
TLDR: CLI entrypoint for dynamic_sampling (argparse, build config, run runner).

Usage:
    python -m dynamic_sampling.entrypoint --budget 256
    python -m dynamic_sampling.entrypoint --budget 512 --model Qwen/Qwen3.5-9B --output-dir my_results
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from dynamic_sampling.config import DynamicSamplingConfig
from dynamic_sampling.constants import DEFAULT_BUDGET, MINIF2F_TEST_SIZE
from dynamic_sampling.runner import run


def _model_safe(model_name: str) -> str:
    """'Qwen/Qwen3.5-4B' -> 'Qwen3.5-4B' for directory names."""
    return model_name.split("/")[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic sampling: multi-round MiniF2F eval with a total attempt budget.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help=f"Max total attempts across all problems (default: {DEFAULT_BUDGET})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Model name for qwen_eval",
    )
    parser.add_argument(
        "--n-problems",
        type=int,
        default=MINIF2F_TEST_SIZE,
        help=f"Number of problems to evaluate (default: {MINIF2F_TEST_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dynamic_sampling_results",
        help="Base output directory (default: dynamic_sampling_results)",
    )
    parser.add_argument(
        "--use-thinking",
        action="store_true",
        help="Enable Qwen <think>...</think> (default: disabled, i.e. --no-think-mode in qwen_eval)",
    )
    parser.add_argument(
        "--kimina-url",
        type=str,
        default="http://localhost:8000",
        help="Kimina Lean server URL",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repository root for subprocess cwd (default: current directory)",
    )
    parser.add_argument(
        "--problem-index-file",
        type=str,
        default=None,
        help="Path to JSON file with problem_indices for first round (e.g. dynamic_sampling/problem_idx.json). If set, only these problems are evaluated.",
    )
    args = parser.parse_args()

    config = DynamicSamplingConfig(
        budget=args.budget,
        model=args.model,
        n_problems=args.n_problems,
        output_dir=args.output_dir,
        problem_index_file=args.problem_index_file,
        no_think_mode=not args.use_thinking,
        kimina_url=args.kimina_url,
        seed=args.seed,
    )
    repo_root = Path(args.repo_root) if args.repo_root else None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(config.output_dir) / f"run_{_model_safe(config.model)}_{ts}"
    out_path = run(config, output_dir=run_output_dir, repo_root=repo_root)
    print(f"Results written to: {out_path}")
    print(f"  summary:   {out_path / 'summary.json'}")
    print(f"  raw_logs:  {out_path / 'raw_logs.json'}")


if __name__ == "__main__":
    main()
