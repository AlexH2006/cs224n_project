"""
TLDR: Single dataclass for all dynamic_sampling run parameters (budget, model, paths).

No I/O; used by runner, invoker, and entrypoint. All paths are strings; caller resolves to Path if needed.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DynamicSamplingConfig:
    """Configuration for a single dynamic_sampling run."""

    # Max total attempts (generations) across all problems. Process stops when reached or no problems left.
    budget: int

    # Model name passed to qwen_eval (e.g. "Qwen/Qwen3.5-4B").
    model: str = "Qwen/Qwen3.5-4B"

    # Dataset (same as qwen_eval).
    dataset_name: str = "cat-searcher/minif2f-lean4"
    dataset_split: str = "test"

    # Number of problems to consider (default: all 244). Used when problem_index_file is not set.
    n_problems: int = 244

    # Optional: path to JSON file with {"problem_indices": [0, 1, ...]} for first-round problems. If set, only these are evaluated.
    problem_index_file: Optional[str] = None

    # Kimina URL for verification (passed through to qwen_eval).
    kimina_url: str = "http://localhost:8000"

    # Output root for this run: summary.json and raw_logs.json written here.
    # Subdirs per round (e.g. round_0, round_1) can be created under a parent run dir.
    output_dir: str = "dynamic_sampling_results"

    # Path to qwen_eval modal app for subprocess. Must be a .py file path so "modal run" accepts it.
    qwen_eval_module: str = "qwen_eval/modal_app.py"

    # Optional: disable <think>...</think> in Qwen (--no-think-mode). Default True = thinking disabled.
    no_think_mode: bool = True

    # Optional: seed for reproducibility (passed to qwen_eval).
    seed: int = 42

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError(f"budget must be >= 1, got {self.budget}")
        if self.n_problems < 0:
            raise ValueError(f"n_problems must be >= 0, got {self.n_problems}")
