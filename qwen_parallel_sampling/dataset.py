"""
TLDR: Dataset loading and field extraction for the qwen_eval pipeline.

get_field() handles dataset-agnostic field access using priority lists from EvalConfig.
load_problem_indices_from_file() reads a JSON subset file (e.g. problem_idx.json).
load_problems() loads the HuggingFace dataset and returns a flat list of problem dicts.

Used by: modal_app.py (before building prompts).
"""

import json
from pathlib import Path
from typing import Any

from qwen_eval.config import EvalConfig


def load_problem_indices_from_file(path: str) -> list[int]:
    """
    Load problem indices from a JSON file.

    Accepts either:
      - "problem_indices": list of ints, or
      - "problems": list of dicts with "problem_idx" (order preserved).

    Raises:
        FileNotFoundError: path does not exist.
        json.JSONDecodeError: file is not valid JSON.
        ValueError: JSON has neither "problem_indices" nor "problems".
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Problem index file not found: {path}")
    raw = p.read_text()
    data = json.loads(raw)
    if "problem_indices" in data:
        return list(data["problem_indices"])
    if "problems" in data:
        return [int(item["problem_idx"]) for item in data["problems"]]
    raise ValueError(
        "JSON must contain 'problem_indices' or 'problems'; "
        f"got keys: {list(data.keys())}"
    )


def get_field(data: dict, field_names: list[str], default: str = "") -> str:
    """
    Return the first non-empty string value found in data using the priority list.

    Handles both plain string fields and list/tuple fields (joins with space).
    Falls back to `default` if no field matches.
    """
    for name in field_names:
        val = data.get(name)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            val = " ".join(str(v) for v in val if v)
        val = str(val).strip()
        if val:
            return val
    return default


def load_problems(cfg: EvalConfig) -> list[dict[str, Any]]:
    """
    Load the MiniF2F dataset and return a list of problem dicts.

    Each returned dict has normalized keys:
        problem_id, formal_statement, informal_stmt, header

    The first cfg.n_problems items from the split are used (in dataset order).
    If cfg.problem_indices is set, those specific indices are used instead.

    Returns:
        List of problem dicts, length == cfg.n_problems (or len(cfg.problem_indices)).

    Raises:
        RuntimeError if the dataset cannot be loaded.
    """
    from datasets import load_dataset as hf_load_dataset

    print(f"Loading dataset: {cfg.dataset_name} (split={cfg.dataset_split})")
    try:
        ds = hf_load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset '{cfg.dataset_name}' split '{cfg.dataset_split}': {e}"
        ) from e

    print(f"  Dataset size: {len(ds)} problems")

    if cfg.problem_indices is not None:
        indices = cfg.problem_indices
        print(f"  Using specified indices: {indices}")
    else:
        indices = list(range(min(cfg.n_problems, len(ds))))
        print(f"  Using first {len(indices)} problems")

    problems = []
    for idx in indices:
        row = ds[idx]
        problems.append({
            "problem_idx": idx,
            "problem_id": get_field(row, cfg.id_fields, default=str(idx)),
            "formal_statement": get_field(row, cfg.theorem_fields),
            "informal_stmt": get_field(row, cfg.informal_fields),
            "header": get_field(row, cfg.header_fields),
        })

    return problems
