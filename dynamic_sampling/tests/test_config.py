"""
Unit tests for dynamic_sampling.config (DynamicSamplingConfig validation and defaults).
"""

import pytest

from dynamic_sampling.config import DynamicSamplingConfig
from dynamic_sampling.constants import DEFAULT_BUDGET, MINIF2F_TEST_SIZE


def test_config_defaults():
    """Config with only budget uses sensible defaults."""
    cfg = DynamicSamplingConfig(budget=256)
    assert cfg.budget == 256
    assert cfg.model == "Qwen/Qwen3.5-4B"
    assert cfg.dataset_name == "cat-searcher/minif2f-lean4"
    assert cfg.dataset_split == "test"
    assert cfg.n_problems == 244
    assert cfg.kimina_url == "http://localhost:8000"
    assert cfg.output_dir == "dynamic_sampling_results"
    assert cfg.qwen_eval_module == "qwen_eval/modal_app.py"
    assert cfg.no_think_mode is True
    assert cfg.problem_index_file is None
    assert cfg.seed == 42


def test_config_custom_values():
    """Config accepts overrides."""
    cfg = DynamicSamplingConfig(
        budget=100,
        model="Qwen/Qwen3.5-9B",
        n_problems=50,
        output_dir="/tmp/out",
        no_think_mode=False,
        seed=0,
    )
    assert cfg.budget == 100
    assert cfg.model == "Qwen/Qwen3.5-9B"
    assert cfg.n_problems == 50
    assert cfg.output_dir == "/tmp/out"
    assert cfg.no_think_mode is False
    assert cfg.problem_index_file is None
    assert cfg.seed == 0


def test_config_problem_index_file():
    """problem_index_file can be set; default is None."""
    cfg = DynamicSamplingConfig(budget=10, problem_index_file="dynamic_sampling/problem_idx.json")
    assert cfg.problem_index_file == "dynamic_sampling/problem_idx.json"


def test_config_budget_validation():
    """Budget must be >= 1."""
    with pytest.raises(ValueError, match="budget must be >= 1"):
        DynamicSamplingConfig(budget=0)
    with pytest.raises(ValueError, match="budget must be >= 1"):
        DynamicSamplingConfig(budget=-1)
    DynamicSamplingConfig(budget=1)


def test_config_n_problems_validation():
    """n_problems must be >= 0."""
    with pytest.raises(ValueError, match="n_problems must be >= 0"):
        DynamicSamplingConfig(budget=10, n_problems=-1)
    DynamicSamplingConfig(budget=10, n_problems=0)
    DynamicSamplingConfig(budget=10, n_problems=1)


def test_constants_match_config_defaults():
    """Constants align with typical config usage."""
    assert MINIF2F_TEST_SIZE == 244
    assert DEFAULT_BUDGET == 256
    cfg = DynamicSamplingConfig(budget=DEFAULT_BUDGET)
    assert cfg.n_problems == MINIF2F_TEST_SIZE
