"""
Unit tests for qwen_multiturn/config.py.

Validates EvalConfig defaults and that new fields (e.g. inference_batch_size)
are present and have expected values. No Modal, no GPU.

Run with:
    python -m pytest qwen_multiturn/tests/test_config.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_multiturn.config import EvalConfig


class TestEvalConfig:
    """EvalConfig default values and optional fields."""

    def test_inference_batch_size_default(self):
        cfg = EvalConfig()
        assert hasattr(cfg, "inference_batch_size")
        assert cfg.inference_batch_size == 256

    def test_inference_batch_size_override(self):
        cfg = EvalConfig(inference_batch_size=64)
        assert cfg.inference_batch_size == 64

    def test_inference_batch_size_none_allowed(self):
        cfg = EvalConfig(inference_batch_size=None)
        assert cfg.inference_batch_size is None
