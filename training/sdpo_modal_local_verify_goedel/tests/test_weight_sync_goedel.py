"""
CPU-only unit tests for sdpo_modal_local_verify_goedel/_weight_sync_goedel.py.

Verifies weight tying: sync_lora_weights_to_vllm_goedel builds the correct weight_dict
and calls collective_rpc("update_weights_from_numpy", ...) so vLLM and HF stay in sync.
Tests name translation (HF PEFT -> vLLM model.layers.*, no language_model. prefix),
unfused keys, and merged weight consistency. No GPU, vLLM, or bitsandbytes required.

Run with:
    python -m pytest sdpo_modal_local_verify_goedel/tests/test_weight_sync_goedel.py -v
    # or:
    python sdpo_modal_local_verify_goedel/tests/test_weight_sync_goedel.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path when run directly.
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sdpo_modal_local_verify_goedel._weight_sync_goedel import (
    GOEDEL_LORA_TARGET_MODULES,
    _hf_to_vllm_name_goedel,
    sync_lora_weights_to_vllm_goedel,
)


# ---------------------------------------------------------------------------
# Tests: _hf_to_vllm_name_goedel (Goedel uses model.layers.*, no language_model.)
# ---------------------------------------------------------------------------

def test_name_translation_q_proj():
    hf = "base_model.model.model.layers.0.self_attn.q_proj"
    expected = "model.layers.0.self_attn.q_proj.weight"
    assert _hf_to_vllm_name_goedel(hf) == expected


def test_name_translation_mlp():
    hf = "base_model.model.model.layers.3.mlp.gate_proj"
    expected = "model.layers.3.mlp.gate_proj.weight"
    assert _hf_to_vllm_name_goedel(hf) == expected


def test_name_translation_no_language_model_prefix():
    """Goedel vLLM names must NOT start with 'language_model.' (that is Qwen3.5-only)."""
    hf = "base_model.model.model.layers.0.self_attn.q_proj"
    result = _hf_to_vllm_name_goedel(hf)
    assert not result.startswith("language_model."), (
        f"Goedel vLLM names should not have language_model. prefix; got {result!r}"
    )


def test_name_translation_ends_with_weight():
    hf = "base_model.model.model.layers.27.mlp.down_proj"
    result = _hf_to_vllm_name_goedel(hf)
    assert result.endswith(".weight"), f"vLLM param name must end with .weight; got {result!r}"


# ---------------------------------------------------------------------------
# Mock model and vLLM engine for sync end-to-end tests (weight tying)
# ---------------------------------------------------------------------------

class _MockLoraLinear:
    """Minimal mock of a PEFT LoRA-wrapped linear (fp32 base, no quantization)."""

    def __init__(self, out_features: int, in_features: int, r: int = 2, alpha: float = 4.0):
        self.base_layer = type("FakeLinear", (), {
            "weight": type("FakeParam", (), {
                "data": torch.zeros(out_features, in_features),
            })(),
        })()
        self.lora_A = {"default": type("LA", (), {
            "weight": torch.randn(r, in_features) * 0.01,
        })()}
        self.lora_B = {"default": type("LB", (), {
            "weight": torch.randn(out_features, r) * 0.01,
        })()}
        self.scaling = {"default": alpha / r}
        self.active_adapter = "default"

        self.lora_A["default"].weight = torch.nn.Parameter(self.lora_A["default"].weight)
        self.lora_B["default"].weight = torch.nn.Parameter(self.lora_B["default"].weight)


class _MockModel:
    """Mock PEFT model with one layer for testing sync -> weight_dict content."""

    def __init__(self):
        self._modules = {
            "base_model.model.model.layers.0.self_attn.q_proj": _MockLoraLinear(8, 16, r=2, alpha=4.0),
            "base_model.model.model.layers.0.self_attn.k_proj": _MockLoraLinear(4, 16, r=2, alpha=4.0),
            "base_model.model.model.layers.0.self_attn.v_proj": _MockLoraLinear(4, 16, r=2, alpha=4.0),
            "base_model.model.model.layers.0.self_attn.o_proj": _MockLoraLinear(16, 8, r=2, alpha=4.0),
            "base_model.model.model.layers.0.mlp.gate_proj": _MockLoraLinear(32, 16, r=2, alpha=4.0),
            "base_model.model.model.layers.0.mlp.up_proj": _MockLoraLinear(32, 16, r=2, alpha=4.0),
            "base_model.model.model.layers.0.mlp.down_proj": _MockLoraLinear(16, 32, r=2, alpha=4.0),
        }

    def named_modules(self):
        for name, mod in self._modules.items():
            yield name, mod


class _MockVllmEngine:
    """Captures weight_dict from collective_rpc for assertions."""

    def __init__(self):
        self.last_weight_dict = None
        self.rpc_calls = []

    def collective_rpc(self, method: str, kwargs: dict = None):
        self.rpc_calls.append((method, kwargs or {}))
        if method == "update_weights_from_numpy":
            self.last_weight_dict = (kwargs or {}).get("weight_dict", {})


# ---------------------------------------------------------------------------
# Weight tying tests: sync calls collective_rpc with correct payload
# ---------------------------------------------------------------------------

def test_sync_calls_collective_rpc_once():
    """Weight tying: sync must call collective_rpc exactly once with update_weights_from_numpy."""
    model = _MockModel()
    engine = _MockVllmEngine()
    sync_lora_weights_to_vllm_goedel(model, engine, GOEDEL_LORA_TARGET_MODULES)

    assert len(engine.rpc_calls) == 1, "collective_rpc must be called exactly once"
    method, kwargs = engine.rpc_calls[0]
    assert method == "update_weights_from_numpy", f"expected update_weights_from_numpy, got {method!r}"
    assert "weight_dict" in kwargs


def test_sync_weight_dict_non_empty():
    """Weight tying: weight_dict passed to vLLM must be non-empty."""
    model = _MockModel()
    engine = _MockVllmEngine()
    sync_lora_weights_to_vllm_goedel(model, engine, GOEDEL_LORA_TARGET_MODULES)

    assert engine.last_weight_dict is not None
    assert len(engine.last_weight_dict) >= 7, (
        f"expected at least 7 LoRA layers; got {len(engine.last_weight_dict)} keys"
    )


def test_sync_keys_match_vllm_pattern():
    """Weight tying: all keys must follow model.layers.N.<sub>.<proj>.weight (Goedel/Qwen2/Qwen3)."""
    model = _MockModel()
    engine = _MockVllmEngine()
    sync_lora_weights_to_vllm_goedel(model, engine, GOEDEL_LORA_TARGET_MODULES)

    for key in engine.last_weight_dict:
        assert key.startswith("model.layers."), f"key must start with model.layers.; got {key!r}"
        assert not key.startswith("language_model."), f"Goedel must not use language_model. prefix; got {key!r}"
        assert key.endswith(".weight"), f"key must end with .weight; got {key!r}"


def test_sync_produces_unfused_keys():
    """Sync must send unfused names (q_proj, k_proj, v_proj, etc.); vLLM fuses internally."""
    model = _MockModel()
    engine = _MockVllmEngine()
    sync_lora_weights_to_vllm_goedel(model, engine, GOEDEL_LORA_TARGET_MODULES)

    keys = set(engine.last_weight_dict.keys())
    for proj in ["q_proj", "k_proj", "v_proj"]:
        assert f"model.layers.0.self_attn.{proj}.weight" in keys, f"Missing {proj}"
    assert "model.layers.0.self_attn.o_proj.weight" in keys
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        assert f"model.layers.0.mlp.{proj}.weight" in keys, f"Missing {proj}"

    assert not any("qkv_proj" in k for k in keys), "must not pre-fuse qkv_proj"
    assert not any("gate_up_proj" in k for k in keys), "must not pre-fuse gate_up_proj"


def test_sync_merged_weight_contains_lora_delta():
    """Merged weight W = W_base + lora_B @ lora_A * scale; with W_base=0, W equals delta (weight tying)."""
    model = _MockModel()
    mod = model._modules["base_model.model.model.layers.0.self_attn.o_proj"]
    lora_A = mod.lora_A["default"].weight.detach()
    lora_B = mod.lora_B["default"].weight.detach()
    scale = mod.scaling["default"]
    expected_delta = (lora_B @ lora_A * scale).to(torch.float16).numpy()

    engine = _MockVllmEngine()
    sync_lora_weights_to_vllm_goedel(model, engine, GOEDEL_LORA_TARGET_MODULES)

    got = engine.last_weight_dict["model.layers.0.self_attn.o_proj.weight"]
    np.testing.assert_allclose(got, expected_delta, atol=1e-3, rtol=1e-3)


def test_sync_raises_if_no_lora_modules():
    """Sync must raise RuntimeError when no LoRA-wrapped modules are found."""
    import pytest

    class _EmptyModel:
        def named_modules(self):
            return iter([("embed", object())])

    with pytest.raises(RuntimeError, match="No LoRA-wrapped layers found"):
        sync_lora_weights_to_vllm_goedel(_EmptyModel(), _MockVllmEngine(), GOEDEL_LORA_TARGET_MODULES)


def test_goedel_lora_target_modules_match_modal_app():
    """GOEDEL_LORA_TARGET_MODULES must match LoraConfig.target_modules in modal_app.py."""
    expected = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert GOEDEL_LORA_TARGET_MODULES == expected, (
        "Keep in sync with modal_app.py LoraConfig.target_modules"
    )


# ---------------------------------------------------------------------------
# Main (run without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_name_translation_q_proj,
        test_name_translation_mlp,
        test_name_translation_no_language_model_prefix,
        test_name_translation_ends_with_weight,
        test_sync_calls_collective_rpc_once,
        test_sync_weight_dict_non_empty,
        test_sync_keys_match_vllm_pattern,
        test_sync_produces_unfused_keys,
        test_sync_merged_weight_contains_lora_delta,
        test_sync_raises_if_no_lora_modules,
        test_goedel_lora_target_modules_match_modal_app,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
