"""
CPU-only unit tests for sdpo_modal_local_verify_kimina/_weight_sync_kimina.py.

Tests cover:
  - _hf_to_vllm_name_kimina: HF PEFT → vLLM Qwen3 parameter name translation
  - _fuse_weights: QKV fusion, gate-up fusion, unfused modules pass-through
  - sync_lora_weights_to_vllm_kimina: end-to-end with mock LoRA modules and
    a mock collective_rpc that captures the weight_dict without GPU

No GPU, no vLLM, no bitsandbytes required.

Run with:
    python -m pytest sdpo_modal_local_verify_kimina/tests/test_weight_sync_kimina.py -v
    # or directly:
    python sdpo_modal_local_verify_kimina/tests/test_weight_sync_kimina.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the path when run directly.
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sdpo_modal_local_verify_kimina._weight_sync_kimina import (
    KIMINA_LORA_TARGET_MODULES,
    KiminaSDPOWorkerExtension,
    _fuse_weights,
    _hf_to_vllm_name_kimina,
    sync_lora_weights_to_vllm_kimina,
)


# ---------------------------------------------------------------------------
# Tests: _hf_to_vllm_name_kimina
# ---------------------------------------------------------------------------

def test_name_translation_q_proj():
    hf = "base_model.model.model.layers.0.self_attn.q_proj"
    expected = "model.layers.0.self_attn.q_proj.weight"
    assert _hf_to_vllm_name_kimina(hf) == expected, f"got {_hf_to_vllm_name_kimina(hf)!r}"


def test_name_translation_k_proj():
    hf = "base_model.model.model.layers.5.self_attn.k_proj"
    expected = "model.layers.5.self_attn.k_proj.weight"
    assert _hf_to_vllm_name_kimina(hf) == expected


def test_name_translation_o_proj():
    hf = "base_model.model.model.layers.2.self_attn.o_proj"
    expected = "model.layers.2.self_attn.o_proj.weight"
    assert _hf_to_vllm_name_kimina(hf) == expected


def test_name_translation_gate_proj():
    hf = "base_model.model.model.layers.3.mlp.gate_proj"
    expected = "model.layers.3.mlp.gate_proj.weight"
    assert _hf_to_vllm_name_kimina(hf) == expected


def test_name_translation_down_proj():
    hf = "base_model.model.model.layers.27.mlp.down_proj"
    expected = "model.layers.27.mlp.down_proj.weight"
    assert _hf_to_vllm_name_kimina(hf) == expected


def test_name_translation_no_language_model_prefix():
    """Qwen3 vLLM names must NOT start with 'language_model.' (that's Qwen3.5-only)."""
    hf = "base_model.model.model.layers.0.self_attn.q_proj"
    result = _hf_to_vllm_name_kimina(hf)
    assert not result.startswith("language_model."), (
        f"Qwen3 vLLM names should not start with 'language_model.', got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Tests: _fuse_weights — QKV fusion
# ---------------------------------------------------------------------------

def test_qkv_fusion_shape():
    """q+k+v with shapes (Hq, D), (Hk, D), (Hv, D) → qkv (Hq+Hk+Hv, D)."""
    q = np.ones((32, 64), dtype=np.float16)
    k = np.ones((16, 64), dtype=np.float16) * 2
    v = np.ones((16, 64), dtype=np.float16) * 3
    individual = {
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.self_attn.k_proj.weight": k,
        "model.layers.0.self_attn.v_proj.weight": v,
    }
    result = _fuse_weights(individual)
    assert "model.layers.0.self_attn.qkv_proj.weight" in result
    fused = result["model.layers.0.self_attn.qkv_proj.weight"]
    assert fused.shape == (32 + 16 + 16, 64), f"expected (64, 64), got {fused.shape}"


def test_qkv_fusion_values():
    """Fused QKV must be dim-0 concatenation of [q, k, v] in that order."""
    q = np.full((4, 8), 1.0, dtype=np.float16)
    k = np.full((4, 8), 2.0, dtype=np.float16)
    v = np.full((4, 8), 3.0, dtype=np.float16)
    individual = {
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.self_attn.k_proj.weight": k,
        "model.layers.0.self_attn.v_proj.weight": v,
    }
    result = _fuse_weights(individual)
    fused = result["model.layers.0.self_attn.qkv_proj.weight"]
    expected = np.concatenate([q, k, v], axis=0)
    np.testing.assert_array_equal(fused, expected)


def test_qkv_originals_removed():
    """After fusion, individual q/k/v weight entries must not appear in the result."""
    q = np.ones((4, 8), dtype=np.float16)
    k = np.ones((4, 8), dtype=np.float16)
    v = np.ones((4, 8), dtype=np.float16)
    individual = {
        "model.layers.0.self_attn.q_proj.weight": q,
        "model.layers.0.self_attn.k_proj.weight": k,
        "model.layers.0.self_attn.v_proj.weight": v,
    }
    result = _fuse_weights(individual)
    # Use dot-bracketed patterns to avoid matching substrings inside fused names
    # (e.g. "v_proj" appears inside "qkv_proj", "up_proj" inside "gate_up_proj")
    for name in ["q_proj", "k_proj", "v_proj"]:
        assert not any(f".{name}.weight" in key for key in result), (
            f"Unfused .{name}.weight should not appear after fusion; got keys: {list(result)}"
        )


def test_qkv_fusion_multiple_layers():
    """QKV fusion applied independently to each layer."""
    individual = {}
    for layer in [0, 5, 27]:
        for proj, val in [("q_proj", 1.0), ("k_proj", 2.0), ("v_proj", 3.0)]:
            individual[f"model.layers.{layer}.self_attn.{proj}.weight"] = np.full((4, 8), val, dtype=np.float16)
    result = _fuse_weights(individual)
    for layer in [0, 5, 27]:
        key = f"model.layers.{layer}.self_attn.qkv_proj.weight"
        assert key in result, f"Missing fused key for layer {layer}"
        assert result[key].shape == (12, 8)


# ---------------------------------------------------------------------------
# Tests: _fuse_weights — gate-up fusion
# ---------------------------------------------------------------------------

def test_gate_up_fusion_shape():
    gate = np.ones((64, 32), dtype=np.float16)
    up = np.ones((64, 32), dtype=np.float16) * 2
    individual = {
        "model.layers.0.mlp.gate_proj.weight": gate,
        "model.layers.0.mlp.up_proj.weight": up,
    }
    result = _fuse_weights(individual)
    assert "model.layers.0.mlp.gate_up_proj.weight" in result
    fused = result["model.layers.0.mlp.gate_up_proj.weight"]
    assert fused.shape == (128, 32), f"expected (128, 32), got {fused.shape}"


def test_gate_up_fusion_values():
    """Fused gate_up must be dim-0 concatenation of [gate, up] in that order."""
    gate = np.full((4, 8), 1.5, dtype=np.float16)
    up = np.full((4, 8), 2.5, dtype=np.float16)
    individual = {
        "model.layers.1.mlp.gate_proj.weight": gate,
        "model.layers.1.mlp.up_proj.weight": up,
    }
    result = _fuse_weights(individual)
    fused = result["model.layers.1.mlp.gate_up_proj.weight"]
    expected = np.concatenate([gate, up], axis=0)
    np.testing.assert_array_equal(fused, expected)


def test_gate_up_originals_removed():
    gate = np.ones((4, 8), dtype=np.float16)
    up = np.ones((4, 8), dtype=np.float16)
    individual = {
        "model.layers.0.mlp.gate_proj.weight": gate,
        "model.layers.0.mlp.up_proj.weight": up,
    }
    result = _fuse_weights(individual)
    # Use dot-bracketed patterns to avoid matching substrings inside fused names
    # (e.g. "up_proj" appears inside "gate_up_proj")
    assert not any(".gate_proj.weight" in k for k in result)
    assert not any(".up_proj.weight" in k for k in result)


# ---------------------------------------------------------------------------
# Tests: _fuse_weights — unfused modules pass through unchanged
# ---------------------------------------------------------------------------

def test_o_proj_passes_through():
    """o_proj is not fused — it must appear verbatim in the result."""
    arr = np.ones((16, 64), dtype=np.float16)
    individual = {"model.layers.0.self_attn.o_proj.weight": arr}
    result = _fuse_weights(individual)
    assert "model.layers.0.self_attn.o_proj.weight" in result
    np.testing.assert_array_equal(result["model.layers.0.self_attn.o_proj.weight"], arr)


def test_down_proj_passes_through():
    """down_proj is not fused — it must appear verbatim in the result."""
    arr = np.ones((32, 64), dtype=np.float16)
    individual = {"model.layers.0.mlp.down_proj.weight": arr}
    result = _fuse_weights(individual)
    assert "model.layers.0.mlp.down_proj.weight" in result
    np.testing.assert_array_equal(result["model.layers.0.mlp.down_proj.weight"], arr)


def test_full_layer_fusion_result_keys():
    """A full set of one layer's LoRA weights should produce exactly 4 fused keys."""
    individual = {
        "model.layers.0.self_attn.q_proj.weight": np.ones((4, 8), dtype=np.float16),
        "model.layers.0.self_attn.k_proj.weight": np.ones((4, 8), dtype=np.float16),
        "model.layers.0.self_attn.v_proj.weight": np.ones((4, 8), dtype=np.float16),
        "model.layers.0.self_attn.o_proj.weight": np.ones((8, 8), dtype=np.float16),
        "model.layers.0.mlp.gate_proj.weight": np.ones((16, 8), dtype=np.float16),
        "model.layers.0.mlp.up_proj.weight": np.ones((16, 8), dtype=np.float16),
        "model.layers.0.mlp.down_proj.weight": np.ones((8, 16), dtype=np.float16),
    }
    result = _fuse_weights(individual)
    expected_keys = {
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }
    assert set(result.keys()) == expected_keys, f"Got keys: {set(result.keys())}"


# ---------------------------------------------------------------------------
# Tests: target modules
# ---------------------------------------------------------------------------

def test_no_linear_attention_modules():
    """Qwen3 (Kimina) has no linear-attention modules — these are Qwen3.5-only."""
    qwen35_only = {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
    for mod in qwen35_only:
        assert mod not in KIMINA_LORA_TARGET_MODULES, (
            f"'{mod}' is a Qwen3.5-only module and should not be in KIMINA_LORA_TARGET_MODULES"
        )


def test_standard_modules_present():
    """Standard attention and MLP modules must be present."""
    required = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    for mod in required:
        assert mod in KIMINA_LORA_TARGET_MODULES, f"'{mod}' must be in KIMINA_LORA_TARGET_MODULES"


# ---------------------------------------------------------------------------
# Tests: sync_lora_weights_to_vllm_kimina end-to-end (mock model + rpc)
# ---------------------------------------------------------------------------

class _MockLoraLinear:
    """Minimal mock of a PEFT LoRA-wrapped linear layer (fp32 base, no quantization)."""

    def __init__(self, out_features: int, in_features: int, r: int = 2, alpha: float = 4.0):
        # base_layer: plain weight, no quantization
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

        # Make lora_A/B weights proper tensors (requires_grad would be set in real training)
        self.lora_A["default"].weight = torch.nn.Parameter(self.lora_A["default"].weight)
        self.lora_B["default"].weight = torch.nn.Parameter(self.lora_B["default"].weight)


class _MockModel:
    """Mock PEFT-wrapped model with a single attention layer for testing."""

    def __init__(self):
        self._modules: dict[str, _MockLoraLinear] = {
            "base_model.model.model.layers.0.self_attn.q_proj": _MockLoraLinear(8, 16, r=2, alpha=4.0),
            "base_model.model.model.layers.0.self_attn.k_proj": _MockLoraLinear(4, 16, r=2, alpha=4.0),
            "base_model.model.model.layers.0.self_attn.v_proj": _MockLoraLinear(4, 16, r=2, alpha=4.0),
            "base_model.model.model.layers.0.self_attn.o_proj": _MockLoraLinear(16, 8, r=2, alpha=4.0),
            "base_model.model.model.layers.0.mlp.gate_proj":    _MockLoraLinear(32, 16, r=2, alpha=4.0),
            "base_model.model.model.layers.0.mlp.up_proj":      _MockLoraLinear(32, 16, r=2, alpha=4.0),
            "base_model.model.model.layers.0.mlp.down_proj":    _MockLoraLinear(16, 32, r=2, alpha=4.0),
        }

    def named_modules(self):
        for name, mod in self._modules.items():
            yield name, mod


class _MockVllmEngine:
    """Mock vLLM engine that captures the weight_dict from collective_rpc."""

    def __init__(self):
        self.last_weight_dict: dict | None = None

    def collective_rpc(self, method: str, kwargs: dict = None):
        assert method == "update_weights_from_numpy"
        self.last_weight_dict = (kwargs or {}).get("weight_dict", {})


def test_sync_produces_unfused_keys():
    """End-to-end: sync should send unfused vLLM parameter names (q/k/v individually).

    vLLM's Qwen3 load_weights (via Qwen2Model.stacked_params_mapping) routes
    q_proj/k_proj/v_proj into the fused qkv_proj param internally. We must NOT
    pre-fuse here — sending qkv_proj.weight directly would bypass that routing.
    """
    model = _MockModel()
    engine = _MockVllmEngine()
    sync_lora_weights_to_vllm_kimina(model, engine, KIMINA_LORA_TARGET_MODULES)

    assert engine.last_weight_dict is not None, "collective_rpc was never called"
    keys = set(engine.last_weight_dict.keys())

    # Unfused individual keys expected — vLLM's stacked_params_mapping handles fusion
    for proj in ["q_proj", "k_proj", "v_proj"]:
        assert f"model.layers.0.self_attn.{proj}.weight" in keys, (
            f"Missing {proj}; got keys: {keys}"
        )
    assert "model.layers.0.self_attn.o_proj.weight" in keys
    for proj in ["gate_proj", "up_proj"]:
        assert f"model.layers.0.mlp.{proj}.weight" in keys, f"Missing {proj}; got keys: {keys}"
    assert "model.layers.0.mlp.down_proj.weight" in keys

    # Pre-fused qkv_proj/gate_up_proj keys must NOT be present
    assert not any("qkv_proj" in k for k in keys), f"Found pre-fused qkv_proj key: {keys}"
    assert not any("gate_up_proj" in k for k in keys), f"Found pre-fused gate_up_proj key: {keys}"


def test_sync_no_language_model_prefix():
    """vLLM keys for Qwen3 must not start with 'language_model.'."""
    model = _MockModel()
    engine = _MockVllmEngine()
    sync_lora_weights_to_vllm_kimina(model, engine, KIMINA_LORA_TARGET_MODULES)

    for key in engine.last_weight_dict:
        assert not key.startswith("language_model."), (
            f"Qwen3 keys should not have language_model. prefix: {key!r}"
        )


def test_sync_individual_proj_shapes():
    """Individual q/k/v/o proj shapes must match the mock model dimensions."""
    model = _MockModel()
    engine = _MockVllmEngine()
    sync_lora_weights_to_vllm_kimina(model, engine, KIMINA_LORA_TARGET_MODULES)

    wd = engine.last_weight_dict
    # q: (8, 16), k: (4, 16), v: (4, 16)
    assert wd["model.layers.0.self_attn.q_proj.weight"].shape == (8, 16)
    assert wd["model.layers.0.self_attn.k_proj.weight"].shape == (4, 16)
    assert wd["model.layers.0.self_attn.v_proj.weight"].shape == (4, 16)
    # gate: (32, 16), up: (32, 16)
    assert wd["model.layers.0.mlp.gate_proj.weight"].shape == (32, 16)
    assert wd["model.layers.0.mlp.up_proj.weight"].shape == (32, 16)


def test_sync_merged_weight_contains_lora_delta():
    """Merged weight W = W_base + lora_B @ lora_A * scale; with W_base=0, W = delta."""
    model = _MockModel()

    # Use a single module with known lora_A, lora_B, scale for a deterministic check.
    mod = model._modules["base_model.model.model.layers.0.self_attn.o_proj"]
    lora_A = mod.lora_A["default"].weight.detach()
    lora_B = mod.lora_B["default"].weight.detach()
    scale = mod.scaling["default"]
    expected_delta = (lora_B @ lora_A * scale).to(torch.float16).numpy()

    engine = _MockVllmEngine()
    sync_lora_weights_to_vllm_kimina(model, engine, KIMINA_LORA_TARGET_MODULES)

    got = engine.last_weight_dict["model.layers.0.self_attn.o_proj.weight"]
    # W_base is zero → merged == delta
    np.testing.assert_allclose(got, expected_delta, atol=1e-3, rtol=1e-3)


def test_sync_raises_if_no_lora_modules():
    """sync should raise RuntimeError when no LoRA-wrapped modules are found."""
    import pytest

    class _EmptyModel:
        def named_modules(self):
            return iter([("embed", object())])

    with pytest.raises(RuntimeError, match="No LoRA-wrapped layers found"):
        sync_lora_weights_to_vllm_kimina(_EmptyModel(), _MockVllmEngine(), KIMINA_LORA_TARGET_MODULES)


# ---------------------------------------------------------------------------
# Main runner (for running directly without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Name translation
        test_name_translation_q_proj,
        test_name_translation_k_proj,
        test_name_translation_o_proj,
        test_name_translation_gate_proj,
        test_name_translation_down_proj,
        test_name_translation_no_language_model_prefix,
        # QKV fusion
        test_qkv_fusion_shape,
        test_qkv_fusion_values,
        test_qkv_originals_removed,
        test_qkv_fusion_multiple_layers,
        # Gate-up fusion
        test_gate_up_fusion_shape,
        test_gate_up_fusion_values,
        test_gate_up_originals_removed,
        # Unfused pass-through
        test_o_proj_passes_through,
        test_down_proj_passes_through,
        test_full_layer_fusion_result_keys,
        # Target modules
        test_no_linear_attention_modules,
        test_standard_modules_present,
        # End-to-end sync
        test_sync_produces_unfused_keys,
        test_sync_no_language_model_prefix,
        test_sync_individual_proj_shapes,
        test_sync_merged_weight_contains_lora_delta,
        test_sync_raises_if_no_lora_modules,
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
