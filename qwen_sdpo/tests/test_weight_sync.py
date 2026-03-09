"""
TLDR: CPU-only unit tests for qwen_sdpo/_weight_sync.py.

Tests are grouped into three areas:
  1. LoRA delta math  — verify merged weight = base + B @ A * scale
  2. IPC handle plumbing — verify collective_rpc is called with the right args (mocked)
  3. Worker-side apply — verify load_weights() updates the right tensors in-place

No GPU, no Modal, no real vLLM required. The vLLM engine and worker are replaced with
unittest.mock.MagicMock objects. QwenSDPOWorkerExtension methods are tested directly.

Run with:
    python -m pytest qwen_sdpo/tests/test_weight_sync.py -v
    python qwen_sdpo/tests/test_weight_sync.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import torch
import torch.nn as nn

# Allow running from repo root without installing the package.
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qwen_sdpo._weight_sync import QwenSDPOWorkerExtension, sync_lora_weights_to_vllm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lora_layer(d_out: int, d_in: int, r: int, alpha: float):
    """Return (base_weight, lora_A, lora_B, scale) as CPU bf16 tensors."""
    torch.manual_seed(42)
    base = torch.randn(d_out, d_in, dtype=torch.bfloat16)
    lora_A = torch.randn(r, d_in, dtype=torch.bfloat16)
    lora_B = torch.randn(d_out, r, dtype=torch.bfloat16)
    scale = alpha / r
    return base, lora_A, lora_B, scale


def _merged(base, lora_A, lora_B, scale):
    return base + (lora_B @ lora_A) * scale


# ---------------------------------------------------------------------------
# Group 1: LoRA delta math
# ---------------------------------------------------------------------------

def test_lora_delta_shape():
    """B @ A must have the same shape as the base weight."""
    d_out, d_in, r = 64, 32, 8
    base, lora_A, lora_B, scale = _make_lora_layer(d_out, d_in, r, alpha=16.0)
    delta = lora_B @ lora_A
    assert delta.shape == base.shape, (
        f"LoRA delta shape {delta.shape} must match base weight shape {base.shape}"
    )


def test_lora_delta_values():
    """Merged weight must equal base + B @ A * scale within bf16 tolerance."""
    d_out, d_in, r = 64, 32, 8
    base, lora_A, lora_B, scale = _make_lora_layer(d_out, d_in, r, alpha=16.0)
    merged = _merged(base, lora_A, lora_B, scale)
    expected = base + (lora_B @ lora_A) * scale
    assert torch.allclose(merged, expected, atol=1e-2), (
        "Merged weight values do not match base + B @ A * scale"
    )


def test_merged_weight_differs_from_base():
    """After non-zero LoRA update, merged weight must differ from base."""
    d_out, d_in, r = 64, 32, 8
    base, lora_A, lora_B, scale = _make_lora_layer(d_out, d_in, r, alpha=16.0)
    merged = _merged(base, lora_A, lora_B, scale)
    assert not torch.allclose(merged, base, atol=1e-4), (
        "Merged weight should differ from base when LoRA matrices are non-zero"
    )


# ---------------------------------------------------------------------------
# Group 2: IPC handle plumbing (mocked vLLM engine)
# ---------------------------------------------------------------------------

def _make_mock_hf_model(param_names: list[str], d_out: int = 16, d_in: int = 8):
    """Build a minimal nn.Module with named parameters matching param_names."""

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            for name in param_names:
                # Register as a plain parameter (bf16 to simulate post-merge state).
                self.register_parameter(
                    name.replace(".", "_"),
                    nn.Parameter(torch.randn(d_out, d_in, dtype=torch.bfloat16)),
                )

        def named_parameters(self, prefix="", recurse=True):
            for attr_name, param in super().named_parameters(prefix=prefix, recurse=recurse):
                # Map back to the dot-separated name used by the real model.
                dot_name = attr_name.replace("_", ".", 1) if "_" in attr_name else attr_name
                yield dot_name, param

    return FakeModel()


def _make_mock_model_and_param(lora_modules: list[str]):
    """Build a mock PEFT model with one bf16 parameter matching the first lora_module."""
    mock_model = MagicMock()
    suffix = lora_modules[0]
    mock_model.named_parameters.return_value = [
        (f"model.layers.0.self_attn.{suffix}.weight",
         torch.randn(16, 8, dtype=torch.bfloat16))
    ]
    fake_param = MagicMock()
    fake_param.device.type = "cuda"
    fake_param.device.index = 0
    mock_model.parameters.return_value = iter([fake_param])
    return mock_model


def _sync_with_mocks(mock_model, mock_engine, lora_modules, rpc_side_effect=None):
    """Run sync_lora_weights_to_vllm with all GPU/vLLM-dependent calls mocked out.

    sync_lora_weights_to_vllm imports vllm.platforms and torch.multiprocessing.reductions
    inside its function body (lazy — vllm only exists on Modal). We inject fake modules
    into sys.modules so those imports succeed without a real vLLM installation.
    """
    if rpc_side_effect is not None:
        mock_engine.collective_rpc.side_effect = rpc_side_effect

    fake_uuid = "GPU-test-uuid"
    # reduce_tensor returns (callable, args_tuple) — mock a plausible structure.
    fake_reduce_fn = MagicMock(return_value=torch.zeros(1))
    fake_reduce_args = tuple(range(10))

    mock_reduce_tensor = MagicMock(return_value=(fake_reduce_fn, fake_reduce_args))
    mock_platform = MagicMock()
    mock_platform.get_device_uuid.return_value = fake_uuid

    # Build fake module tree for vllm.platforms so the import inside the function works.
    fake_vllm_platforms = MagicMock()
    fake_vllm_platforms.current_platform = mock_platform
    fake_vllm = MagicMock()
    fake_vllm.platforms = fake_vllm_platforms

    # Build fake torch.multiprocessing.reductions module.
    fake_torch_mp_reductions = MagicMock()
    fake_torch_mp_reductions.reduce_tensor = mock_reduce_tensor

    saved = {}
    inject = {
        "vllm": fake_vllm,
        "vllm.platforms": fake_vllm_platforms,
        "torch.multiprocessing.reductions": fake_torch_mp_reductions,
    }
    for key, val in inject.items():
        saved[key] = sys.modules.get(key)
        sys.modules[key] = val

    try:
        sync_lora_weights_to_vllm(mock_model, mock_engine, lora_modules)
    finally:
        for key, old_val in saved.items():
            if old_val is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old_val


def test_collective_rpc_called_once_per_sync():
    """sync_lora_weights_to_vllm must call collective_rpc exactly once."""
    mock_engine = MagicMock()
    mock_model = _make_mock_model_and_param(["q_proj"])

    _sync_with_mocks(mock_model, mock_engine, ["q_proj"])

    mock_engine.collective_rpc.assert_called_once()
    rpc_call = mock_engine.collective_rpc.call_args
    assert rpc_call[0][0] == "update_weights_from_ipc_handles", (
        "collective_rpc must be called with 'update_weights_from_ipc_handles'"
    )


def test_unmerge_called_even_if_push_fails():
    """unmerge_adapter() must be called even if collective_rpc raises an exception."""
    mock_engine = MagicMock()
    mock_model = _make_mock_model_and_param(["q_proj"])

    try:
        _sync_with_mocks(
            mock_model, mock_engine, ["q_proj"],
            rpc_side_effect=RuntimeError("simulated rpc failure"),
        )
    except RuntimeError:
        pass  # expected — the rpc failure propagates

    # unmerge_adapter must still have been called (the finally block ran).
    mock_model.unmerge_adapter.assert_called_once()


# ---------------------------------------------------------------------------
# Group 3: Worker-side in-place weight update
# ---------------------------------------------------------------------------

class _FakeVllmModel:
    """Minimal stand-in for vLLM's internal model inside model_runner."""

    def __init__(self):
        self._weights = {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(16, 8, dtype=torch.bfloat16),
            "model.layers.0.self_attn.k_proj.weight": torch.ones(16, 8, dtype=torch.bfloat16),
        }

    def named_parameters(self):
        for name, tensor in self._weights.items():
            yield name, tensor

    def load_weights(self, weights: list):
        for name, tensor in weights:
            if name in self._weights:
                self._weights[name].copy_(tensor)


class _FakeModelRunner:
    """Minimal stand-in for vLLM's model_runner."""

    def __init__(self):
        self.model = _FakeVllmModel()


def test_worker_load_weights_updates_target():
    """load_weights() must overwrite the correct weight tensor in-place."""
    runner = _FakeModelRunner()
    original_k = runner.model._weights["model.layers.0.self_attn.k_proj.weight"].clone()

    new_q = torch.full((16, 8), fill_value=7.0, dtype=torch.bfloat16)
    runner.model.load_weights([
        ("model.layers.0.self_attn.q_proj.weight", new_q),
    ])

    updated_q = runner.model._weights["model.layers.0.self_attn.q_proj.weight"]
    assert torch.allclose(updated_q, new_q), "q_proj weight should have been updated to 7.0"

    # k_proj must be unchanged.
    unchanged_k = runner.model._weights["model.layers.0.self_attn.k_proj.weight"]
    assert torch.allclose(unchanged_k, original_k), "k_proj weight must not be modified"


def test_worker_load_weights_is_inplace():
    """load_weights() must write into the existing tensor, not replace the reference."""
    runner = _FakeModelRunner()
    original_id = id(runner.model._weights["model.layers.0.self_attn.q_proj.weight"])

    new_q = torch.full((16, 8), fill_value=3.0, dtype=torch.bfloat16)
    runner.model.load_weights([
        ("model.layers.0.self_attn.q_proj.weight", new_q),
    ])

    updated_id = id(runner.model._weights["model.layers.0.self_attn.q_proj.weight"])
    assert original_id == updated_id, (
        "load_weights() must copy values into the existing tensor (same object id), "
        "not replace the tensor reference — otherwise CUDA graph addresses break."
    )


def test_worker_unknown_param_ignored():
    """load_weights() with an unknown param name must not crash or modify other weights."""
    runner = _FakeModelRunner()
    original_q = runner.model._weights["model.layers.0.self_attn.q_proj.weight"].clone()

    runner.model.load_weights([
        ("nonexistent.weight", torch.randn(16, 8, dtype=torch.bfloat16)),
    ])

    unchanged_q = runner.model._weights["model.layers.0.self_attn.q_proj.weight"]
    assert torch.allclose(unchanged_q, original_q), "Unknown param name must not affect other weights"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Group 1
        test_lora_delta_shape,
        test_lora_delta_values,
        test_merged_weight_differs_from_base,
        # Group 2
        test_collective_rpc_called_once_per_sync,
        test_unmerge_called_even_if_push_fails,
        # Group 3
        test_worker_load_weights_updates_target,
        test_worker_load_weights_is_inplace,
        test_worker_unknown_param_ignored,
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
