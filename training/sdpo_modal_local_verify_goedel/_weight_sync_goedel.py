"""
TLDR: LoRA-to-vLLM weight synchronization for the Goedel prover pipeline (Qwen2/Qwen3-style).

Keeps the vLLM generation model tied to the HuggingFace QLoRA model: after each
optimizer.step(), merged LoRA weights are pushed into vLLM in-place so the next
generate_only() uses the updated policy. Parsing logic is NOT shared with qwen_sdpo.

Architecture (same pattern as sdpo_modal_local_verify_kimina):
  - Goedel-Prover-V2-8B is Qwen-based; vLLM exposes params as model.layers.* (no
    language_model. prefix; that is Qwen3.5-only in qwen_sdpo).
  - We send UNFUSED names (q_proj, k_proj, v_proj, gate_proj, up_proj, etc.); vLLM's
    load_weights() / stacked_params_mapping routes them into fused params internally.

Flow per SDPO step:
  optimizer.step() → sync_lora_weights_to_vllm_goedel() → collective_rpc → load_weights
  → next generate_only() uses updated weights. No chunking; chunked sync can be added
  later if OOM on smaller GPUs.

Used by: sdpo_modal_local_verify_goedel/modal_trainer.py, modal_app.py (worker_extension_cls).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM


# ---------------------------------------------------------------------------
# LoRA target modules (must match LoraConfig.target_modules in modal_app.py)
# ---------------------------------------------------------------------------

GOEDEL_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# ---------------------------------------------------------------------------
# vLLM Worker Extension (V1 native pattern)
# ---------------------------------------------------------------------------

class GoedelSDPOWorkerExtension:
    """Extension class mixed into vLLM's internal worker via worker_extension_cls=.

    Registered at LLM() init time via:
        worker_extension_cls="sdpo_modal_local_verify_goedel._weight_sync_goedel.GoedelSDPOWorkerExtension"

    Receives merged LoRA weights as numpy arrays (unfused vLLM param names),
    converts to bfloat16 on device, and applies in-place via load_weights() so
    CUDA graphs remain valid.
    """

    def update_weights_from_numpy(self, weight_dict: dict) -> bool:
        """Apply in-place weight update from numpy arrays.

        Args:
            weight_dict: {vllm_param_name: float16 numpy array (CPU)}, e.g.
                "model.layers.0.self_attn.q_proj.weight" -> array.

        Returns:
            True on success.
        """
        import torch
        import numpy as np

        weights = []
        for param_name, arr in weight_dict.items():
            tensor = torch.from_numpy(np.asarray(arr, dtype=np.float16)).to(
                device=self.device, dtype=torch.bfloat16
            )
            weights.append((param_name, tensor))

        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()
        return True

    def check_weights_updated(self, param_name: str, expected_checksum: float) -> bool:
        """Verify a weight tensor's L2 norm matches expected value (for debugging)."""
        import torch

        for name, param in self.model_runner.model.named_parameters():
            if name == param_name:
                actual = param.data.norm(2).item()
                return abs(actual - expected_checksum) < 1e-3
        return False


# ---------------------------------------------------------------------------
# Name translation: HF PEFT → vLLM (Goedel/Qwen2/Qwen3: model.layers.*, no language_model.)
# ---------------------------------------------------------------------------

def _hf_to_vllm_name_goedel(module_name: str) -> str:
    """Translate HF PEFT module name to vLLM parameter name for Goedel (Qwen2/Qwen3-style).

    HF PEFT: base_model.model.model.layers.0.self_attn.q_proj
    vLLM:    model.layers.0.self_attn.q_proj.weight

    Goedel-Prover-V2-8B is Qwen-based; vLLM loads it with the LM under "model", not
    "language_model" (unlike Qwen3.5 in qwen_sdpo).
    """
    inner = module_name.removeprefix("base_model.model.")
    return inner + ".weight"


# ---------------------------------------------------------------------------
# Sync function: called by the trainer after optimizer.step()
# ---------------------------------------------------------------------------

def sync_lora_weights_to_vllm_goedel(
    hf_model,
    vllm_engine: "LLM",
    lora_target_modules: list[str],
) -> None:
    """Push merged LoRA weights from the HF QLoRA model into the vLLM engine in-place.

    For each LoRA-wrapped layer:
      W_merged = dequant(W_base_4bit) + lora_B @ lora_A * (alpha / r)

    Names are translated to vLLM's unfused parameter names (model.layers.*). We send
    q_proj, k_proj, v_proj, etc. individually; vLLM's load_weights routes them into
    fused params internally via stacked_params_mapping.

    Args:
        hf_model: PEFT-wrapped QLoRA model (get_peft_model(...)).
        vllm_engine: LLM() instance with worker_extension_cls=GoedelSDPOWorkerExtension.
        lora_target_modules: Leaf names with LoRA adapters (e.g. GOEDEL_LORA_TARGET_MODULES).
    """
    import torch
    import numpy as np

    try:
        from bitsandbytes.nn import Linear4bit
        import bitsandbytes.functional as bnb_F
    except ImportError:
        Linear4bit = None
        bnb_F = None

    lora_module_names = set(lora_target_modules)
    weight_dict: dict[str, object] = {}

    for module_name, module in hf_model.named_modules():
        if not (
            hasattr(module, "base_layer")
            and hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
        ):
            continue

        leaf = module_name.split(".")[-1]
        if leaf not in lora_module_names:
            continue

        base_layer = module.base_layer
        if Linear4bit is not None and isinstance(base_layer, Linear4bit):
            w_base = bnb_F.dequantize_4bit(
                base_layer.weight.data,
                base_layer.weight.quant_state,
            ).to(torch.bfloat16)
        else:
            w_base = base_layer.weight.data.to(torch.bfloat16)

        active_adapter = module.active_adapter
        if isinstance(active_adapter, (list, tuple)):
            active_adapter = active_adapter[0]
        lora_A = module.lora_A[active_adapter].weight.detach()
        lora_B = module.lora_B[active_adapter].weight.detach()
        scale = module.scaling[active_adapter]

        delta = (lora_B @ lora_A).to(torch.bfloat16) * scale
        w_merged = (w_base + delta).contiguous()

        vllm_name = _hf_to_vllm_name_goedel(module_name)
        arr = w_merged.cpu().to(torch.float16).numpy()
        weight_dict[vllm_name] = arr

    if not weight_dict:
        module_sample = [(n, type(m).__name__) for n, m in hf_model.named_modules()][:20]
        sample_str = "\n".join(f"  {n}: {t}" for n, t in module_sample)
        raise RuntimeError(
            f"No LoRA-wrapped layers found for target modules: {lora_target_modules}.\n"
            f"(Checked for modules with base_layer + lora_A + lora_B.)\n"
            f"First 20 modules:\n{sample_str}"
        )

    vllm_engine.collective_rpc(
        "update_weights_from_numpy",
        kwargs={"weight_dict": weight_dict},
    )
