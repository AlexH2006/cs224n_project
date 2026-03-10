"""
TLDR: LoRA-to-vLLM weight synchronization for Qwen3.5-4B (Qwen3_5ForConditionalGeneration).

Qwen3.5 architecture in vLLM:
  Qwen3_5ForConditionalGeneration wraps Qwen3_5ForCausalLM under self.language_model,
  so vLLM parameter names are: language_model.model.layers.*

  HF PEFT wraps the full ConditionalGeneration model, so PEFT module names are:
    base_model.model.language_model.model.layers.0.self_attn.q_proj
    → strip base_model.model.: language_model.model.layers.0.self_attn.q_proj
    → add .weight:             language_model.model.layers.0.self_attn.q_proj.weight
  This matches vLLM's naming, so the same strip+suffix translation works.

  vLLM's packed_modules_mapping for Qwen3_5ForCausalLMBase fuses:
    q_proj + k_proj + v_proj  →  qkv_proj         (standard attention)
    gate_proj + up_proj       →  gate_up_proj      (MLP)
    in_proj_qkv + in_proj_z   →  in_proj_qkvz     (GDN linear attention)
    in_proj_b + in_proj_a      →  in_proj_ba       (GDN linear attention)

  vLLM's load_weights() routes individual (unfused) weight names to fused params
  via stacked_params_mapping, so we send weights unfused.

Qwen3.5 4B hybrid attention (32 layers, 3:1 ratio):
  - 24 Gated DeltaNet layers (linear attention): in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
  - 8 standard attention layers: q_proj, k_proj, v_proj, o_proj
  - All layers have MLP: gate_proj, up_proj, down_proj

Flow per SDPO step:
  optimizer.step()                     # lora_A/B updated in HF model
  sync_lora_weights_to_vllm_kimina()   # dequant → merge → rpc → load_weights
  vllm_engine.generate()               # uses updated weights, no CUDA graph recapture

Used by: sdpo_modal_local_verify_kimina/modal_trainer.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM


# ---------------------------------------------------------------------------
# Target modules for Qwen3.5 (standard attention + GDN linear attention + MLP)
# ---------------------------------------------------------------------------

KIMINA_LORA_TARGET_MODULES = [
    # Standard attention (25% of layers)
    "q_proj", "k_proj", "v_proj", "o_proj",
    # Gated DeltaNet / linear attention (75% of layers)
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    # MLP (all layers)
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# vLLM Worker Extension (V1 native pattern)
# ---------------------------------------------------------------------------

class KiminaSDPOWorkerExtension:
    """Extension class mixed into vLLM's internal worker via worker_extension_cls=.

    Registered at LLM() init time via:
        worker_extension_cls="sdpo_modal_local_verify_kimina._weight_sync_kimina.KiminaSDPOWorkerExtension"

    Receives merged (unfused) weights from the HF QLoRA model, converts each
    numpy array to a bfloat16 CUDA tensor, and applies them in-place via
    load_weights() — tensor addresses stay the same, so CUDA graphs remain valid.

    Weight names use vLLM's unfused naming (load_weights routes to fused params):
      "language_model.model.layers.N.self_attn.q_proj.weight"    (standard attn)
      "language_model.model.layers.N.self_attn.o_proj.weight"
      "language_model.model.layers.N.temporal_block.in_proj_qkv.weight"  (GDN)
      "language_model.model.layers.N.temporal_block.out_proj.weight"
      "language_model.model.layers.N.mlp.gate_proj.weight"
      "language_model.model.layers.N.mlp.down_proj.weight"
    """

    def update_weights_from_numpy(self, weight_dict: dict) -> bool:
        """Apply in-place weight update from numpy arrays (fused vLLM format).

        Args:
            weight_dict: {vllm_param_name: float16 numpy array (CPU)}

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
# Name translation: HF PEFT → vLLM (Qwen3.5 ConditionalGeneration)
# ---------------------------------------------------------------------------

def _hf_to_vllm_name_kimina(module_name: str) -> str:
    """Translate a HF PEFT module name to vLLM's Qwen3_5 parameter name.

    HF PEFT wraps the full Qwen3_5ForConditionalGeneration under base_model.model.,
    so the full name is e.g.:
      base_model.model.language_model.model.layers.0.self_attn.q_proj

    Stripping base_model.model. gives:
      language_model.model.layers.0.self_attn.q_proj

    Adding .weight gives the vLLM parameter name:
      language_model.model.layers.0.self_attn.q_proj.weight

    Args:
        module_name: PEFT module name (from hf_model.named_modules()).

    Returns:
        vLLM parameter name (unfused; load_weights routes to fused params).
    """
    inner = module_name.removeprefix("base_model.model.")
    return inner + ".weight"


# ---------------------------------------------------------------------------
# Fusion helpers: individual HF weights → vLLM fused tensors
# ---------------------------------------------------------------------------

def _fuse_weights(individual: dict[str, object]) -> dict[str, object]:
    """Fuse individual projections into vLLM's packed format.

    vLLM's Qwen3_5ForCausalLMBase.packed_modules_mapping:
      qkv_proj:      [q_proj, k_proj, v_proj]       (standard attention)
      gate_up_proj:  [gate_proj, up_proj]            (MLP)
      in_proj_qkvz:  [in_proj_qkv, in_proj_z]       (GDN linear attention)
      in_proj_ba:    [in_proj_b, in_proj_a]          (GDN linear attention)

    NOTE: This function is kept for reference but is NOT called in the current
    sync path. We send unfused names and let vLLM's load_weights() route them
    via stacked_params_mapping. Only used if the vLLM model's load_weights
    does not support stacked routing (not the case for Qwen3.5).

    Args:
        individual: {vllm_unfused_name: numpy_array}

    Returns:
        New dict with projections fused into packed format.
    """
    import numpy as np

    result: dict[str, object] = {}
    consumed: set[str] = set()

    # Collect all layer prefixes that appear in individual weights.
    # e.g. "model.layers.0.self_attn" from "model.layers.0.self_attn.q_proj.weight"
    attn_prefixes: set[str] = set()
    mlp_prefixes: set[str] = set()

    for name in individual:
        if ".self_attn." in name and name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            # prefix is everything up to and including ".self_attn"
            prefix = name.rsplit(".", 2)[0]  # strip ".<proj>.weight"
            attn_prefixes.add(prefix)
        elif ".mlp." in name and name.endswith(("gate_proj.weight", "up_proj.weight")):
            prefix = name.rsplit(".", 2)[0]
            mlp_prefixes.add(prefix)

    # Fuse QKV per attention layer
    for prefix in attn_prefixes:
        q_name = f"{prefix}.q_proj.weight"
        k_name = f"{prefix}.k_proj.weight"
        v_name = f"{prefix}.v_proj.weight"
        if q_name in individual and k_name in individual and v_name in individual:
            fused = np.concatenate(
                [individual[q_name], individual[k_name], individual[v_name]], axis=0
            )
            # Replace the self_attn prefix segment with the fused key
            # e.g. "model.layers.0.self_attn" → "model.layers.0.self_attn.qkv_proj.weight"
            result[f"{prefix}.qkv_proj.weight"] = fused
            consumed.update([q_name, k_name, v_name])

    # Fuse gate+up per MLP layer
    for prefix in mlp_prefixes:
        gate_name = f"{prefix}.gate_proj.weight"
        up_name = f"{prefix}.up_proj.weight"
        if gate_name in individual and up_name in individual:
            fused = np.concatenate([individual[gate_name], individual[up_name]], axis=0)
            result[f"{prefix}.gate_up_proj.weight"] = fused
            consumed.update([gate_name, up_name])

    # Pass through unfused weights (o_proj, down_proj, etc.)
    for name, arr in individual.items():
        if name not in consumed:
            result[name] = arr

    return result


# ---------------------------------------------------------------------------
# Sync function: called by the trainer after optimizer.step()
# ---------------------------------------------------------------------------

def sync_lora_weights_to_vllm_kimina(
    hf_model,
    vllm_engine: "LLM",
    lora_target_modules: list[str],
) -> None:
    """Push merged LoRA weights from the HF QLoRA model into the vLLM engine in-place.

    For each LoRA-wrapped layer, computes:
      W_merged = dequant(W_base_4bit) + lora_B @ lora_A * (alpha / r)

    Weight names are translated from HF PEFT names to vLLM's unfused parameter names
    (e.g. "language_model.model.layers.0.self_attn.q_proj.weight"). We send unfused
    names; vLLM's Qwen3_5 load_weights() uses stacked_params_mapping to route
    q_proj/k_proj/v_proj into qkv_proj, gate/up into gate_up_proj, and GDN
    projections into in_proj_qkvz/in_proj_ba internally.

    Args:
        hf_model:            PEFT-wrapped QLoRA model (result of get_peft_model()).
        vllm_engine:         vLLM LLM() instance with
                             worker_extension_cls=KiminaSDPOWorkerExtension.
        lora_target_modules: Module leaf names that have LoRA adapters
                             (e.g. KIMINA_LORA_TARGET_MODULES).
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
        # Detect PEFT LoRA-wrapped modules by duck-typing.
        if not (
            hasattr(module, "base_layer")
            and hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
        ):
            continue

        leaf = module_name.split(".")[-1]
        if leaf not in lora_module_names:
            continue

        # Dequantize 4-bit base weight to bf16.
        base_layer = module.base_layer
        if Linear4bit is not None and isinstance(base_layer, Linear4bit):
            w_base = bnb_F.dequantize_4bit(
                base_layer.weight.data,
                base_layer.weight.quant_state,
            ).to(torch.bfloat16)
        else:
            w_base = base_layer.weight.data.to(torch.bfloat16)

        # Compute the LoRA delta.
        active_adapter = module.active_adapter
        if isinstance(active_adapter, (list, tuple)):
            active_adapter = active_adapter[0]
        lora_A = module.lora_A[active_adapter].weight.detach()  # (r, in_features)
        lora_B = module.lora_B[active_adapter].weight.detach()  # (out_features, r)
        scale = module.scaling[active_adapter]                   # alpha / r

        delta = (lora_B @ lora_A).to(torch.bfloat16) * scale
        w_merged = (w_base + delta).contiguous()

        # Translate HF PEFT name → vLLM Qwen3.5 unfused name.
        # HF: "base_model.model.language_model.model.layers.0.self_attn.q_proj"
        # vLLM: "language_model.model.layers.0.self_attn.q_proj.weight"
        # vLLM's load_weights uses stacked_params_mapping to route unfused names
        # into fused params (qkv_proj, gate_up_proj, in_proj_qkvz, in_proj_ba).
        vllm_name = _hf_to_vllm_name_kimina(module_name)

        arr = w_merged.cpu().to(torch.float16).numpy()
        weight_dict[vllm_name] = arr

    if not weight_dict:
        module_sample = [(n, type(m).__name__) for n, m in hf_model.named_modules()][:20]
        sample_str = "\n".join(f"  {n}: {t}" for n, t in module_sample)
        raise RuntimeError(
            f"No LoRA-wrapped layers found for target modules: {lora_target_modules}.\n"
            f"(Checked for modules with base_layer + lora_A + lora_B attributes.)\n"
            f"First 20 modules:\n{sample_str}"
        )

    # Push to vLLM worker in-place.
    # vLLM's Qwen3.5 load_weights uses stacked_params_mapping to fuse individual
    # projections into packed params internally — no pre-fusion needed here.
    vllm_engine.collective_rpc(
        "update_weights_from_numpy",
        kwargs={"weight_dict": weight_dict},
    )
