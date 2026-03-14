"""
TLDR: LoRA-to-vLLM weight synchronization for the Kimina pipeline (Qwen3ForCausalLM).

This is the Kimina-specific counterpart of qwen_sdpo/_weight_sync.py.  The two
differ in two architecture-driven ways:

1. Parameter name prefix
   Qwen3.5 (qwen_sdpo): vLLM wraps the LM under language_model.model.layers.*
   Qwen3   (kimina):    vLLM exposes the LM directly as model.layers.*

   Name translation for Qwen3 (no extra prefix needed):
     HF PEFT:  base_model.model.model.layers.0.self_attn.q_proj
     → strip:  model.layers.0.self_attn.q_proj
     → vLLM:   model.layers.0.self_attn.q_proj.weight

2. Fused projections in vLLM
   Qwen3ForCausalLM.packed_modules_mapping fuses:
     q_proj + k_proj + v_proj  →  self_attn.qkv_proj.weight  (dim-0 concat)
     gate_proj + up_proj        →  mlp.gate_up_proj.weight    (dim-0 concat)

   After computing individual merged weights (W_merged = dequant(W_base) + ΔLoRA),
   we must concatenate them before calling collective_rpc — vLLM's load_weights()
   only knows about the fused parameter names, not the originals.

Flow per SDPO step (same as qwen_sdpo):
  optimizer.step()                     # lora_A/B updated in HF model
  sync_lora_weights_to_vllm_kimina()  # dequant → merge → fuse → rpc → load_weights
  vllm_engine.generate()              # uses updated weights, no CUDA graph recapture

Used by: sdpo_modal_local_verify_kimina/modal_trainer.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM


# ---------------------------------------------------------------------------
# Target modules for Qwen3 (standard attention + MLP only, no linear-attn)
# ---------------------------------------------------------------------------

KIMINA_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# vLLM Worker Extension (V1 native pattern — identical logic to qwen_sdpo)
# ---------------------------------------------------------------------------

class KiminaSDPOWorkerExtension:
    """Extension class mixed into vLLM's internal worker via worker_extension_cls=.

    Registered at LLM() init time via:
        worker_extension_cls="sdpo_modal_local_verify_kimina._weight_sync_kimina.KiminaSDPOWorkerExtension"

    Receives merged (unfused) weights from the HF QLoRA model, converts each
    numpy array to a bfloat16 CUDA tensor, and applies them in-place via
    load_weights() — tensor addresses stay the same, so CUDA graphs remain valid.

    weight_dict keys use vLLM's fused naming:
      "model.layers.N.self_attn.qkv_proj.weight"   ← q+k+v stacked on dim 0
      "model.layers.N.self_attn.o_proj.weight"
      "model.layers.N.mlp.gate_up_proj.weight"      ← gate+up stacked on dim 0
      "model.layers.N.mlp.down_proj.weight"
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
# Name translation: HF PEFT → vLLM (Qwen3, no language_model. prefix)
# ---------------------------------------------------------------------------

def _hf_to_vllm_name_kimina(module_name: str) -> str:
    """Translate a HF PEFT module name to vLLM's Qwen3ForCausalLM parameter name.

    HF PEFT wraps the model under base_model.model., so the full name is:
      base_model.model.model.layers.0.self_attn.q_proj

    vLLM's Qwen3ForCausalLM stores the Qwen3Model directly under self.model,
    so the corresponding parameter name (before fusion) is:
      model.layers.0.self_attn.q_proj.weight

    Args:
        module_name: PEFT module name (from hf_model.named_modules()).

    Returns:
        vLLM parameter name (before any QKV / gate-up fusion step).
    """
    inner = module_name.removeprefix("base_model.model.")
    return inner + ".weight"


# ---------------------------------------------------------------------------
# Fusion helpers: individual HF weights → vLLM fused tensors
# ---------------------------------------------------------------------------

def _fuse_weights(individual: dict[str, object]) -> dict[str, object]:
    """Fuse individual q/k/v and gate/up weight arrays into vLLM's packed format.

    vLLM's Qwen3ForCausalLM uses:
      packed_modules_mapping = {
          "qkv_proj":    ["q_proj", "k_proj", "v_proj"],
          "gate_up_proj": ["gate_proj", "up_proj"],
      }

    The fused tensors are dim-0 concatenations in the order above.

    Args:
        individual: {vllm_unfused_name: numpy_array}, e.g.
            {"model.layers.0.self_attn.q_proj.weight": arr_q, ...}

    Returns:
        New dict with q/k/v replaced by qkv_proj and gate/up by gate_up_proj.
        Unfused weights (o_proj, down_proj) pass through unchanged.
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
    (e.g. "model.layers.0.self_attn.q_proj.weight"). We do NOT pre-fuse q/k/v → qkv_proj:
    vLLM's Qwen3 load_weights() inherits Qwen2Model.load_weights() which has a
    stacked_params_mapping that routes "q_proj" / "k_proj" / "v_proj" into the
    fused qkv_proj parameter internally. Sending pre-fused tensors would bypass
    this routing and cause a KeyError on the fused name.

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

        # Translate HF PEFT name → vLLM Qwen3 unfused name.
        # HF: "base_model.model.model.layers.0.self_attn.q_proj"
        # vLLM: "model.layers.0.self_attn.q_proj.weight"
        # NOTE: We send q_proj/k_proj/v_proj individually (NOT pre-fused as qkv_proj).
        # Qwen2Model.load_weights (inherited by Qwen3Model) uses stacked_params_mapping
        # to route these into the fused qkv_proj parameter internally.
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
    # vLLM's Qwen3 load_weights uses stacked_params_mapping to fuse q/k/v → qkv_proj
    # and gate/up → gate_up_proj internally — no pre-fusion needed here.
    vllm_engine.collective_rpc(
        "update_weights_from_numpy",
        kwargs={"weight_dict": weight_dict},
    )
