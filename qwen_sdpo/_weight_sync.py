"""
TLDR: LoRA-to-vLLM weight synchronization for online RL training.

Problem: vLLM and the HuggingFace QLoRA model are independent instances on the same
GPU. After each optimizer.step(), LoRA updates live only in the HF model — the vLLM
engine still generates with the original frozen base weights, making online RL a no-op.

Solution: After each gradient step, for each LoRA-wrapped layer compute the merged
bf16 weight (dequant(W_base_4bit) + lora_B @ lora_A * scale), transfer it as a CPU
numpy array through vLLM's collective_rpc mechanism, and apply it in-place inside the
vLLM worker via load_weights(). Since load_weights() overwrites values at existing
tensor addresses, CUDA graphs remain valid — no recapture needed.

Architecture:
  1. QwenSDPOWorkerExtension — a plain class (NOT a Worker subclass) registered via
     worker_extension_cls= at LLM() init time. vLLM V1's UniProcExecutor mixes this
     class's methods into its internal worker so they are callable via collective_rpc.

  2. sync_lora_weights_to_vllm() — called by the trainer after optimizer.step().
     Directly computes W_merged for each LoRA layer (no merge_adapter() needed),
     transfers weights as numpy arrays via collective_rpc, and the worker applies
     them in-place.

Flow per SDPO step:
  optimizer.step()                   # A, B matrices updated
  sync_lora_weights_to_vllm(...)     # dequant → delta → rpc → load_weights
  vllm_engine.generate(...)          # uses updated weights, no CUDA graph recapture

QLoRA constraint:
  The HF model stores base weights in 4-bit NF4. We cannot push 4-bit data into
  vLLM's bf16 weight slots. For each LoRA layer we directly compute:
    W_merged = dequant(W_base_4bit) + lora_B @ lora_A * (alpha / r)
  This avoids merge_adapter() / unmerge_adapter(), which can fail silently when
  gradient checkpointing is enabled or with certain bitsandbytes versions.

  The merged weight is moved to CPU as a numpy array and sent through
  collective_rpc. With VLLM_ENABLE_V1_MULTIPROCESSING=0 (set in @app.cls env)
  this is a direct in-process Python call — no ZMQ, no msgpack, no serialization.
  The worker converts the array back to a bfloat16 CUDA tensor via load_weights().

Used by: qwen_sdpo/modal_trainer.py (_setup_trainer and run_sdpo_step).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM


# ---------------------------------------------------------------------------
# vLLM Worker Extension (V1 native pattern)
# ---------------------------------------------------------------------------

class QwenSDPOWorkerExtension:
    """Extension class mixed into vLLM's internal worker via worker_extension_cls=.

    vLLM V1's UniProcExecutor adds these methods to its worker object so they can
    be called via llm.collective_rpc("method_name", args=(...,)).

    This is the V1-native pattern — do NOT subclass vllm.worker.worker.Worker.
    The extension class is a plain Python class; vLLM handles the mixin internally.

    Registered at LLM() init time via:
        worker_extension_cls="qwen_sdpo._weight_sync.QwenSDPOWorkerExtension"
    """

    def update_weights_from_numpy(self, weight_dict: dict) -> bool:
        """Receive merged LoRA weights as numpy arrays and apply in-place to the model.

        weight_dict: {param_name: numpy_array}
          - param_name: vLLM model parameter name, e.g. "language_model.model.layers.0.self_attn.q_proj.weight"
          - numpy_array: merged bf16 weight as a float16 numpy array (on CPU)

        Converts each array back to a bfloat16 CUDA tensor and calls load_weights() in-place.
        Tensor memory addresses do not change → CUDA graphs remain valid.
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
        """Verify a weight tensor matches an expected L2 norm (for debugging/testing).

        Returns True if the norm matches within 1e-3 tolerance.
        """
        import torch
        for name, param in self.model_runner.model.named_parameters():
            if name == param_name:
                actual = param.data.norm(2).item()
                return abs(actual - expected_checksum) < 1e-3
        return False


# ---------------------------------------------------------------------------
# Sync function: called by the trainer after optimizer.step()
# ---------------------------------------------------------------------------

def sync_lora_weights_to_vllm(
    hf_model,
    vllm_engine: "LLM",
    lora_target_modules: list[str],
) -> None:
    """Push merged LoRA weights from the HF QLoRA model into the vLLM engine in-place.

    For each LoRA-wrapped layer, directly computes:
      W_merged = dequant(W_base_4bit) + lora_B @ lora_A * (alpha / r)
    without calling merge_adapter() / unmerge_adapter(), which can fail silently
    with gradient checkpointing enabled. The merged tensor is cast to bf16,
    moved to CPU as numpy for msgpack-compatible transmission via collective_rpc,
    and applied in-place inside the vLLM worker via load_weights() — no CUDA
    graph recapture needed.

    Args:
      hf_model: the PEFT-wrapped QLoRA model (result of get_peft_model()).
      vllm_engine: the vLLM LLM() instance initialized with
        worker_extension_cls="qwen_sdpo._weight_sync.QwenSDPOWorkerExtension".
      lora_target_modules: list of module leaf names that have LoRA adapters,
        e.g. ["q_proj", "k_proj", ...].
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
        # Detect PEFT LoRA-wrapped modules by duck-typing: they have a base_layer
        # attribute AND lora_A / lora_B ModuleDicts. This works regardless of whether
        # the PEFT class is lora.layer.Linear, lora.bnb.Linear4bit, or any other subclass.
        if not (hasattr(module, "base_layer") and hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        # Check if this module's leaf name is one of our LoRA target modules.
        leaf = module_name.split(".")[-1]
        if leaf not in lora_module_names:
            continue

        # Dequantize the 4-bit base weight to bf16.
        base_layer = module.base_layer
        if Linear4bit is not None and isinstance(base_layer, Linear4bit):
            w_base = bnb_F.dequantize_4bit(
                base_layer.weight.data,
                base_layer.weight.quant_state,
            ).to(torch.bfloat16)
        else:
            w_base = base_layer.weight.data.to(torch.bfloat16)

        # Compute the active LoRA adapter's delta and fold it in.
        # detach() is required: lora_B @ lora_A carries requires_grad=True from training.
        active_adapter = module.active_adapter
        if isinstance(active_adapter, (list, tuple)):
            active_adapter = active_adapter[0]
        lora_A = module.lora_A[active_adapter].weight.detach()  # (r, in_features)
        lora_B = module.lora_B[active_adapter].weight.detach()  # (out_features, r)
        scale = module.scaling[active_adapter]                   # alpha / r

        delta = (lora_B @ lora_A).to(torch.bfloat16) * scale
        w_merged = (w_base + delta).contiguous()

        # Strip PEFT's "base_model.model." prefix and replace the bare "model." root
        # with "language_model." to match vLLM's Qwen3_5ForConditionalGeneration naming.
        #
        # HF PEFT name:  "base_model.model.model.layers.0.self_attn.q_proj"
        # After strip:   "model.layers.0.self_attn.q_proj"
        # vLLM expects:  "language_model.model.layers.0.self_attn.q_proj.weight"
        #
        # This is because vLLM loads the full Qwen3_5ForConditionalGeneration wrapper
        # (even with language_model_only=True) whose top-level module is "language_model".
        inner_name = module_name.removeprefix("base_model.model.")  # → "model.layers.0..."
        vllm_param_name = "language_model." + inner_name + ".weight"

        # Move to CPU as float16 numpy array. With VLLM_ENABLE_V1_MULTIPROCESSING=0
        # collective_rpc is a direct in-process call so no serialization occurs.
        # The worker converts back to bfloat16 CUDA tensor via load_weights().
        arr = w_merged.cpu().to(torch.float16).numpy()
        weight_dict[vllm_param_name] = arr

    if not weight_dict:
        # Diagnose: show the first 20 module names and their types.
        module_sample = [
            (n, type(m).__name__) for n, m in hf_model.named_modules()
        ][:20]
        sample_str = "\n".join(f"  {n}: {t}" for n, t in module_sample)
        raise RuntimeError(
            f"No LoRA-wrapped layers found for target modules: {lora_target_modules}.\n"
            f"(Checked for modules with base_layer + lora_A + lora_B attributes.)\n"
            f"First 20 modules:\n{sample_str}"
        )

    # Push to vLLM worker via collective_rpc.
    # weight_dict is {str: numpy_array} — fully msgpack-serializable.
    vllm_engine.collective_rpc(
        "update_weights_from_numpy",
        kwargs={"weight_dict": weight_dict},
    )
