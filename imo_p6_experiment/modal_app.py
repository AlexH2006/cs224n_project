"""
TLDR: Modal app for local-lean-verify pipeline: inference image, volumes, and trainer setup.

Defines the Modal app (lean-sdpo-local-compile), inference image, volumes, and
_setup_trainer() for loading vLLM + HuggingFace QLoRA model. Verification runs
locally via local_lean_verifier; this app has no KiminaLeanServer or LeanVerifier.

Target model: Qwen/Qwen3.5-4B (Qwen3_5ForConditionalGeneration — natively multimodal,
used text-only here). HF loads via AutoModelForImageTextToText; vLLM loads the full
ConditionalGeneration model which wraps the CausalLM under language_model.

Weight sync: after each gradient step, merged LoRA weights are pushed into the vLLM
engine in-place via CUDA IPC (KiminaSDPOWorkerExtension + sync_lora_weights_to_vllm_kimina).
CUDA graphs remain valid — no recapture.

Volumes:
  hf_cache_volume      → /cache       — model weights
  output_volume        → /output      — training results
  compile_cache_volume → /root/.cache/vllm/torch_compile_cache
                         Persists torch.compile artifacts: reduces cold-start latency
                         from ~220s (compile + graph capture) to ~11s (graph capture only).

Used by: modal_trainer, entrypoint.
"""

import os
from typing import Any

from imo_p6_experiment.config import SDPOConfig

try:
    import modal
except ImportError:
    modal = None

if modal is None:
    app = None
    inference_image = None
    hf_cache_volume = None
    output_volume = None
    compile_cache_volume = None
else:
    app = modal.App("lean-sdpo-local-compile")

    hf_cache_volume = modal.Volume.from_name("sdpo-hf-cache", create_if_missing=True)
    # Separate volume from Kimina pipeline so local-verify results never mix with sdpo-output.
    output_volume = modal.Volume.from_name("sdpo-output-local-verify", create_if_missing=True)
    # Persists torch.compile kernel cache across cold starts.
    compile_cache_volume = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)

    inference_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch",
            "accelerate",
            "vllm>=0.8.0",
            "sentencepiece",
            "protobuf",
            "datasets",
            "matplotlib",
            "httpx",
            "bitsandbytes",
            "peft",
        )
        # transformers>=5.2.0 is required: qwen3_5 model type was added in v5.2.0 (Feb 2026).
        # Install separately after other libs so this version pin is not overridden.
        .pip_install("transformers>=5.2.0")
    )

    # Per-GPU vLLM settings. Qwen3.5-4B in bf16 needs ~8GB for weights alone,
    # so A100-40GB needs at least 0.35 to fit weights + KV cache.
    GPU_VLLM_MEMORY = {"A100-40GB": 0.35, "A100-80GB": 0.4, "H100": 0.4}
    DEFAULT_GPU_MEMORY = 0.35
    DEFAULT_MAX_MODEL_LEN = 8096

    # LoRA target modules for Qwen3.5 (Qwen/Qwen3.5-4B):
    # Standard attention (q/k/v/o) + GDN linear attention (in_proj_*, out_proj) + MLP.
    from imo_p6_experiment._weight_sync_kimina import KIMINA_LORA_TARGET_MODULES as _KIMINA_LORA_TARGET_MODULES

    def _setup_trainer(trainer_self: Any, gpu_name: str) -> None:
        """Shared setup: env, tokenizer, vLLM engine (with KiminaSDPOWorkerExtension), HF QLoRA model.

        vLLM gpu_memory_utilization is chosen by gpu_name (A100-80GB/H100 use 0.4, else 0.35).
        KiminaSDPOWorkerExtension enables in-place weight updates via collective_rpc — without
        it, the vLLM engine stays frozen at the base model throughout training.

        Sets trainer_self.{tokenizer, vllm_engine, model, lora_target_modules}.
        lora_target_modules is stored on trainer_self so run_sdpo_step can pass it to
        sync_lora_weights_to_vllm_kimina without hard-coding it in the trainer class.
        """
        os.environ["HF_HOME"] = "/cache"
        if os.environ.get("HF_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
        from vllm import LLM

        gpu_memory_utilization = GPU_VLLM_MEMORY.get(gpu_name, DEFAULT_GPU_MEMORY)
        max_model_len = DEFAULT_MAX_MODEL_LEN

        print(f"Loading model: {trainer_self.model_name}...")

        trainer_self.tokenizer = AutoTokenizer.from_pretrained(
            trainer_self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if trainer_self.tokenizer.pad_token is None:
            trainer_self.tokenizer.pad_token = trainer_self.tokenizer.eos_token

        # KiminaSDPOWorkerExtension: enables in-place weight updates after each gradient step.
        # enforce_eager=True: Qwen3.5 GDN hits AssertionError during CUDA graph capture
        # (num_cache_lines >= batch). Eager mode avoids capture and runs reliably.
        print(f"Initializing vLLM engine ({gpu_name}, gpu_memory_utilization={gpu_memory_utilization})...")
        trainer_self.vllm_engine = LLM(
            model=trainer_self.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            download_dir="/cache",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
            enable_prefix_caching=True,
            worker_extension_cls="imo_p6_experiment._weight_sync_kimina.KiminaSDPOWorkerExtension"
        )
        print("vLLM engine initialized!")

        # QLoRA: 4-bit NF4 base + LoRA on standard attn + GDN linear attn + MLP.
        # Qwen3.5 is Qwen3_5ForConditionalGeneration (natively multimodal);
        # loaded via AutoModelForImageTextToText. Text-only forward works fine.
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print("Loading HuggingFace model with 4-bit QLoRA for training...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        trainer_self.model = AutoModelForImageTextToText.from_pretrained(
            trainer_self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        trainer_self.model = prepare_model_for_kbit_training(trainer_self.model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=_KIMINA_LORA_TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        trainer_self.model = get_peft_model(trainer_self.model, lora_config)
        trainer_self.model.print_trainable_parameters()
        trainer_self.model.train()
        if hasattr(trainer_self.model, "gradient_checkpointing_enable"):
            trainer_self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled!")

        trainer_self.lora_target_modules = _KIMINA_LORA_TARGET_MODULES
        print(f"Model loaded!")
