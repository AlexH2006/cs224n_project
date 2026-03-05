"""
Modal app for local-lean-verify pipeline: inference image and trainer setup only (no Kimina).

TLDR: Defines the Modal app (lean-sdpo-local-compile), inference image, volumes, and
_setup_trainer for loading vLLM + HuggingFace model. Verification runs locally via
local_lean_verifier; this app has no KiminaLeanServer or LeanVerifier.
Used by: modal_trainer, entrypoint.
"""

import os
from typing import Any

from sdpo_modal_local_verify_goedel.config import SDPOConfig

try:
    import modal
except ImportError:
    modal = None

if modal is None:
    app = None
    inference_image = None
    hf_cache_volume = None
    output_volume = None
else:
    app = modal.App("lean-sdpo-local-compile")

    hf_cache_volume = modal.Volume.from_name("sdpo-hf-cache", create_if_missing=True)
    # Separate volume from Kimina pipeline so local-verify results never mix with sdpo-output
    output_volume = modal.Volume.from_name("sdpo-output-local-verify", create_if_missing=True)

    inference_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch",
            "transformers>=4.40.0",
            "accelerate",
            "vllm>=0.6.0",
            "sentencepiece",
            "protobuf",
            "datasets",
            "matplotlib",
            "httpx",
            "bitsandbytes",
            "peft",
        )
    )

    # Per-GPU vLLM settings: A100-40GB needs 0.45+ for the 8B model (weights ~16GB,
    # leaving ~24GB; 0.25 = only 10GB for KV cache which is not enough after model load).
    GPU_VLLM_MEMORY = {"A100-40GB": 0.45, "A100-80GB": 0.4, "H100": 0.4}
    DEFAULT_GPU_MEMORY = 0.45
    DEFAULT_MAX_MODEL_LEN = 8096

    def _setup_trainer(trainer_self: Any, gpu_name: str) -> None:
        """Shared setup: env, tokenizer, vLLM engine, HuggingFace model (with optional LoRA for 7B+).
        vLLM gpu_memory_utilization is chosen by gpu_name (A100-80GB/H100 use 0.4, else 0.25)."""
        os.environ["HF_HOME"] = "/cache"
        if os.environ.get("HF_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from vllm import LLM

        gpu_memory_utilization = GPU_VLLM_MEMORY.get(gpu_name, DEFAULT_GPU_MEMORY)
        max_model_len = DEFAULT_MAX_MODEL_LEN

        print(f"Loading model: {trainer_self.model_name}...")
        trainer_self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer_self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        trainer_self.tokenizer = AutoTokenizer.from_pretrained(
            trainer_self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if trainer_self.tokenizer.pad_token is None:
            trainer_self.tokenizer.pad_token = trainer_self.tokenizer.eos_token

        print(f"Initializing vLLM engine ({gpu_name}, gpu_memory_utilization={gpu_memory_utilization})...")
        trainer_self.vllm_engine = LLM(
            model=trainer_self.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            download_dir="/cache",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        print("vLLM engine initialized!")

        is_large_model = (
            ("8B" in trainer_self.model_name or "7B" in trainer_self.model_name)
            and "1.7B" not in trainer_self.model_name
        )

        if is_large_model:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            print("Loading HuggingFace model with 4-bit quantization + LoRA for training...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            trainer_self.model = AutoModelForCausalLM.from_pretrained(
                trainer_self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            trainer_self.model = prepare_model_for_kbit_training(trainer_self.model)
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            trainer_self.model = get_peft_model(trainer_self.model, lora_config)
            trainer_self.model.print_trainable_parameters()
        else:
            trainer_self.model = AutoModelForCausalLM.from_pretrained(
                trainer_self.model_name,
                torch_dtype=trainer_self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        trainer_self.model.train()
        if hasattr(trainer_self.model, "gradient_checkpointing_enable"):
            trainer_self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled!")
        print(f"Model loaded on {trainer_self.device}!")
