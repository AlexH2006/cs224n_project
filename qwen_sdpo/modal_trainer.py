"""
TLDR: Modal trainer class for Qwen3.5 SDPO — vLLM generation + QLoRA training on H100.

QwenSDPOTrainer holds two model instances on the same GPU:
  1. vLLM engine   — fast batched generation (BF16, CUDA graphs on)
  2. HuggingFace   — training with backprop (QLoRA: 4-bit NF4 base + LoRA adapters)

After every gradient step, merged LoRA weights are pushed into the vLLM engine
in-place via collective_rpc (see _weight_sync.py). This closes the online RL loop:
generation quality improves with each iteration instead of staying frozen at the
base model. CUDA graphs are NOT recaptured — the in-place update preserves tensor
addresses.

Both Qwen3.5-4B and Qwen3.5-9B run on H100 (80GB). GPU is fixed at H100 because Modal
requires the gpu argument in @app.cls() to be a literal string at class-definition time.
model_name is a modal.parameter so the caller can pick 4B or 9B at runtime.

Key implementation notes:
  - vLLM: worker_cls=QwenSDPOWorker (enables in-place weight updates via collective_rpc)
  - vLLM: language_model_only=True (skip vision encoder), enforce_eager=False (CUDA graphs)
  - LoRA: covers ALL linear layers in Qwen3.5's hybrid architecture:
      Qwen3_5Attention (q/k/v/o_proj), Qwen3_5GatedDeltaNet (in_proj_*/out_proj),
      Qwen3_5MLP (gate/up/down_proj)

GPU memory budget on H100 (80GB):
  Qwen3.5-9B + QLoRA | vLLM util=0.45 → ~36GB | HF 4-bit ~5GB   | LoRA ~0.5GB  | ~38GB headroom
  Qwen3.5-4B + QLoRA | vLLM util=0.45 → ~36GB | HF 4-bit ~2.1GB | LoRA ~0.45GB | ~41GB headroom

Methods (exposed via Modal):
  generate_only(config_dict, prompt) → (raw_text, generated_ids as list)
  run_sdpo_step(config_dict, payload) → iter_log dict
  finalize_run(config_dict, logs) → logs with model_save_path
  reset_to_base(config_dict) → {"status": "ok", "model_name": ...} — reload base HF LoRA,
    sync into vLLM; used between problems in a batch so the next problem trains from
    base on the same GPU without cold start (same container when within scaledown_window).

Used by: modal_app.py (instantiates QwenSDPOTrainer at module level, passes to run_main).
"""

from pathlib import Path

from qwen_sdpo.config import SDPOConfig
from qwen_sdpo.sdpo_loss import compute_sdpo_loss
from qwen_sdpo.results import collect_per_token_kl, save_run
from qwen_sdpo._weight_sync import sync_lora_weights_to_vllm

try:
    import modal
    from qwen_sdpo._modal_infra import (
        app,
        inference_image,
        hf_cache_volume,
        output_volume,
        compile_cache_volume,
    )
except ImportError:
    modal = None
    app = None
    inference_image = None
    hf_cache_volume = None
    output_volume = None
    compile_cache_volume = None


# LoRA target modules covering the full Qwen3.5 hybrid architecture.
# Derived from modeling_qwen3_5.py:
#   Qwen3_5Attention:      q_proj, k_proj, v_proj, o_proj (8 full_attention layers)
#   Qwen3_5GatedDeltaNet:  in_proj_qkv (fused QKV), in_proj_z (gate), in_proj_b/a (decay),
#                          out_proj (output) — 24 linear_attention layers
#   Qwen3_5MLP:            gate_proj, up_proj, down_proj (all 32 layers)
_QWEN35_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _load_hf_lora_model(model_name: str):
    """Load base HuggingFace model with 4-bit QLoRA for training.

    Used by _setup_trainer (initial load) and reset_to_base (reload between
    batch problems). Returns the PEFT model; caller attaches optimizer and
    syncs to vLLM as needed.
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=_QWEN35_LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    return model


def _setup_trainer(trainer_self) -> None:
    """Load tokenizer, vLLM engine, and QLoRA HuggingFace model on the Modal GPU.

    Called inside @modal.enter() so it runs once at container startup.
    Sets trainer_self.{tokenizer, vllm_engine, model}.
    """
    import os
    from transformers import AutoTokenizer
    from vllm import LLM

    os.environ["HF_HOME"] = "/hf_cache"
    if os.environ.get("HF_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    model_name = trainer_self.model_name
    print(f"Setting up QwenSDPOTrainer: model={model_name}, gpu=H100")

    # --- Tokenizer ---
    trainer_self.tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if trainer_self.tokenizer.pad_token is None:
        trainer_self.tokenizer.pad_token = trainer_self.tokenizer.eos_token

    # --- vLLM engine ---
    # worker_cls=QwenSDPOWorker: enables in-place weight updates via collective_rpc
    #   after each gradient step. Without this, vLLM generates with the frozen base
    #   model throughout training — making online RL a no-op.
    # language_model_only=True: skip Qwen3.5's vision encoder (saves ~4GB).
    # enforce_eager=False: CUDA graphs ON — safe with CUDA 12.6 + vLLM nightly.
    #   In-place weight updates via load_weights() preserve tensor addresses, so
    #   graphs remain valid after each sync (no recapture).
    # enable_prefix_caching=True: cache the shared problem prefix across generations.
    # 0.4 × 80GB (H100) = 32GB for vLLM KV cache.
    # The QLoRA training model (4-bit 4B) peaks at ~10GB (weights + activations +
    # AdamW optimizer state on LoRA params only). The remaining ~38GB is headroom.
    print("Initializing vLLM (gpu_memory_utilization=0.4)...")
    trainer_self.vllm_engine = LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        download_dir="/hf_cache",
        gpu_memory_utilization=0.4,
        max_model_len=16384,
        language_model_only=True,
        enforce_eager=False,
        enable_prefix_caching=True,
        # worker_extension_cls: V1-native way to add collective_rpc methods to the worker.
        # Must be a fully-qualified string. vLLM mixes the extension class's methods
        # into its internal worker so they are callable via collective_rpc().
        worker_extension_cls="qwen_sdpo._weight_sync.QwenSDPOWorkerExtension",
    )
    print("vLLM engine ready.")

    # --- HuggingFace QLoRA model (shared path for setup and reset_to_base) ---
    trainer_self.model = _load_hf_lora_model(model_name)
    print("HuggingFace model ready.")


if modal is not None and app is not None:

    @app.cls(
        image=inference_image,
        gpu="H100",
        timeout=21600,  # 6 hours (multi-problem batches need longer runs)
        scaledown_window=600,
        volumes={
            "/hf_cache": hf_cache_volume,
            "/output": output_volume,
            # Persists torch.compile artifacts: reduces cold-start first-inference
            # latency from ~220s (compile + graph capture) to ~11s (graph capture only).
            "/root/.cache/vllm/torch_compile_cache": compile_cache_volume,
        },
        secrets=[modal.Secret.from_name("huggingface")],
        # Disable vLLM V1 multiprocessing so collective_rpc uses direct in-process
        # calls instead of ZMQ + msgpack. With TP=1 on a single H100 there is no
        # benefit to multiprocessing, and keeping it on forces all collective_rpc
        # kwargs (including large numpy weight arrays) through the msgpack serializer,
        # which does not support arbitrary dict[str, ndarray] payloads without
        # VLLM_ALLOW_INSECURE_SERIALIZATION. In-process mode: direct Python call,
        # zero serialization overhead, zero risk of encoding failures.
        env={"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    )
    class QwenSDPOTrainer:
        """SDPO trainer for Qwen3.5-4B / 9B on H100. Verification runs locally (not here)."""

        model_name: str = modal.parameter(default="Qwen/Qwen3.5-4B")

        @modal.enter()
        def setup(self):
            _setup_trainer(self)
            self._optimizer = None

        def _generate_proof(self, config: SDPOConfig, prompt: str):
            """Run vLLM generation for one prompt.

            Returns (raw_text, token_ids tensor, finish_reason).
            finish_reason is "stop" (EOS) or "length" (token limit hit) — the
            only reliable truncation signal regardless of output format.
            """
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
                repetition_penalty=config.repetition_penalty,
                max_tokens=config.max_new_tokens,
            )
            outputs = self.vllm_engine.generate([prompt], sampling_params)
            completion = outputs[0].outputs[0]
            generated_text = completion.text
            finish_reason = completion.finish_reason  # "stop" | "length"
            # Re-tokenize the generated text to get IDs for SDPO loss computation.
            generated_ids = self.tokenizer(
                generated_text, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            return generated_text, generated_ids, finish_reason

        @modal.method()
        def generate_only(self, config_dict: dict, prompt: str) -> tuple[str, list[int], str]:
            """Generate one response for the given prompt.

            Returns (raw_text, generated_ids as Python list, finish_reason).
            finish_reason is "stop" (natural EOS) or "length" (token limit hit).
            Lists survive Modal's serialization round-trip; tensors do not.
            """
            config = SDPOConfig(**config_dict)
            raw_text, generated_ids, finish_reason = self._generate_proof(config, prompt)
            return raw_text, generated_ids.tolist(), finish_reason

        @modal.method()
        def run_sdpo_step(self, config_dict: dict, payload: dict) -> dict:
            """Run one SDPO gradient step, sync weights to vLLM, and return the iteration log.

            Skips the gradient step (but still logs) if:
              - is_success=True (proof verified; no update needed)
              - is_server_error=True (Kimina was unavailable; skip to avoid poisoning)

            After a successful gradient step, merged LoRA weights are pushed into the
            vLLM engine via CUDA IPC so the next generate_only() call uses updated weights.

            payload keys:
              iteration, base_prompt, teacher_prompt, raw_output, extracted_block,
              full_code, verification, generated_ids (list), num_tokens,
              teacher_response_ids (list), teacher_response_mode, cot_len,
              is_success, is_server_error, feedback (optional)

            Returns iter_log dict with loss/reward/kl_div/entropy/grad_norm set.
            """
            import torch

            config = SDPOConfig(**config_dict)
            if self._optimizer is None:
                self._optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=config.learning_rate
                )

            iter_log = {
                "iteration": payload.get("iteration"),
                "student_prompt": payload.get("base_prompt"),
                "teacher_prompt": payload.get("teacher_prompt"),
                "raw_output": payload.get("raw_output"),
                "extracted_block": payload.get("extracted_block"),
                "full_code": payload.get("full_code"),
                "verification": payload.get("verification"),
                "success": payload.get("is_success", False),
                "num_tokens": payload.get("num_tokens", 0),
            }

            # Skip the gradient step when proof succeeded or a server error occurred.
            # On success, there is no failure to learn from. On server error, the
            # feedback signal is unreliable and would poison the update.
            if payload.get("is_success") or payload.get("is_server_error"):
                iter_log["loss"] = None
                iter_log["reward"] = None
                iter_log["kl_div"] = None
                iter_log["entropy"] = None
                iter_log["grad_norm"] = None
                if payload.get("is_server_error"):
                    iter_log["server_error"] = True
                return iter_log

            model_device = next(self.model.parameters()).device
            generated_ids = torch.tensor(
                payload["generated_ids"], dtype=torch.long, device=model_device
            )

            teacher_response_mode = payload.get("teacher_response_mode", "full")
            cot_len = payload.get("cot_len", 0)
            teacher_response_ids_py = payload.get("teacher_response_ids")
            use_response_slice = (
                teacher_response_mode in ("answer_only", "code_only")
                and cot_len is not None
                and teacher_response_ids_py is not None
            )

            if use_response_slice:
                teacher_response_ids = torch.tensor(
                    teacher_response_ids_py, dtype=torch.long, device=model_device
                )
                per_token_kl, reward, avg_kl, entropy = compute_sdpo_loss(
                    self.model,
                    self.tokenizer,
                    config,
                    payload["base_prompt"],
                    payload["teacher_prompt"],
                    generated_ids,
                    teacher_response_ids=teacher_response_ids,
                    cot_len=cot_len,
                )
                ids_for_kl = teacher_response_ids
            else:
                per_token_kl, reward, avg_kl, entropy = compute_sdpo_loss(
                    self.model,
                    self.tokenizer,
                    config,
                    payload["base_prompt"],
                    payload["teacher_prompt"],
                    generated_ids,
                )
                ids_for_kl = generated_ids
            loss = per_token_kl.mean()
            self._optimizer.zero_grad()
            loss.backward()

            grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.model.parameters()
                if p.grad is not None
            ) ** 0.5

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._optimizer.step()

            # Sync the updated LoRA weights into vLLM immediately after the gradient step.
            # This ensures the next generate_only() call uses the improved policy.
            # merge_adapter() is called internally, then unmerge_adapter() restores QLoRA
            # state — the model remains trainable for the next iteration.
            sync_lora_weights_to_vllm(
                self.model,
                self.vllm_engine,
                _QWEN35_LORA_TARGET_MODULES,
            )

            iter_log["loss"] = loss.item()
            iter_log["reward"] = reward
            iter_log["kl_div"] = avg_kl
            iter_log["entropy"] = entropy
            iter_log["grad_norm"] = grad_norm
            iter_log["feedback"] = payload.get("feedback", "")
            # KL records live here temporarily; save_run strips them into kl/ subdirectory.
            iter_log["per_token_kl"] = collect_per_token_kl(
                per_token_kl, ids_for_kl, self.tokenizer
            )
            return iter_log

        @modal.method()
        def finalize_run(self, config_dict: dict, logs: dict) -> dict:
            """Save model, logs, and metrics to the Modal output volume. Returns logs."""
            config = SDPOConfig(**config_dict)
            model_tag = config.model_name.split("/")[-1]
            problem_idx = (
                logs.get("problem", {}).get("problem_idx")
                or logs.get("config", {}).get("problem_idx", 0)
            )
            mode_dir = config.teacher_response_mode
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            run_dir = Path("/output") / model_tag / mode_dir / f"run_{model_tag}_{problem_idx}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            metrics = logs.get("metrics", {})
            save_run(
                run_dir=run_dir,
                cfg=config,
                logs=logs,
                metrics=metrics,
                model=self.model,
                tokenizer=self.tokenizer,
                save_kl=False,  # KL heatmaps are generated locally to save volume space
            )
            logs["model_save_path"] = str(run_dir / "final_model")
            logs["run_dir"] = str(run_dir)
            return logs

        @modal.method()
        def reset_to_base(self, config_dict: dict) -> dict:
            """Reload base HuggingFace QLoRA and sync into vLLM for the next problem.

            Used between problems in a batch so each problem trains from the same
            base model on the same GPU (no cold start). Does not tear down or
            recreate the tokenizer or vLLM engine; only the HF model and in-engine
            weights are replaced. After this call, vLLM is effectively a clean
            base HF model for the next run_main.
            """
            config = SDPOConfig(**config_dict)
            model_name = config.model_name
            print(f"reset_to_base: reloading base HF LoRA for {model_name}...")
            self.model = _load_hf_lora_model(model_name)
            self._optimizer = None
            sync_lora_weights_to_vllm(
                self.model,
                self.vllm_engine,
                _QWEN35_LORA_TARGET_MODULES,
            )
            print("reset_to_base: vLLM synced with base weights.")
            return {"status": "ok", "model_name": model_name}
