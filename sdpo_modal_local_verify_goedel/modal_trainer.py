"""
Modal SDPOTrainer for local-lean-verify: generate_only and run_sdpo_step (no verifier on Modal).

TLDR: SDPOTrainer sets up vLLM + HF model in setup(). Entrypoint runs the loop locally:
generate_only.remote() -> local parse + local verify -> run_sdpo_step.remote(payload).
Verification is done locally via local_lean_verifier; this trainer does not call any verifier.
"""

from pathlib import Path

from sdpo_modal_local_verify_goedel.config import SDPOConfig
from sdpo_modal_local_verify_goedel.sdpo_loss import compute_sdpo_loss
from sdpo_modal_local_verify_goedel.utils import collect_per_token_kl, save_run

try:
    import modal
    from sdpo_modal_local_verify_goedel.modal_app import (
        _setup_trainer,
        app,
        hf_cache_volume,
        inference_image,
        output_volume,
    )
except ImportError:
    modal = None
    app = None
    _setup_trainer = None
    inference_image = None
    hf_cache_volume = None
    output_volume = None


if modal is not None and app is not None:

    SUPPORTED_GPUS = {"A100-40GB", "A100-80GB", "H100"}

    def normalize_gpu(gpu: str) -> str:
        """Normalize GPU string to Modal form; fallback to A100-40GB if unsupported."""
        s = (gpu or "").strip().upper().replace("_", "-")
        if s in SUPPORTED_GPUS:
            return s
        if s == "A100_80GB":
            return "A100-80GB"
        return "A100-40GB"

    @app.cls(
        image=inference_image,
        gpu="A100-80GB",
        timeout=3600,
        scaledown_window=600,
        volumes={"/cache": hf_cache_volume, "/output": output_volume},
        secrets=[modal.Secret.from_name("huggingface")],
    )
    class SDPOTrainer:
        """SDPO trainer on Modal GPU. Exposes generate_only and run_sdpo_step; verification is local."""

        model_name: str = modal.parameter(default="Goedel-LM/Goedel-Prover-V2-8B")
        gpu: str = modal.parameter(default="A100-80GB")

        @modal.enter()
        def setup(self):
            _setup_trainer(self, gpu_name=self.gpu)
            self._optimizer = None

        def _generate_proof(self, config: SDPOConfig, prompt: str):
            """Generate proof with vLLM; return (raw_text, token_ids tensor)."""
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                stop=config.stop_tokens,
            )
            outputs = self.vllm_engine.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            generated_ids = self.tokenizer(
                generated_text, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            return generated_text, generated_ids

        @modal.method()
        def generate_only(self, config_dict: dict, prompt: str) -> tuple[str, list[int]]:
            """Generate one response; return (raw_text, generated_ids as list for round-trip)."""
            config = SDPOConfig(**config_dict)
            raw_text, generated_ids = self._generate_proof(config, prompt)
            return raw_text, generated_ids.tolist()

        @modal.method()
        def run_sdpo_step(self, config_dict: dict, problem: dict, payload: dict) -> dict:
            """Run one SDPO step: compute loss, backward, optimizer step. Skip if success or server_error.
            payload: base_prompt, teacher_prompt, generated_ids (list), is_success, is_server_error, plus fields for iter_log.
            Returns iter_log dict (with loss, reward, etc. if step was taken).
            """
            import torch

            config = SDPOConfig(**config_dict)
            if self._optimizer is None:
                self._optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

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

            if payload.get("is_success") or payload.get("is_server_error"):
                iter_log["loss"] = None
                iter_log["reward"] = None
                iter_log["kl_div"] = None
                iter_log["entropy"] = None
                iter_log["grad_norm"] = None
                if payload.get("is_server_error"):
                    iter_log["server_error"] = True
                return iter_log

            base_prompt = payload["base_prompt"]
            teacher_prompt = payload["teacher_prompt"]
            model_device = next(self.model.parameters()).device
            generated_ids = torch.tensor(payload["generated_ids"], dtype=torch.long, device=model_device)

            per_token_kl, reward, avg_kl, entropy = compute_sdpo_loss(
                self.model, self.tokenizer, config, base_prompt, teacher_prompt, generated_ids
            )
            loss = per_token_kl.mean()
            self._optimizer.zero_grad()
            loss.backward()

            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._optimizer.step()

            iter_log["loss"] = loss.item()
            iter_log["reward"] = reward
            iter_log["kl_div"] = avg_kl
            iter_log["entropy"] = entropy
            iter_log["grad_norm"] = grad_norm
            iter_log["feedback"] = payload.get("feedback", "")
            # Stored here; finalize_run → save_run strips this from logs.json and writes to kl/ separately
            iter_log["per_token_kl"] = collect_per_token_kl(
                per_token_kl, generated_ids, self.tokenizer
            )
            return iter_log

        @modal.method()
        def finalize_run(self, config_dict: dict, logs: dict) -> dict:
            """Save model and logs to Modal volume; return logs with model_save_path set."""
            config = SDPOConfig(**config_dict)
            metrics = logs.get("metrics", {})
            model_save_path = save_run(
                output_root=Path("/output"),
                config=config,
                logs=logs,
                metrics=metrics,
                model=self.model,
                tokenizer=self.tokenizer,
            )
            logs["model_save_path"] = model_save_path
            return logs

    def get_trainer_cls(gpu: str):
        """Return SDPOTrainer bound to the requested GPU (A100-40GB, A100-80GB, or H100)."""
        normalized = normalize_gpu(gpu)
        return SDPOTrainer.with_options(gpu=normalized)
