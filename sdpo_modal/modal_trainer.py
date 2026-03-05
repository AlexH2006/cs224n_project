"""
Modal SDPOTrainer class: GPU container that runs the SDPO loop via trainer_core.

TLDR: SDPOTrainer sets up vLLM + HF model in setup(), then run_sdpo() builds
verify_fn and generate_fn and calls trainer_core.run_sdpo. Used by: entrypoint.
"""

from datetime import datetime
from pathlib import Path

from sdpo_modal.config import SDPOConfig
from sdpo_modal.utils import get_field
from sdpo_modal.trainer_core import run_sdpo as run_sdpo_core

try:
    import modal
    from sdpo_modal.modal_app import (
        LeanVerifier,
        _setup_trainer,
        app,
        hf_cache_volume,
        inference_image,
        output_volume,
    )
except ImportError:
    modal = None
    app = None
    LeanVerifier = None
    _setup_trainer = None
    inference_image = None
    hf_cache_volume = None
    output_volume = None


if modal is not None and app is not None:

    @app.cls(
        image=inference_image,
        gpu="A100-40GB",
        timeout=3600,
        scaledown_window=600,
        volumes={"/cache": hf_cache_volume, "/output": output_volume},
        secrets=[modal.Secret.from_name("huggingface")],
    )
    class SDPOTrainer:
        """SDPO trainer on Modal GPU (A100-40GB). Delegates loop to trainer_core."""

        model_name: str = modal.parameter(default="AI-MO/Kimina-Prover-RL-1.7B")

        @modal.enter()
        def setup(self):
            _setup_trainer(self, gpu_memory_utilization=0.25, max_model_len=8096, gpu_name="A100-40GB")

        def _generate_proof(self, config: SDPOConfig, prompt: str):
            """Generate proof with vLLM; return (raw_text, token_ids)."""
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
        def run_sdpo(self, config_dict: dict, problem: dict) -> dict:
            """Run SDPO on one problem. verify_fn and generate_fn wire to Modal services."""
            config = SDPOConfig(**config_dict)

            debug_lean_dir = Path("/output") / config.output_dir / "debug" / "lean_server"
            debug_lean_dir.mkdir(parents=True, exist_ok=True)
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_lean_path = debug_lean_dir / f"verify_{session_id}.jsonl"

            def verify_fn(lean_code: str) -> dict:
                return LeanVerifier().verify.remote(lean_code)

            def generate_fn(prompt: str):
                return self._generate_proof(config, prompt)

            logs = run_sdpo_core(
                config=config,
                problem=problem,
                verify_fn=verify_fn,
                generate_fn=generate_fn,
                model=self.model,
                tokenizer=self.tokenizer,
                get_field=get_field,
                debug_lean_path=debug_lean_path,
                output_root=Path("/output"),
                config_dict_for_logs=config_dict,
            )
            return logs
