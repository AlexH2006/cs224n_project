"""
Modal app: images, volumes, Kimina Lean Server, LeanVerifier, and trainer setup.

TLDR: Defines the Modal app and all Modal-backed services (Kimina, LeanVerifier)
plus _setup_trainer for loading vLLM + HuggingFace model (with optional LoRA for large models).
Used by: modal_trainer, entrypoint (for app).
"""

import os
from typing import Any

from sdpo_modal.config import SDPOConfig
from sdpo_modal.lean_verification import parse_kimina_response, verification_error_result

try:
    import modal
except ImportError:
    modal = None

if modal is None:
    app = None
    KiminaLeanServer = None
    LeanVerifier = None
    inference_image = None
    kimina_image = None
    hf_cache_volume = None
    output_volume = None
else:
    app = modal.App("lean-sdpo-kimina")

    hf_cache_volume = modal.Volume.from_name("sdpo-hf-cache", create_if_missing=True)
    output_volume = modal.Volume.from_name("sdpo-output", create_if_missing=True)

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

    kimina_image = modal.Image.from_registry(
        "projectnumina/kimina-lean-server:2.0.0",
    ).pip_install("httpx")

    @app.cls(
        image=kimina_image,
        cpu=8,
        memory=16384,
        timeout=1200,
        scaledown_window=600,
    )
    @modal.concurrent(max_inputs=100)
    class KiminaLeanServer:
        """High-performance Lean verification using Kimina Lean Server."""

        def _start_lean_server(self):
            import subprocess
            import time
            import httpx

            if hasattr(self, "server_proc") and self.server_proc is not None:
                try:
                    self.server_proc.kill()
                    self.server_proc.wait(timeout=5)
                except Exception:
                    pass

            env = {
                **os.environ,
                "LEAN_SERVER_HOST": "0.0.0.0",
                "LEAN_SERVER_PORT": "8000",
                "LEAN_SERVER_MAX_REPLS": "7",
                "LEAN_SERVER_LOG_LEVEL": "INFO",
            }
            self.server_proc = subprocess.Popen(
                ["python", "-m", "server"],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            print("Starting Kimina Lean Server...")
            max_wait = 60
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    with httpx.Client(timeout=5.0) as client:
                        response = client.post(
                            "http://localhost:8000/verify",
                            json={
                                "codes": [{"custom_id": "health", "proof": "example : True := trivial"}],
                                "infotree_type": "original",
                            },
                        )
                        if response.status_code == 200:
                            print(f"Kimina Lean Server ready after {time.time() - start_time:.1f}s")
                            return
                except Exception:
                    pass
                time.sleep(2)
            print(f"Warning: Kimina server may not be fully ready after {max_wait}s")

        def _ensure_server_alive(self):
            import httpx

            if self.server_proc.poll() is not None:
                print("Lean server subprocess exited, restarting...")
                self._start_lean_server()
                return
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(
                        "http://localhost:8000/verify",
                        json={
                            "codes": [{"custom_id": "health", "proof": "example : True := trivial"}],
                            "infotree_type": "original",
                        },
                    )
                    if resp.status_code == 200:
                        return
            except Exception:
                pass
            print("Lean server unresponsive, restarting...")
            self._start_lean_server()

        @modal.enter()
        def start_server(self):
            self._start_lean_server()

        @modal.method()
        def verify(self, lean_code: str, custom_id: str = "1") -> dict:
            import httpx

            self._ensure_server_alive()
            http_timeout = 60.0
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with httpx.Client(timeout=http_timeout) as client:
                        response = client.post(
                            "http://localhost:8000/verify",
                            json={"codes": [{"custom_id": custom_id, "proof": lean_code}], "infotree_type": "original"},
                        )
                        response.raise_for_status()
                        return response.json()
                except httpx.ConnectError as e:
                    if attempt < max_retries - 1:
                        self._start_lean_server()
                        continue
                    return {"error": str(e), "is_server_error": True}
                except httpx.TimeoutException as e:
                    if attempt < max_retries - 1:
                        self._start_lean_server()
                        continue
                    return {"error": f"Verification timeout: {str(e)}", "is_server_error": True}
                except Exception as e:
                    if attempt < max_retries - 1:
                        self._start_lean_server()
                        continue
                    return {"error": str(e), "is_server_error": True}

    @app.cls(
        image=inference_image,
        timeout=900,
        scaledown_window=300,
    )
    class LeanVerifier:
        """Verify Lean 4 proofs using Kimina Lean Server on Modal."""

        @modal.enter()
        def setup(self):
            print("LeanVerifier initialized - will use Kimina Lean Server")

        @modal.method()
        def verify(self, lean_code: str) -> dict:
            import time

            try:
                t0 = time.time()
                server = KiminaLeanServer()
                result = server.verify.remote(lean_code)
                verifier_wall_s = time.time() - t0
                return parse_kimina_response(result, lean_code, verifier_wall_s)
            except Exception as e:
                return verification_error_result(
                    lean_code, f"Verification error: {str(e)}", is_server_error=True, verifier_wall_s=0.0
                )

    def _setup_trainer(
        trainer_self: Any,
        gpu_memory_utilization: float,
        max_model_len: int,
        gpu_name: str,
    ) -> None:
        """Shared setup: env, tokenizer, vLLM engine, HuggingFace model (with optional LoRA for 7B+)."""
        os.environ["HF_HOME"] = "/cache"
        if os.environ.get("HF_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from vllm import LLM

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

        print(f"Initializing vLLM engine ({gpu_name} config)...")
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
