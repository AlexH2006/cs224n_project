"""
Test-Time RL for Lean Theorem Proving (SDPO) on Modal

This script runs SDPO (Self-Distilled Policy Optimization) on Modal with:
- Model inference on GPU via vLLM for fast generation
- Lean verification via Kimina server (also on Modal)
- Fully model-agnostic: works with any HuggingFace causal LM
- Fully dataset-agnostic: auto-detects common field names across datasets
- Single problem test-time RL with iterative feedback
- Comprehensive logging and visualization

SDPO Architecture:
    The key insight of SDPO is that the same model plays two roles:
    
    - STUDENT: Generates proofs given ONLY the problem statement.
              This is what we're training - it must learn to solve problems
              without seeing previous failed attempts.
    
    - TEACHER: Provides the distillation target given problem + feedback.
              The teacher has "hindsight" - it sees what went wrong in
              previous attempts and can provide a better target distribution.
    
    The student learns by minimizing KL divergence to the teacher's distribution.
    Over iterations, the student internalizes the feedback without needing it
    at generation time.

Supported Datasets (auto-detected field names):
    - cat-searcher/minif2f-lean4 (lean4_code, header)
    - amitayusht/PutnamBench (lean4_statement, informal_statement)
    - deepmind/math (problem, solution)
    - Custom datasets with: statement, code, theorem, problem_statement, etc.

Supported Models:
    - AI-MO/Kimina-Prover-RL-1.7B (default, thinking model)
    - Goedel-LM/Goedel-Prover-V2-8B
    - Any HuggingFace causal LM with chat template support

Usage:
    # Run on Modal with default settings (miniF2F dataset, Kimina model)
    modal run lean_sdpo_modal.py --problem-idx 0

    # Custom dataset (PutnamBench)
    modal run lean_sdpo_modal.py --dataset amitayusht/PutnamBench --dataset-split train --problem-idx 0

    # Custom model
    modal run lean_sdpo_modal.py --model Goedel-LM/Goedel-Prover-V2-8B --problem-idx 5

    # More iterations with custom temperature
    modal run lean_sdpo_modal.py --max-iterations 10 --temperature 0.7 --problem-idx 0

    # Override field names for custom datasets
    modal run lean_sdpo_modal.py --dataset my/custom-dataset --theorem-field my_theorem_code --problem-idx 0

    # Custom system prompt
    modal run lean_sdpo_modal.py --system-prompt "You are an expert mathematician." --problem-idx 0
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SDPOConfig:
    """Configuration for SDPO on Modal."""
    # Model settings
    model_name: str = "AI-MO/Kimina-Prover-RL-1.7B"
    
    # Dataset settings
    dataset_name: str = "cat-searcher/minif2f-lean4"
    dataset_subset: Optional[str] = None
    dataset_split: str = "test"
    problem_idx: int = 0
    
    # Dataset field mapping (for dataset-agnostic loading)
    # These are lists of possible field names, tried in order
    theorem_fields: list = field(default_factory=lambda: [
        "lean4_code", "formal_statement", "lean4_statement", 
        "statement", "code", "theorem", "problem_statement"
    ])
    informal_fields: list = field(default_factory=lambda: [
        "informal_prefix", "problem", "informal_statement",
        "natural_language", "description", "question", "informal"
    ])
    header_fields: list = field(default_factory=lambda: [
        "header", "imports", "preamble", "prefix"
    ])
    id_fields: list = field(default_factory=lambda: [
        "problem_id", "name", "id", "idx", "index"
    ])
    
    # Generation settings
    max_new_tokens: int = 8192  # Increased for thinking models that need more tokens
    temperature: float = 0.6
    top_p: float = 0.95
    stop_tokens: list = field(default_factory=lambda: [
        "<|im_end|>", "<|endoftext|>", "</s>", "<|end|>", 
        "[/INST]", "```\n\n", "<|eot_id|>"
    ])
    
    # Test-time RL settings
    max_iterations: int = 5
    learning_rate: float = 1e-5
    distillation_topk: int = 20
    
    # Prompt customization
    system_prompt: str = "You are an expert Lean 4 theorem prover. Output proof tactics that can replace `sorry`."
    
    # Default Lean header (used when dataset doesn't provide one and model doesn't generate imports)
    default_header: str = """import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat"""
    
    # Feedback mode
    feedback_include_failed_proof: bool = False
    # Template for individual error (simple format, no "Attempt N" prefix)
    feedback_attempt_template: str = """- Error: {feedback}
  Failed proof: {failed_proof}"""
    feedback_attempt_template_errors_only: str = """- {feedback}"""
    feedback_separator: str = "\n"
    
    # Output settings
    output_dir: str = "test-time-SDPO"


# ============================================================================
# Modal App Definition
# ============================================================================

try:
    import modal
    
    app = modal.App("lean-sdpo-ttt")
    
    # Volume for HF cache (persistent across runs)
    hf_cache_volume = modal.Volume.from_name("sdpo-hf-cache", create_if_missing=True)
    
    # Volume for output logs (persistent)
    output_volume = modal.Volume.from_name("sdpo-output", create_if_missing=True)
    
    # Image for inference + training (GPU) - includes vLLM for fast generation
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
            "bitsandbytes",  # For 8-bit quantization of large models
            "peft",  # For LoRA training with quantized models
        )
    )
    
    # Image for Kimina Lean Server - uses official Docker image with pre-built Mathlib
    # Note: The Docker image already has Python installed, so we don't use add_python
    kimina_image = modal.Image.from_registry(
        "projectnumina/kimina-lean-server:2.0.0",
    ).pip_install("httpx")

    # ========================================================================
    # Kimina Lean Server (High-performance Lean verification)
    # ========================================================================
    
    @app.cls(
        image=kimina_image,
        cpu=8,
        memory=16384,
        timeout=600,
        scaledown_window=300,
        allow_concurrent_inputs=100,
    )
    class KiminaLeanServer:
        """High-performance Lean verification using Kimina Lean Server."""
        
        @modal.enter()
        def start_server(self):
            """Start the Kimina Lean Server."""
            import subprocess
            import os
            import time
            import httpx
            
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
            
            # Wait for server to be ready with health check
            max_wait = 60
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    with httpx.Client(timeout=5.0) as client:
                        # Try a simple verification to check if server is ready
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
        
        @modal.method()
        def verify(self, lean_code: str, custom_id: str = "1") -> dict:
            """Verify Lean code using the Kimina server."""
            import httpx
            import time
            
            max_retries = 10
            retry_delay = 3
            for attempt in range(max_retries):
                try:
                    with httpx.Client(timeout=180.0) as client:
                        response = client.post(
                            "http://localhost:8000/verify",
                            json={
                                "codes": [{"custom_id": custom_id, "proof": lean_code}],
                                "infotree_type": "original",
                            },
                        )
                        response.raise_for_status()
                        return response.json()
                except httpx.ConnectError as e:
                    # Connection refused - server not ready yet
                    if attempt < max_retries - 1:
                        print(f"Kimina verify attempt {attempt + 1} failed: {e}, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    return {"error": str(e), "is_server_error": True}
                except httpx.TimeoutException as e:
                    # Timeout - server overloaded or proof too complex
                    if attempt < max_retries - 1:
                        print(f"Kimina verify timeout on attempt {attempt + 1}, retrying...")
                        time.sleep(retry_delay)
                        continue
                    return {"error": f"Verification timeout: {str(e)}", "is_server_error": True}
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Kimina verify attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(retry_delay)
                        continue
                    return {"error": str(e), "is_server_error": True}
    
    # ========================================================================
    # Lean Verification Service (uses Kimina server)
    # ========================================================================
    
    @app.cls(
        image=inference_image,
        timeout=300,
        scaledown_window=300,
    )
    class LeanVerifier:
        """Verify Lean 4 proofs using Kimina Lean Server on Modal."""
        
        @modal.enter()
        def setup(self):
            """Initialize the verifier."""
            print("LeanVerifier initialized - will use Kimina Lean Server")
        
        @modal.method()
        def verify(self, lean_code: str) -> dict:
            """Verify Lean code using Kimina Lean Server."""
            try:
                server = KiminaLeanServer()
                result = server.verify.remote(lean_code)
                
                if "error" in result:
                    is_server_error = result.get("is_server_error", False)
                    return {
                        "success": False,
                        "complete": False,
                        "has_sorry": "sorry" in lean_code.lower(),
                        "feedback": f"Kimina server error: {result['error']}",
                        "errors": [result["error"]],
                        "messages": [],
                        "sorries": [],
                        "source": "kimina",
                        "is_server_error": is_server_error,
                    }
                
                if "results" in result and len(result["results"]) > 0:
                    r = result["results"][0]
                    messages = r.get("messages", []) or []
                    sorries = r.get("sorries", []) or []
                    status = r.get("status", "")
                    
                    errors = []
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get("severity") == "error":
                            errors.append(msg.get("data", str(msg)))
                    
                    has_error = len(errors) > 0 or status == "error"
                    has_sorry = len(sorries) > 0 or "sorry" in lean_code.lower()
                    
                    return {
                        "success": not has_error,
                        "complete": not has_error and not has_sorry,
                        "has_sorry": has_sorry,
                        "feedback": "\n".join(errors) if errors else "",
                        "errors": errors,
                        "messages": [str(m) for m in messages],
                        "sorries": [str(s) for s in sorries],
                        "source": "kimina",
                        "is_server_error": False,
                    }
                
                return {
                    "success": False,
                    "complete": False,
                    "has_sorry": "sorry" in lean_code.lower(),
                    "feedback": "Unexpected response format from Kimina server",
                    "errors": ["Unexpected response format"],
                    "messages": [],
                    "sorries": [],
                    "source": "kimina",
                    "is_server_error": True,
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "complete": False,
                    "has_sorry": "sorry" in lean_code.lower(),
                    "feedback": f"Verification error: {str(e)}",
                    "errors": [str(e)],
                    "messages": [],
                    "sorries": [],
                    "source": "kimina",
                    "is_server_error": True,
                }

    # ========================================================================
    # GPU Configuration Presets
    # ========================================================================
    
    GPU_CONFIGS = {
        "A100-40GB": {
            "gpu": "A100-40GB",
            "vllm_gpu_memory_utilization": 0.25,
            "vllm_max_model_len": 4096,
            "description": "Default: A100 40GB, suitable for 1-2B models",
        },
        "A100-80GB": {
            "gpu": "A100-80GB",
            "vllm_gpu_memory_utilization": 0.45,
            "vllm_max_model_len": 8192,
            "description": "A100 80GB, suitable for 7-8B models",
        },
        "H100": {
            "gpu": "H100",
            "vllm_gpu_memory_utilization": 0.50,
            "vllm_max_model_len": 8192,
            "description": "H100 80GB, best performance for large models",
        },
        "A10G": {
            "gpu": "A10G",
            "vllm_gpu_memory_utilization": 0.20,
            "vllm_max_model_len": 2048,
            "description": "A10G 24GB, budget option for small models only",
        },
    }

    # ========================================================================
    # SDPO Training Service - Shared Setup Logic
    # ========================================================================
    
    def _setup_trainer(trainer_self, gpu_memory_utilization: float, max_model_len: int, gpu_name: str):
        """Shared setup logic for all GPU configurations."""
        import os
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
        
        # HuggingFace model for training (gradient computation)
        # For large models (7B+), use 4-bit quantization + LoRA for trainable gradients
        # Check for models >= 7B (but not 1.7B which contains "7B" substring)
        is_large_model = ("8B" in trainer_self.model_name or "7B" in trainer_self.model_name) and "1.7B" not in trainer_self.model_name
        
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
            
            # Prepare model for k-bit training
            trainer_self.model = prepare_model_for_kbit_training(trainer_self.model)
            
            # Add LoRA adapters for training
            lora_config = LoraConfig(
                r=16,  # LoRA rank
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
        
        # Enable gradient checkpointing to reduce memory usage
        if hasattr(trainer_self.model, "gradient_checkpointing_enable"):
            trainer_self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled!")
        
        print(f"Model loaded on {trainer_self.device}!")

    # ========================================================================
    # SDPO Training Service (GPU) - A100-40GB (default, for 1-2B models)
    # ========================================================================
    
    @app.cls(
        image=inference_image,
        gpu="A100-40GB",
        timeout=3600,
        scaledown_window=600,
        volumes={"/cache": hf_cache_volume, "/output": output_volume},
        secrets=[modal.Secret.from_name("huggingface")],
    )
    class SDPOTrainer:
        """SDPO trainer running on Modal GPU (A100-40GB for 1-2B models)."""
        
        model_name: str = modal.parameter(default="AI-MO/Kimina-Prover-RL-1.7B")
        
        @modal.enter()
        def setup(self):
            _setup_trainer(self, gpu_memory_utilization=0.25, max_model_len=4096, gpu_name="A100-40GB")
        
        @staticmethod
        def _get_field(data: dict, field_names: list, default: str = "") -> str:
            """Get a field from data dict, trying multiple possible field names.
            
            This enables dataset-agnostic loading by supporting various naming conventions.
            """
            for field_name in field_names:
                if field_name in data and data[field_name]:
                    value = data[field_name]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        return str(value[0])
            return default
        
        @modal.method()
        def run_sdpo(
            self,
            config_dict: dict,
            problem: dict,
            verifier_results: list[dict] | None = None,
        ) -> dict:
            """
            Run SDPO test-time RL on a single problem.
            
            Returns comprehensive results including:
            - All iteration logs
            - Loss curve data
            - Gradient norms
            - Entropy values
            - Final model state (if successful)
            """
            import torch
            import torch.nn.functional as F
            from datetime import datetime
            
            config = SDPOConfig(**config_dict)
            
            metrics = {
                "iterations": [],
                "losses": [],
                "rewards": [],
                "kl_divs": [],
                "entropies": [],
                "grad_norms": [],
                "timestamps": [],
            }
            
            logs = {
                "problem": problem,
                "config": config_dict,
                "iteration_logs": [],
                "start_time": datetime.now().isoformat(),
            }
            
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
            )
            
            feedback_history: list[tuple[str, str]] = []
            best_proof = None
            
            # Base prompt (problem statement only) - used by STUDENT model for generation
            # This never changes across iterations
            base_prompt = self._create_base_prompt(config, problem)
            
            for iteration in range(config.max_iterations):
                iter_start = time.time()
                print(f"\n--- Iteration {iteration + 1}/{config.max_iterations} ---")
                
                # SDPO Architecture:
                # - STUDENT model: Always generates from base_prompt (problem only)
                # - TEACHER model: Uses feedback_prompt (problem + error feedback) for KL target
                # The student learns to produce better proofs by distilling from the teacher
                # who has access to feedback about what went wrong
                
                # Student always generates from base prompt (no feedback in generation context)
                raw_output, generated_ids = self._generate_proof(config, base_prompt)
                
                print(f"  Generated {len(raw_output)} chars")
                
                # Check for degenerate output (model stuck in loop)
                is_degenerate = self._is_degenerate_output(raw_output)
                
                # Check for truncated output (has <think> but no </think>)
                is_truncated = "<think>" in raw_output and "</think>" not in raw_output
                
                tactics = self._extract_proof_tactics(raw_output)
                # Use dataset-agnostic field extraction
                lean4_code = self._get_field(problem, config.theorem_fields)
                header = self._get_field(problem, config.header_fields)
                full_code = self._create_full_lean_code(config, lean4_code, tactics, header)
                
                # Verify with retry logic for server errors
                max_verify_retries = 3
                verification = None
                for verify_attempt in range(max_verify_retries):
                    verification = LeanVerifier().verify.remote(full_code)
                    if not verification.get("is_server_error", False):
                        break
                    if verify_attempt < max_verify_retries - 1:
                        print(f"  Server error on verification, retrying ({verify_attempt + 1}/{max_verify_retries})...")
                        time.sleep(5)
                
                is_success = verification["success"] and verification["complete"]
                is_server_error = verification.get("is_server_error", False)
                
                # Check if the extracted tactics is just "sorry" - this should be treated as a failure
                # even if the Lean code compiles (since sorry is a placeholder, not a real proof)
                is_sorry_tactic = tactics.strip().lower() == "sorry"
                if is_sorry_tactic:
                    is_success = False
                    verification["complete"] = False
                    verification["has_sorry"] = True
                    # Provide specific feedback based on why we got "sorry"
                    if is_degenerate:
                        verification["feedback"] = "Output got stuck in a reasoning loop. Focus on the proof strategy and output tactics directly without excessive deliberation."
                        verification["is_degenerate"] = True
                    elif is_truncated:
                        verification["feedback"] = "Output was truncated before completing. Be more concise in reasoning and output the proof tactics sooner."
                        verification["is_truncated"] = True
                    else:
                        verification["feedback"] = "No valid proof tactics found. Output actual Lean 4 tactics in a ```lean4 code block."
                
                if is_server_error:
                    print(f"  Verification: SERVER ERROR (will not count as failed proof)")
                elif is_degenerate:
                    print(f"  Verification: FAILED (model stuck in reasoning loop)")
                elif is_truncated:
                    print(f"  Verification: FAILED (output truncated, no tactics produced)")
                elif is_sorry_tactic:
                    print(f"  Verification: FAILED (no valid tactics extracted)")
                else:
                    print(f"  Verification: {'SUCCESS' if is_success else 'FAILED'}")
                
                # Create teacher prompt with current feedback history
                # (even if we don't use it for training, we log it for analysis)
                current_teacher_prompt = self._create_feedback_prompt(
                    config, problem, feedback_history
                ) if feedback_history else None
                
                iter_log = {
                    "iteration": iteration + 1,
                    "student_prompt": base_prompt,  # Full prompt for student (problem only)
                    "teacher_prompt": current_teacher_prompt,  # Full prompt for teacher (problem + feedback)
                    "raw_output": raw_output,
                    "extracted_tactics": tactics,
                    "full_code": full_code,
                    "verification": verification,
                    "success": is_success,
                }
                
                if is_success:
                    best_proof = tactics
                    iter_log["loss"] = None
                    iter_log["reward"] = None
                    iter_log["kl_div"] = None
                    iter_log["entropy"] = None
                    iter_log["grad_norm"] = None
                    logs["iteration_logs"].append(iter_log)
                    
                    metrics["iterations"].append(iteration + 1)
                    metrics["losses"].append(0.0)
                    metrics["rewards"].append(1.0)
                    metrics["kl_divs"].append(0.0)
                    metrics["entropies"].append(0.0)
                    metrics["grad_norms"].append(0.0)
                    metrics["timestamps"].append(time.time() - iter_start)
                    
                    print(f"  ✓ Proof found!")
                    break
                
                # Skip SDPO training if verification failed due to server error
                # (we don't want to train on unverified proofs)
                if is_server_error:
                    iter_log["loss"] = None
                    iter_log["reward"] = None
                    iter_log["kl_div"] = None
                    iter_log["entropy"] = None
                    iter_log["grad_norm"] = None
                    iter_log["server_error"] = True
                    logs["iteration_logs"].append(iter_log)
                    print(f"  Skipping SDPO update due to server error")
                    continue
                
                # Generate appropriate feedback message
                if verification.get("has_sorry"):
                    feedback = verification["feedback"] or "Forbidden to output `sorry` tactic. You must provide actual proof tactics that complete the proof."
                else:
                    feedback = verification["feedback"] or "Proof verification failed"
                feedback_history.append((feedback, tactics))
                
                # Create teacher prompt with feedback history
                # TEACHER model sees: problem + all previous failed attempts + error messages
                # This allows the teacher to provide a better target distribution
                teacher_prompt = self._create_feedback_prompt(
                    config, problem, feedback_history
                )
                
                print(f"  SDPO: Student sees problem only, Teacher sees {len(feedback_history)} feedback(s)")
                
                per_token_kl, reward, avg_kl, entropy = self._compute_sdpo_loss(
                    config, base_prompt, teacher_prompt, generated_ids
                )
                
                loss = per_token_kl.mean()
                optimizer.zero_grad()
                loss.backward()
                
                grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                loss_val = loss.item()
                
                print(f"  Loss: {loss_val:.4f}, Reward: {reward:.4f}, KL: {avg_kl:.4f}")
                print(f"  Entropy: {entropy:.4f}, Grad norm: {grad_norm:.4f}")
                
                iter_log["teacher_prompt"] = teacher_prompt  # Full prompt for teacher (problem + feedback)
                iter_log["loss"] = loss_val
                iter_log["reward"] = reward
                iter_log["kl_div"] = avg_kl
                iter_log["entropy"] = entropy
                iter_log["grad_norm"] = grad_norm
                iter_log["feedback"] = feedback
                logs["iteration_logs"].append(iter_log)
                
                metrics["iterations"].append(iteration + 1)
                metrics["losses"].append(loss_val)
                metrics["rewards"].append(reward)
                metrics["kl_divs"].append(avg_kl)
                metrics["entropies"].append(entropy)
                metrics["grad_norms"].append(grad_norm)
                metrics["timestamps"].append(time.time() - iter_start)
            
            logs["end_time"] = datetime.now().isoformat()
            logs["success"] = best_proof is not None
            logs["best_proof"] = best_proof
            logs["metrics"] = metrics
            
            # Save results, plots, and model weights
            # NOTE: Model weights are NOT reset between problems - this is intentional!
            # SDPO is a test-time training method where the model accumulates learning
            # across problems, improving its ability to solve future problems.
            model_save_path = self._save_results(config, logs, metrics)
            logs["model_save_path"] = model_save_path
            
            return logs
        
        def _create_base_prompt(self, config: SDPOConfig, theorem: dict) -> str:
            """Create base prompt WITHOUT feedback (for STUDENT model).
            
            The student prompt contains ONLY the formal theorem statement.
            No informal description is included - the student must learn to
            solve problems from the formal statement alone.
            
            Uses dataset-agnostic field extraction to support various dataset formats.
            Works with thinking models that output <think>...</think> then the answer.
            """
            # Use dataset-agnostic field extraction
            lean4_code = self._get_field(theorem, config.theorem_fields)
            header = self._get_field(theorem, config.header_fields)
            has_header = bool(header.strip())
            
            # Student prompt: formal theorem only (no informal description)
            user_content = f"Prove the following Lean 4 theorem.\n\n"
            user_content += f"```lean4\n{lean4_code}\n```\n\n"
            user_content += "After your reasoning, output ONLY the proof tactics (not the full theorem) in a ```lean4 code block. "
            user_content += "The tactics should replace `sorry`. "
            
            if has_header:
                # Dataset provides imports, so model should not include them
                user_content += "Do NOT include `import`, `theorem`, or `:= sorry` in your final answer."
            else:
                # Dataset doesn't provide imports, model should include necessary ones
                user_content += "Include any necessary `import` statements at the beginning of your code block. Do NOT include `theorem` or `:= sorry` in your final answer."
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": user_content}
                ]
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except:
                    prompt = f"System: {config.system_prompt}\n\nUser: {user_content}\n\nAssistant:"
            else:
                prompt = f"System: {config.system_prompt}\n\nUser: {user_content}\n\nAssistant:"
            
            return prompt
        
        def _create_feedback_prompt(
            self,
            config: SDPOConfig,
            theorem: dict,
            feedback_history: list[tuple[str, str]],
        ) -> str:
            """Create prompt WITH accumulated feedback (for TEACHER model).
            
            The teacher prompt includes:
            - The formal theorem statement
            - The informal problem description (if available)
            - A summary of all previous failed attempts with their errors
            
            This gives the teacher "hindsight" to provide a better target distribution.
            """
            # Use dataset-agnostic field extraction
            lean4_code = self._get_field(theorem, config.theorem_fields)
            informal = self._get_field(theorem, config.informal_fields)
            header = self._get_field(theorem, config.header_fields)
            has_header = bool(header.strip())
            
            user_content = f"Prove the following Lean 4 theorem.\n\n"
            
            # Teacher gets informal description (student does not)
            if informal:
                user_content += f"Problem: {informal}\n\n"
            
            user_content += f"```lean4\n{lean4_code}\n```\n\n"
            
            # Add feedback: list of errors from previous attempts
            if feedback_history:
                user_content += "Previous errors:\n"
                
                # Add each error (simple bullet list)
                if config.feedback_include_failed_proof:
                    blocks = [
                        config.feedback_attempt_template.format(feedback=f, failed_proof=p)
                        for f, p in feedback_history
                    ]
                else:
                    blocks = [
                        config.feedback_attempt_template_errors_only.format(feedback=f)
                        for f, p in feedback_history
                    ]
                user_content += config.feedback_separator.join(blocks)
                user_content += "\n\nAvoid these errors."
            
            if has_header:
                user_content += "\n\nProvide corrected proof tactics:"
            else:
                user_content += "\n\nProvide corrected proof tactics (include any necessary imports):"
            
            # Use configurable system prompt
            feedback_system_prompt = config.system_prompt + " After reasoning, output the proof tactics in a ```lean4 code block."
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": feedback_system_prompt},
                    {"role": "user", "content": user_content}
                ]
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except:
                    prompt = f"System: {feedback_system_prompt}\n\nUser: {user_content}\n\nAssistant:"
            else:
                prompt = f"System: {feedback_system_prompt}\n\nUser: {user_content}\n\nAssistant:"
            
            return prompt
        
        def _generate_proof(self, config: SDPOConfig, prompt: str) -> tuple[str, "torch.Tensor"]:
            """Generate a proof using vLLM for fast inference.
            
            Uses configurable stop tokens to support various model architectures.
            """
            import torch
            from vllm import SamplingParams
            
            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                stop=config.stop_tokens,
            )
            
            outputs = self.vllm_engine.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            # Get token IDs for SDPO loss computation
            # We need to tokenize the generated text to get IDs for the HF model
            generated_ids = self.tokenizer(
                generated_text, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
            
            return generated_text, generated_ids
        
        def _is_degenerate_output(self, output: str, threshold: int = 5) -> bool:
            """Detect if the model output is degenerate (stuck in a loop).
            
            Returns True if:
            1. Any phrase of 5+ words appears more than `threshold` times
            2. Common looping patterns are detected (e.g., "Wait," appearing many times)
            
            This indicates the model got stuck repeating itself.
            """
            # Quick check for common looping indicators
            # Count occurrences of "Wait," which is a common loop pattern
            wait_count = output.count("Wait,")
            if wait_count >= threshold:
                return True
            
            # Count occurrences of repeated sentence starters
            for pattern in ["the theorem statement is", "We need to prove", "Let me", "I need to"]:
                if output.lower().count(pattern) >= threshold * 2:
                    return True
            
            # Split into words for phrase-based detection
            words = output.split()
            if len(words) < 30:
                return False
            
            # Check for repeated phrases of various lengths (5 to 20 words)
            for phrase_len in [5, 8, 10, 15, 20]:
                if len(words) < phrase_len:
                    continue
                phrase_counts = {}
                for i in range(len(words) - phrase_len):
                    phrase = " ".join(words[i:i + phrase_len])
                    phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                
                # If any phrase appears more than threshold times, it's degenerate
                max_count = max(phrase_counts.values()) if phrase_counts else 0
                if max_count >= threshold:
                    return True
            
            return False
        
        def _extract_tactics_from_code_block(self, block: str) -> str:
            """Extract tactics from a single code block, filtering out non-tactic lines.
            
            Preserves import/open/set_option statements so they can be used as header
            when the dataset doesn't provide one.
            """
            lines = []
            for line in block.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                # Preserve imports, open, set_option for header extraction
                if stripped.startswith("import "):
                    lines.append(stripped)
                    continue
                if stripped.startswith("open "):
                    lines.append(stripped)
                    continue
                if stripped.startswith("set_option "):
                    lines.append(stripped)
                    continue
                # Skip theorem/lemma declarations
                if stripped.startswith("theorem ") or stripped.startswith("lemma "):
                    continue
                # Skip parameter declarations like "(x y : ℤ)" or "(x y : ℤ) :" or just "(x y : ℤ) :"
                if stripped.startswith("(") and ":" in stripped:
                    # This handles (x : T), (x y : T), (x y : T) :, etc.
                    continue
                # Skip conclusion lines with := sorry
                if ":= sorry" in stripped:
                    continue
                # Skip lines that are just the theorem statement
                if re.search(r"^\d+\s*\*.*:=\s*(by|sorry)", stripped):
                    continue
                if re.search(r"≠\s*\d+\s*:=", stripped):
                    continue
                # Skip type signature lines like "4 * x^3 - 7 * y^3 ≠ 2003 :=" 
                if re.search(r"[≠=<>]\s*\d+\s*:=", stripped):
                    continue
                # Skip lines that look like theorem conclusions
                if stripped.endswith(":=") or stripped.endswith(":= by"):
                    continue
                # Skip lines that look like type annotations without tactics (e.g., "4 * x^3 - 7 * y^3 ≠ 2003")
                if re.search(r"^\d+\s*\*.*[≠=<>]", stripped) and ":=" not in stripped and "by" not in stripped.lower():
                    continue
                lines.append(stripped)
            return "\n".join(lines)
        
        def _extract_proof_tactics(self, output: str) -> str:
            """Extract proof tactics from model output.
            
            Kimina-Prover outputs <think>reasoning</think> then the answer.
            We extract tactics from:
            1. Content after </think> (preferred)
            2. Code blocks within <think> if no </think>
            3. Any code blocks in the output
            
            Also detects degenerate/looping outputs where the model gets stuck.
            """
            output = output.strip()
            tactics = None
            
            # Detect degenerate/looping output
            # If the same phrase appears many times, the model got stuck
            if self._is_degenerate_output(output):
                print("  WARNING: Detected degenerate/looping output from model")
                return "sorry"
            
            # Strategy 1: If there's a complete thinking block, extract from after it
            if "</think>" in output:
                after_think = output.split("</think>")[-1].strip()
                if after_think:
                    # Look for code blocks after </think>
                    code_pattern = r"```(?:lean4?|lean|tactics)?\n?(.*?)```"
                    matches = re.findall(code_pattern, after_think, re.DOTALL)
                    if matches:
                        for match in matches:
                            extracted = self._extract_tactics_from_code_block(match)
                            if extracted and extracted.lower() not in ["sorry", "by"]:
                                tactics = extracted
                                break
                    # If no code block, try the raw text after </think>
                    if not tactics:
                        extracted = self._extract_tactics_from_code_block(after_think)
                        if extracted and extracted.lower() not in ["sorry", "by"]:
                            tactics = extracted
            
            # Strategy 2: Extract from code blocks inside <think> (model often puts partial proofs there)
            if not tactics and "<think>" in output:
                think_match = re.search(r"<think>(.*?)(?:</think>|$)", output, re.DOTALL)
                if think_match:
                    think_content = think_match.group(1)
                    code_pattern = r"```(?:lean4?|lean|tactics)?\n?(.*?)```"
                    matches = re.findall(code_pattern, think_content, re.DOTALL)
                    if matches:
                        # Collect all tactic-like content from code blocks
                        all_tactics = []
                        for match in matches:
                            extracted = self._extract_tactics_from_code_block(match)
                            if extracted and extracted.lower() not in ["sorry", "by"]:
                                all_tactics.append(extracted)
                        if all_tactics:
                            tactics = "\n".join(all_tactics)
            
            # Strategy 3: Look for code blocks anywhere in output
            if not tactics:
                code_pattern = r"```(?:lean4?|lean|tactics)?\n?(.*?)```"
                matches = re.findall(code_pattern, output, re.DOTALL)
                if matches:
                    for match in matches:
                        extracted = self._extract_tactics_from_code_block(match)
                        if extracted and extracted.lower() not in ["sorry", "by"]:
                            tactics = extracted
                            break
            
            # Strategy 4: Look for ":= by" pattern and extract what follows
            if not tactics and ":= by" in output:
                by_idx = output.rfind(":= by")
                after_by = output[by_idx + 5:].strip()
                # Take until end of line or code block
                if "```" in after_by:
                    after_by = after_by.split("```")[0]
                lines = after_by.split("\n")
                tactic_lines = []
                for line in lines[:10]:  # Take first 10 lines max
                    stripped = line.strip()
                    if stripped and stripped.lower() not in ["sorry", "by", ""]:
                        if not stripped.startswith("--"):  # Skip comments
                            tactic_lines.append(stripped)
                if tactic_lines:
                    tactics = "\n".join(tactic_lines)
            
            # Clean up the tactics
            if tactics:
                # Remove leading "by" if present
                if tactics.lower().startswith("by\n") or tactics.lower().startswith("by "):
                    tactics = tactics[2:].strip()
                elif tactics.lower() == "by":
                    tactics = None
                    
            # Validate tactics
            if tactics:
                # Check it's not just the theorem statement
                if ":= sorry" in tactics or tactics.strip() == "sorry":
                    tactics = None
                # Check it has some actual content
                elif len(tactics.strip()) < 3:
                    tactics = None
            
            # If no tactics extracted, return sorry - don't try to infer from reasoning
            # The previous heuristic of looking for keywords like "simp" in reasoning was
            # too aggressive and could match incidental word usage (e.g., "simply")
            # Better to return sorry and let the verification fail properly
            
            return tactics if tactics else "sorry"
        
        def _create_full_lean_code(self, config: SDPOConfig, theorem_code: str, proof_tactics: str, header: str = "") -> str:
            """Create full Lean 4 code by replacing sorry with proof tactics.
            
            If header is provided (from dataset), uses it and strips model-generated imports.
            If no header, extracts imports from model output and uses configurable default as fallback.
            """
            has_header = bool(header.strip())
            
            # Extract model-generated imports if present
            model_imports = []
            model_opens = []
            model_set_options = []
            tactics_clean_lines = []
            
            for line in proof_tactics.split("\n"):
                stripped = line.strip()
                if stripped.startswith("import "):
                    model_imports.append(stripped)
                elif stripped.startswith("open "):
                    model_opens.append(stripped)
                elif stripped.startswith("set_option "):
                    model_set_options.append(stripped)
                elif stripped.startswith("namespace "):
                    continue  # Always skip namespace
                elif stripped.startswith("section "):
                    continue  # Always skip section
                else:
                    tactics_clean_lines.append(line)
            
            tactics_clean = "\n".join(tactics_clean_lines).strip()
            if not tactics_clean:
                tactics_clean = "sorry"
            
            proof_lines = tactics_clean.split("\n")
            indented_proof = "\n  ".join(proof_lines)
            
            # Replace only the main theorem's proof placeholder, not earlier sorries (e.g. abbrev := sorry).
            # Try explicit patterns first; otherwise replace only the last "sorry" in the file.
            if ":= by sorry" in theorem_code:
                theorem_with_proof = theorem_code.replace(":= by sorry", f":= by\n  {indented_proof}")
            elif ":= by\n  sorry" in theorem_code:
                theorem_with_proof = theorem_code.replace(":= by\n  sorry", f":= by\n  {indented_proof}")
            else:
                last_sorry = theorem_code.rfind("sorry")
                if last_sorry != -1:
                    theorem_with_proof = (
                        theorem_code[:last_sorry] + indented_proof + theorem_code[last_sorry + 5:]
                    )
                else:
                    theorem_with_proof = theorem_code
            
            # Determine which header to use
            if has_header:
                # Dataset provides header, use it directly
                final_header = header
            elif model_imports:
                # No dataset header, but model generated imports - use them
                final_header_parts = []
                final_header_parts.extend(model_imports)
                if model_set_options:
                    final_header_parts.append("")
                    final_header_parts.extend(model_set_options)
                if model_opens:
                    final_header_parts.append("")
                    final_header_parts.extend(model_opens)
                final_header = "\n".join(final_header_parts)
            else:
                # No dataset header and no model imports - use configurable default
                final_header = config.default_header
            
            return f"{final_header}\n\n{theorem_with_proof}"
        
        @staticmethod
        def _add_tail(log_probs: "torch.Tensor") -> "torch.Tensor":
            """Append tail bucket for KL computation."""
            import torch
            log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
            log_s = torch.clamp(log_s, max=-1e-7)
            tail_log = torch.log(-torch.expm1(log_s))
            return torch.cat([log_probs, tail_log], dim=-1)
        
        def _compute_sdpo_loss(
            self,
            config: SDPOConfig,
            base_prompt: str,
            teacher_prompt: str,
            generated_ids: "torch.Tensor",
        ) -> tuple["torch.Tensor", float, float, float]:
            """Compute SDPO (Self-Distilled Policy Optimization) loss.
            
            SDPO Architecture:
            - STUDENT: Sees only the problem statement (base_prompt)
            - TEACHER: Sees problem + feedback from failed attempts (teacher_prompt)
            
            The loss distills knowledge from teacher to student by minimizing KL divergence
            between student's distribution and teacher's distribution on the top-K tokens.
            This allows the student to learn from the teacher's "hindsight" about what
            went wrong in previous attempts.
            
            Args:
                config: SDPO configuration
                base_prompt: Problem statement only (for student)
                teacher_prompt: Problem + feedback history (for teacher)
                generated_ids: Token IDs of the generated response
                
            Returns:
                per_token_kl: KL divergence per token
                reward: Reward signal
                avg_kl: Average KL divergence
                entropy: Policy entropy
            """
            import torch
            import torch.nn.functional as F
            
            K = config.distillation_topk
            model_device = next(self.model.parameters()).device
            
            # Tokenize student prompt (problem only)
            student_prompt_ids = self.tokenizer(
                base_prompt, return_tensors="pt", truncation=True, max_length=2048
            ).input_ids.to(model_device)
            
            # Tokenize teacher prompt (problem + feedback)
            teacher_prompt_ids = self.tokenizer(
                teacher_prompt, return_tensors="pt", truncation=True, max_length=2048
            ).input_ids.to(model_device)
            
            response_ids = generated_ids.to(model_device)
            if response_ids.dim() == 1:
                response_ids = response_ids.unsqueeze(0)
            
            # Concatenate prompts with response for forward pass
            student_input_ids = torch.cat([student_prompt_ids, response_ids], dim=1)
            teacher_input_ids = torch.cat([teacher_prompt_ids, response_ids], dim=1)
            
            student_prompt_len = student_prompt_ids.shape[1]
            teacher_prompt_len = teacher_prompt_ids.shape[1]
            seq_len = response_ids.shape[1]
            
            # Get student logits (conditioned on problem only)
            student_logits = self.model(
                input_ids=student_input_ids,
            ).logits[0, student_prompt_len - 1 : student_prompt_len - 1 + seq_len]
            
            # Compute entropy efficiently without materializing full softmax
            # H = log(sum(exp(x))) - sum(x * softmax(x))
            # = logsumexp(x) - sum(x * exp(x - logsumexp(x)))
            logsumexp_vals = torch.logsumexp(student_logits, dim=-1, keepdim=True)
            log_probs = student_logits - logsumexp_vals
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean().item()
            
            K_actual = min(K, student_logits.size(-1))
            student_topk_logits, topk_indices = torch.topk(student_logits, K_actual, dim=-1)
            student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
            student_topk_logps = student_topk_logits - student_logsumexp
            
            # Get teacher logits (conditioned on problem + feedback)
            # Teacher has "hindsight" about what went wrong
            with torch.no_grad():
                teacher_logits = self.model(
                    input_ids=teacher_input_ids,
                ).logits[0, teacher_prompt_len - 1 : teacher_prompt_len - 1 + seq_len]
                
                teacher_topk_logits = torch.gather(teacher_logits, dim=-1, index=topk_indices)
                teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
                teacher_topk_logps = teacher_topk_logits - teacher_logsumexp
            
            student_with_tail = self._add_tail(student_topk_logps)
            teacher_with_tail = self._add_tail(teacher_topk_logps)
            
            kl_per_bucket = F.kl_div(
                teacher_with_tail.detach(),
                student_with_tail,
                reduction="none",
                log_target=True,
            )
            per_token_kl = kl_per_bucket.sum(dim=-1)
            
            with torch.no_grad():
                target_ids = response_ids[0]
                student_lp = -F.cross_entropy(student_logits.detach(), target_ids, reduction="none")
                teacher_lp = -F.cross_entropy(teacher_logits, target_ids, reduction="none")
                total_reward = (student_lp - teacher_lp).sum().item()
                avg_kl = per_token_kl.mean().item()
            
            return per_token_kl, total_reward, avg_kl, entropy
        
        def _save_results(self, config: SDPOConfig, logs: dict, metrics: dict) -> str:
            """Save results, plots, and final model weights. Returns model save path."""
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import json
            from pathlib import Path
            
            output_dir = Path("/output") / config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_dir / f"run_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Save final model weights (before reloading to initial)
            model_save_dir = run_dir / "final_model"
            model_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving final model weights to {model_save_dir}...")
            self.model.save_pretrained(model_save_dir)
            self.tokenizer.save_pretrained(model_save_dir)
            print(f"Model saved to {model_save_dir}")
            
            logs_path = run_dir / "logs.json"
            with open(logs_path, "w") as f:
                json.dump(logs, f, indent=2, default=str)
            print(f"Logs saved to {logs_path}")
            
            if len(metrics["iterations"]) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"SDPO Test-Time RL - {config.model_name}", fontsize=12)
                
                ax1 = axes[0, 0]
                ax1.plot(metrics["iterations"], metrics["losses"], 'b-o', linewidth=2, markersize=8)
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Loss")
                ax1.set_title("Loss Curve")
                ax1.grid(True, alpha=0.3)
                
                ax2 = axes[0, 1]
                ax2.plot(metrics["iterations"], metrics["grad_norms"], 'r-o', linewidth=2, markersize=8)
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Gradient Norm")
                ax2.set_title("Gradient Update Steps")
                ax2.grid(True, alpha=0.3)
                
                ax3 = axes[1, 0]
                ax3.plot(metrics["iterations"], metrics["entropies"], 'g-o', linewidth=2, markersize=8)
                ax3.set_xlabel("Iteration")
                ax3.set_ylabel("Entropy")
                ax3.set_title("Policy Entropy")
                ax3.grid(True, alpha=0.3)
                
                ax4 = axes[1, 1]
                ax4.plot(metrics["iterations"], metrics["kl_divs"], 'm-o', linewidth=2, markersize=8)
                ax4.set_xlabel("Iteration")
                ax4.set_ylabel("KL Divergence")
                ax4.set_title("KL Divergence (Student vs Teacher)")
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                plot_path = run_dir / "training_curves.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Plots saved to {plot_path}")
            
            metrics_path = run_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {metrics_path}")
            
            return str(model_save_dir)

    # ========================================================================
    # NOTE: A100-80GB class removed to avoid unnecessary GPU allocation
    # ========================================================================
    # 
    # Running 7-8B models (like Goedel-Prover-V2-8B) on A100-80GB currently has
    # memory issues due to loading both vLLM and HuggingFace models simultaneously.
    # 
    # Known issues with 8B models:
    # 1. OOM during optimizer.step() - vLLM + HF model + optimizer states > 80GB
    # 2. 8-bit quantization causes NaN loss (no gradient support)
    # 3. LoRA + 4-bit quantization is experimental
    #
    # For now, use the default A100-40GB configuration with 1-2B models.
    # See docs/GPU_CONFIG_NOTES.md for details.
    # ========================================================================

    # ========================================================================
    # Local Entrypoint
    # ========================================================================
    
    # Default field mappings for dataset-agnostic loading
    DEFAULT_THEOREM_FIELDS = [
        "lean4_code", "formal_statement", "lean4_statement", 
        "statement", "code", "theorem", "problem_statement"
    ]
    DEFAULT_INFORMAL_FIELDS = [
        "informal_prefix", "problem", "informal_statement",
        "natural_language", "description", "question", "informal"
    ]
    DEFAULT_HEADER_FIELDS = ["header", "imports", "preamble", "prefix"]
    DEFAULT_ID_FIELDS = ["problem_id", "name", "id", "idx", "index"]
    
    def get_field(data: dict, field_names: list, default: str = "") -> str:
        """Get a field from data dict, trying multiple possible field names."""
        for field_name in field_names:
            if field_name in data and data[field_name]:
                value = data[field_name]
                if isinstance(value, str):
                    return value
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    return str(value[0])
        return default

    @app.local_entrypoint()
    def main(
        model: str = "AI-MO/Kimina-Prover-RL-1.7B",
        dataset: str = "cat-searcher/minif2f-lean4",
        dataset_subset: str = "",
        dataset_split: str = "test",
        problem_idx: int = 0,
        max_iterations: int = 5,
        learning_rate: float = 1e-5,
        temperature: float = 0.6,
        feedback_errors_only: bool = False,
        system_prompt: str = "",
        default_header: str = "",
        theorem_field: str = "",
        informal_field: str = "",
        header_field: str = "",
        gpu: str = "A100-40GB",
    ):
        """
        Run SDPO test-time RL on Modal.
        
        This pipeline is model and dataset agnostic. It automatically detects
        common field names used by various Lean theorem proving datasets.
        
        Args:
            model: HuggingFace model ID (any causal LM that can generate Lean proofs)
            dataset: HuggingFace dataset ID (any dataset with theorem statements)
            dataset_subset: Dataset subset/config name (if any)
            dataset_split: Dataset split (train/test/validation)
            problem_idx: Index of problem to solve (0-indexed)
            max_iterations: Max RL iterations for test-time training
            learning_rate: Learning rate for gradient updates
            temperature: Sampling temperature for generation
            feedback_errors_only: Only include error messages in feedback (not failed proofs)
            system_prompt: Custom system prompt (optional, uses default if empty)
            default_header: Custom default Lean header for imports (optional)
            theorem_field: Override field name for theorem code (optional)
            informal_field: Override field name for informal statement (optional)
            header_field: Override field name for Lean header/imports (optional)
            gpu: GPU configuration (currently only A100-40GB is supported).
        
        GPU configurations:
            - A100-40GB (default): For 1-2B models (e.g., Kimina-Prover-RL-1.7B)
            - A100-80GB/H100: NOT SUPPORTED - 8B models have memory issues with SDPO training
              (see docs/GPU_CONFIG_NOTES.md for details)
        
        Supported datasets (auto-detected field names):
            - cat-searcher/minif2f-lean4 (lean4_code, header)
            - amitayusht/PutnamBench (lean4_statement, informal_statement)
            - deepmind/math (problem, solution)
            - Custom datasets with: statement, code, theorem, problem_statement, etc.
        
        Supported models:
            - AI-MO/Kimina-Prover-RL-1.7B (default, thinking model)
            - Other 1-2B models that fit in A100-40GB memory
        """
        from datasets import load_dataset
        
        print("="*60)
        print("SDPO Test-Time RL on Modal")
        print("="*60)
        print(f"Model: {model}")
        print(f"GPU: {gpu}")
        print(f"Dataset: {dataset}")
        if dataset_subset:
            print(f"Dataset subset: {dataset_subset}")
        print(f"Dataset split: {dataset_split}")
        print(f"Problem index: {problem_idx}")
        print(f"Max iterations: {max_iterations}")
        print(f"Feedback mode: {'errors only' if feedback_errors_only else 'errors + failed proofs'}")
        print("="*60)
        
        print(f"\nLoading dataset {dataset}...")
        try:
            if dataset_subset:
                ds = load_dataset(dataset, dataset_subset, split=dataset_split)
            else:
                ds = load_dataset(dataset, split=dataset_split)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            print("Trying alternative loading...")
            try:
                ds = load_dataset(dataset, split=dataset_split, trust_remote_code=True)
            except Exception as e2:
                print(f"Also failed with trust_remote_code: {e2}")
                # Try without split specification
                try:
                    full_ds = load_dataset(dataset, trust_remote_code=True)
                    if dataset_split in full_ds:
                        ds = full_ds[dataset_split]
                    else:
                        # Use first available split
                        available_splits = list(full_ds.keys())
                        print(f"Split '{dataset_split}' not found. Available: {available_splits}")
                        ds = full_ds[available_splits[0]]
                        print(f"Using split: {available_splits[0]}")
                except Exception as e3:
                    raise RuntimeError(f"Could not load dataset {dataset}: {e3}")
        
        print(f"Dataset loaded with {len(ds)} examples")
        print(f"Available columns: {ds.column_names}")
        
        if problem_idx >= len(ds):
            print(f"Problem index {problem_idx} out of range (dataset has {len(ds)} examples)")
            problem_idx = 0
        
        problem = dict(ds[problem_idx])
        
        # Build field lists with user overrides at the front
        theorem_fields = ([theorem_field] if theorem_field else []) + DEFAULT_THEOREM_FIELDS
        informal_fields = ([informal_field] if informal_field else []) + DEFAULT_INFORMAL_FIELDS
        header_fields = ([header_field] if header_field else []) + DEFAULT_HEADER_FIELDS
        id_fields = DEFAULT_ID_FIELDS
        
        print(f"\nLoaded problem {problem_idx}:")
        problem_id = get_field(problem, id_fields, f"problem_{problem_idx}")
        print(f"  ID: {problem_id}")
        
        lean4_code = get_field(problem, theorem_fields)
        if lean4_code:
            print(f"  Lean4 code: {lean4_code[:200]}...")
        else:
            print("  WARNING: No theorem code found! Check dataset field names.")
            print(f"  Available fields: {list(problem.keys())}")
        
        informal = get_field(problem, informal_fields)
        if informal:
            print(f"  Informal: {informal[:100]}...")
        
        header = get_field(problem, header_fields)
        if header:
            print(f"  Header: {header[:100]}...")
        
        # Build config dict with all settings
        config_dict = {
            "model_name": model,
            "dataset_name": dataset,
            "dataset_subset": dataset_subset or None,
            "dataset_split": dataset_split,
            "problem_idx": problem_idx,
            "max_iterations": max_iterations,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "feedback_include_failed_proof": not feedback_errors_only,
            # Custom field mappings
            "theorem_fields": theorem_fields,
            "informal_fields": informal_fields,
            "header_fields": header_fields,
            "id_fields": id_fields,
        }
        
        # Add optional overrides
        if system_prompt:
            config_dict["system_prompt"] = system_prompt
        if default_header:
            config_dict["default_header"] = default_header
        
        # Select trainer class based on GPU configuration
        # NOTE: A100-80GB and H100 classes have been removed due to memory issues with 8B models
        # See docs/GPU_CONFIG_NOTES.md for details
        gpu_upper = gpu.upper()
        if gpu_upper in ["A100-80GB", "A100_80GB", "H100"]:
            print(f"\n⚠️  WARNING: {gpu} GPU requested but not currently supported for SDPO training.")
            print(f"    8B models have memory issues (OOM during optimizer step).")
            print(f"    Falling back to A100-40GB with 1-2B models.")
            print(f"    See docs/GPU_CONFIG_NOTES.md for details.\n")
        
        print(f"Using A100-40GB GPU (for 1-2B models)")
        trainer = SDPOTrainer(model_name=model)
        
        print("Starting SDPO training on Modal...")
        results = trainer.run_sdpo.remote(config_dict, problem)
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Success: {results['success']}")
        print(f"Iterations used: {len(results['iteration_logs'])}")
        
        if results['success']:
            print(f"Best proof: {results['best_proof'][:200]}...")
        
        if results['metrics']['losses']:
            print(f"\nFinal metrics:")
            print(f"  Final loss: {results['metrics']['losses'][-1]:.4f}")
            print(f"  Final entropy: {results['metrics']['entropies'][-1]:.4f}")
            print(f"  Final grad norm: {results['metrics']['grad_norms'][-1]:.4f}")
        
        print("\nResults saved to Modal volume 'sdpo-output' under 'test-time-SDPO/'")
        
        # Save a local copy of the results, metrics, and plots
        import json
        from pathlib import Path
        from datetime import datetime
        
        local_output_dir = Path("sdpo_results")
        local_output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = local_output_dir / f"run_{problem_idx}_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        
        # Save full logs
        local_log_path = run_dir / "logs.json"
        with open(local_log_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Local log saved to: {local_log_path}")
        
        # Save metrics separately
        metrics = results.get("metrics", {})
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {metrics_path}")
        
        # Generate and save plots locally
        if metrics.get("iterations") and len(metrics["iterations"]) > 0:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"SDPO Test-Time RL - Problem {problem_idx}", fontsize=12)
            
            ax1 = axes[0, 0]
            ax1.plot(metrics["iterations"], metrics["losses"], 'b-o', linewidth=2, markersize=8)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Loss")
            ax1.set_title("Loss Curve")
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[0, 1]
            ax2.plot(metrics["iterations"], metrics["grad_norms"], 'r-o', linewidth=2, markersize=8)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Gradient Norm")
            ax2.set_title("Gradient Update Steps")
            ax2.grid(True, alpha=0.3)
            
            ax3 = axes[1, 0]
            ax3.plot(metrics["iterations"], metrics["entropies"], 'g-o', linewidth=2, markersize=8)
            ax3.set_xlabel("Iteration")
            ax3.set_ylabel("Entropy")
            ax3.set_title("Policy Entropy")
            ax3.grid(True, alpha=0.3)
            
            ax4 = axes[1, 1]
            ax4.plot(metrics["iterations"], metrics["kl_divs"], 'm-o', linewidth=2, markersize=8)
            ax4.set_xlabel("Iteration")
            ax4.set_ylabel("KL Divergence")
            ax4.set_title("KL Divergence (Student vs Teacher)")
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = run_dir / "training_curves.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Training curves saved to: {plot_path}")
        
        print("="*60)
        
        return results

except ImportError:
    print("Modal not installed. Install with: pip install modal")
    
    def main():
        print("This script requires Modal. Install with: pip install modal")
        print("Then run: modal run lean_sdpo_modal.py --model <model> --problem-idx <idx>")


if __name__ == "__main__":
    main()
