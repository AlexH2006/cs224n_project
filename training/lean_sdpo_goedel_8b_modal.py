"""
SDPO Test-Time RL for Goedel-Prover-V2-8B on Modal

Runs Self-Distilled Policy Optimization (SDPO) on Modal for Lean 4 theorem
proving using the Goedel-Prover-V2-8B model with LoRA fine-tuning via Unsloth.

Key design:
    - Two model instances on one GPU:
      * vLLM (bf16 base model, ~16GB): fast inference with LoRA overlay
      * Unsloth (4-bit quantized + LoRA adapters): gradient computation
    - LoRA weights bridge the two: Unsloth saves -> vLLM loads via LoRARequest
    - Base model is frozen; only LoRA adapter weights are trained
    - Gradient accumulation over configurable minibatch size (default 4)

SDPO Architecture:
    - STUDENT: Generates proofs given ONLY the problem statement.
    - TEACHER: Provides distillation target given problem + error feedback.
    - Student learns by minimizing KL divergence to teacher's distribution.

Usage:
    modal run lean_sdpo_goedel_8b_modal.py --problem-idx 0
    modal run lean_sdpo_goedel_8b_modal.py --problem-idx 0 --lora-rank 32
    modal run lean_sdpo_goedel_8b_modal.py --problem-idx 0 --gradient-accumulation-steps 8
    python lean_sdpo_goedel_8b_modal.py --test-extraction   # run tactic extraction tests (no Modal)
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
    """Configuration for Goedel-8B SDPO with LoRA on Modal."""

    # Model
    model_name: str = "Goedel-LM/Goedel-Prover-V2-8B"

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"

    # Training
    max_iterations: int = 5
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 4
    distillation_topk: int = 20

    # Generation
    # Goedel-Prover-V2-8B (Qwen2Tokenizer): eos_token="<|im_end|>", pad_token="<|endoftext|>",
    # bos_token=null. Chat format: <|im_start|>role\ncontent<|im_end|>\n; generation prompt ends with <|im_start|>assistant\n
    # 8192 is sufficient for Lean proofs; 32K wastes time when the model rambles.
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    stop_tokens: list = field(default_factory=lambda: [
        "<|im_end|>",   # Goedel/Qwen eos
        "<|endoftext|>",  # Goedel/Qwen pad
        "<|im_start|>",   # Prevent starting a new turn
        "</s>", "<|end|>", "[/INST]", "<|eot_id|>",  # Other common eos tokens
    ])

    # Dataset
    dataset_name: str = "cat-searcher/minif2f-lean4"
    dataset_subset: Optional[str] = None
    dataset_split: str = "test"
    problem_idx: int = 0

    # Dataset field mapping -- lists of candidate field names, tried in order
    theorem_fields: list = field(default_factory=lambda: [
        "lean4_code", "formal_statement", "lean4_statement",
        "statement", "code", "theorem", "problem_statement"
    ])
    informal_fields: list = field(default_factory=lambda: [
        "informal_prefix", "informal_stmt", "problem", "informal_statement",
        "natural_language", "description", "question", "informal"
    ])
    header_fields: list = field(default_factory=lambda: [
        "header", "imports", "preamble", "prefix"
    ])
    id_fields: list = field(default_factory=lambda: [
        "problem_id", "name", "id", "idx", "index"
    ])

    # Default Lean header (fallback when dataset doesn't provide one)
    default_header: str = """import Mathlib
import Aesop

set_option maxHeartbeats 400000

open BigOperators Real Nat Topology Rat"""

    # Feedback
    feedback_include_failed_proof: bool = False
    feedback_attempt_template: str = """- Error: {feedback}
  Failed proof: {failed_proof}"""
    feedback_attempt_template_errors_only: str = """- {feedback}"""
    feedback_separator: str = "\n"

    # Output
    output_dir: str = "goedel-sdpo-results"


# ============================================================================
# Tactic extraction (module-level for testing without Modal)
# ============================================================================

def _extract_tactics_from_code_block(block: str) -> str:
    """Extract tactic body from raw contents of a ```lean / ```lean4 / ``` code block (no fences).
    
    Split into lines; remove theorem/lemma/example header through `:= by`.
    Preserve module directives: import , open , set_option .
    If block has no theorem header (raw tactics), treat entire block as tactic body.
    Return tactic body as newline-joined string with trailing blank lines removed.
    """
    lines = block.split("\n")
    result_lines: list[str] = []
    in_tactic_body = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_tactic_body:
                result_lines.append("")
            continue

        if stripped.startswith(("import ", "open ", "set_option ")):
            result_lines.append(stripped)
            continue

        if not in_tactic_body:
            if stripped.startswith(("theorem ", "lemma ", "example ")) and ":= by" in stripped:
                after_by = stripped.split(":= by", 1)[1].strip()
                if after_by:
                    result_lines.append(after_by)
                in_tactic_body = True
                continue
            if stripped.endswith(":= by") or stripped in (":= by", "by"):
                in_tactic_body = True
                continue
            if re.match(r"^[\(\[]", stripped) or stripped.endswith(":="):
                continue
            in_tactic_body = True

        result_lines.append(stripped)

    while result_lines and not result_lines[-1].strip():
        result_lines.pop()

    return "\n".join(result_lines)


def _extract_proof_tactics(output: str) -> str:
    """Extract proof tactics using priority order and best-block scoring.
    
    A) After </think>: score all code blocks, take best; else raw text.
    B) Inside <think>...</think>|$: extract all code blocks, concatenate.
    C) Any code block in output: select best.
    D) After := by: up to 10 non-empty non-comment lines.
    
    Score: fewer sorry > longer length. Reject empty/sorry/by. Return sorry on failure.
    """
    code_pattern = r"```(?:lean4?|lean|tactics)?\n?(.*?)```"

    def score_block(t: str) -> tuple[int, int]:
        if not t or t.lower() in ("sorry", "by"):
            return (-999, 0)
        sc = t.lower().count("sorry")
        return (-sc, len(t))

    def pick_best(matches: list[str]) -> Optional[str]:
        best, best_score = None, None
        for m in matches:
            extracted = _extract_tactics_from_code_block(m)
            sc = score_block(extracted)
            if sc[0] == -999:
                continue
            if best_score is None or sc > best_score:
                best, best_score = extracted, sc
        return best

    tactics: Optional[str] = None

    # A) </think> present: only text after last </think>
    if "</think>" in output:
        after = output.split("</think>")[-1].strip()
        matches = re.findall(code_pattern, after, re.DOTALL)
        if matches:
            tactics = pick_best(matches)
        if not tactics:
            tactics = _extract_tactics_from_code_block(after)
            if not tactics or tactics.lower() in ("sorry", "by"):
                tactics = None

    # B) <think> present: inside think region
    if not tactics and "<think>" in output:
        m = re.search(r"<think>(.*?)(?:</think>|$)", output, re.DOTALL)
        if m:
            matches = re.findall(code_pattern, m.group(1), re.DOTALL)
            if matches:
                parts = []
                for blk in matches:
                    ex = _extract_tactics_from_code_block(blk)
                    if ex and ex.lower() not in ("sorry", "by"):
                        parts.append(ex)
                if parts:
                    tactics = "\n".join(parts)

    # C) Any code block in output
    if not tactics:
        matches = re.findall(code_pattern, output, re.DOTALL)
        if matches:
            tactics = pick_best(matches)

    # D) := by fallback
    if not tactics and ":= by" in output:
        idx = output.rfind(":= by")
        after = output[idx + 5:].strip()
        if "```" in after:
            after = after.split("```")[0]
        lines = []
        for line in after.split("\n")[:10]:
            s = line.strip()
            if s and not s.startswith("--") and s.lower() not in ("sorry", "by"):
                lines.append(s)
        if lines:
            tactics = "\n".join(lines)

    # Final cleanup
    if tactics:
        if tactics.lower().startswith("by\n") or tactics.lower().startswith("by "):
            tactics = tactics[2:].strip()
        elif tactics.lower().strip() == "by":
            tactics = None
    if tactics:
        st = tactics.strip()
        if st in ("sorry", "by", "by sorry") or len(st) < 3:
            tactics = None

    return tactics if tactics else "sorry"


def _run_extraction_tests() -> None:
    """Run sanity checks on tactic extraction. No Modal required."""
    cases = [
        # (i) clean ```lean4 block with theorem header
        (
            "i",
            "Some text\n```lean4\ntheorem foo (n : Nat) : n + 0 = n := by\n  simp\n```",
        ),
        # (ii) ```lean block with raw tactics only
        (
            "ii",
            "```lean\n  rw [add_comm]\n  simp\n```",
        ),
        # (iii) output with <think>...</think> and code after </think>
        (
            "iii",
            "<think>Let me think...</think>\n```lean4\ntheorem bar : 1 + 1 = 2 := by\n  norm_num\n```",
        ),
        # (iv) truncated <think> with code inside
        (
            "iv",
            "<think>The proof is:\n```lean\n  simp\n  rfl\n```",
        ),
        # (v) multiple code blocks, first contains sorry, second is longer/cleaner
        (
            "v",
            "```lean4\ntheorem x : 1 = 1 := by sorry\n```\nActually:\n```lean4\ntheorem x : 1 = 1 := by\n  rfl\n```",
        ),
        # (vi) no code blocks but has := by and a few tactic lines
        (
            "vi",
            "theorem baz : True := by\n  trivial\n  done",
        ),
    ]
    for label, inp in cases:
        out = _extract_proof_tactics(inp)
        print(f"[{label}] extracted ({len(out)} chars):")
        print(out[:200] + ("..." if len(out) > 200 else ""))
        print()


# ============================================================================
# Modal App Definition
# ============================================================================

try:
    import modal
    
    app = modal.App("lean-sdpo-goedel-8b")
    
    # Volume for HF cache (persistent across runs)
    hf_cache_volume = modal.Volume.from_name("sdpo-hf-cache", create_if_missing=True)
    
    # Volume for output logs (persistent)
    output_volume = modal.Volume.from_name("sdpo-output", create_if_missing=True)
    
    # Image for inference + training (GPU)
    # Unsloth must be imported before transformers to apply its patches.
    # Uses uv for fast, conflict-tolerant dependency resolution.
    # Versions pinned from Modal's official Unsloth example.
    inference_image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "unsloth[cu128-torch270]==2025.7.8",
            "unsloth_zoo==2025.7.10",
            "trl==0.19.1",
            "vllm>=0.6.0",
            "transformers==4.53.2",
            "accelerate==1.9.0",
            "peft==0.16.0",
            "sentencepiece",
            "protobuf",
            "datasets",
            "matplotlib",
            "httpx",
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
    )
    @modal.concurrent(max_inputs=100)
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
    # SDPO Training Service - Setup Logic
    #
    # Two model instances coexist on one GPU:
    #   1. vLLM  (bf16 base + LoRA overlay) — fast inference via LoRARequest
    #   2. Unsloth (4-bit base + trainable LoRA) — gradient computation
    # LoRA weights are the bridge: Unsloth saves → vLLM loads from disk.
    # ========================================================================

    # LoRA on all linear layers: embed + lm_head + attn + MLP (~48.6M params @ r=16).
    LORA_TARGET_MODULES = [
        "embed_tokens", "lm_head",
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    LORA_WEIGHT_DIR = "/tmp/sdpo_lora_weights"

    def _setup_trainer(trainer_self, gpu_memory_utilization: float, max_model_len: int, gpu_name: str):
        """Load Unsloth training model and vLLM inference engine with LoRA support."""
        import os
        os.environ["HF_HOME"] = "/cache"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        # Force legacy (V0) engine so enforce_eager is respected; V1 always runs torch.compile (~3+ min).
        os.environ["VLLM_USE_V1"] = "0"
        if os.environ.get("HF_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

        import torch

        # Unsloth must be imported before transformers to apply its patches
        from unsloth import FastLanguageModel
        from vllm import LLM

        print(f"Loading model: {trainer_self.model_name}...")
        trainer_self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Unsloth training model (4-bit quantized + LoRA) ---
        print("Loading Unsloth 4-bit model with LoRA adapters...")
        trainer_self.model, trainer_self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=trainer_self.model_name,
            max_seq_length=max_model_len,
            load_in_4bit=True,
            dtype=None,
        )
        if trainer_self.tokenizer.pad_token is None:
            trainer_self.tokenizer.pad_token = trainer_self.tokenizer.eos_token

        lora_rank = trainer_self.lora_rank
        lora_alpha = trainer_self.lora_alpha

        trainer_self.model = FastLanguageModel.get_peft_model(
            trainer_self.model,
            r=lora_rank,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True,
        )
        trainer_self.model.print_trainable_parameters()

        # Save initial LoRA weights so vLLM can reference them from iteration 0
        initial_lora_path = f"{LORA_WEIGHT_DIR}/v0"
        os.makedirs(initial_lora_path, exist_ok=True)
        trainer_self.model.save_pretrained(initial_lora_path)
        trainer_self.tokenizer.save_pretrained(initial_lora_path)
        trainer_self.lora_version = 0
        print(f"Initial LoRA weights saved to {initial_lora_path}")

        # --- vLLM inference engine (bf16 base + LoRA overlay) ---
        # VLLM_USE_V1=0 above forces legacy engine; V1 ignores enforce_eager and always compiles (~3+ min).
        # enforce_eager=True: no torch.compile, no CUDA graph capture — fast startup for few-iteration runs.
        # gpu_memory_utilization: vLLM sees full GPU; we set 0.35 so vLLM + Unsloth fit.
        print(f"Initializing vLLM engine ({gpu_name}, V0/eager, enable_lora=True)...")
        trainer_self.vllm_engine = LLM(
            model=trainer_self.model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            download_dir="/cache",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=1,
            enable_lora=True,
            max_lora_rank=lora_rank,
            enforce_eager=True,
        )
        print("vLLM engine initialized!")

        print(f"Setup complete on {trainer_self.device}!")

    # ========================================================================
    # SDPO Training Service (GPU) - A100-80GB
    # Memory budget (~80GB) with max_new_tokens=8192, max_model_len=10240:
    #   vLLM bf16 model + KV cache + CUDA graphs:
    #     - Model weights (bf16):         ~15.4 GB
    #     - KV cache (10240 tokens):      ~1.3 GB
    #     - CUDA graphs:                  ~1 GB
    #     - Total vLLM (gpu_memory_utilization=0.35): ~28 GB budget
    #   Unsloth 4-bit + LoRA:             ~6 GB
    #   Optimizer (LoRA) + activations:   ~35 GB
    #   Headroom:                          ~11 GB
    # Inference is always batch_size=1 (one prompt per _generate_proof call).
    # ========================================================================
    
    # Startup can take 5–10+ min on cold cache: Unsloth downloads/loads 8B in 4-bit,
    # then vLLM downloads/loads same model in bf16 in a separate process. A heartbeat
    # warning during setup is normal; the worker continues. startup_timeout allows
    # this phase to finish; timeout applies to the run after setup.
    @app.cls(
        image=inference_image,
        gpu="A100-80GB",
        startup_timeout=900,  # 15 min for model download + Unsloth + vLLM init
        timeout=3600,
        scaledown_window=600,
        volumes={"/cache": hf_cache_volume, "/output": output_volume},
        secrets=[modal.Secret.from_name("huggingface")],
    )
    class SDPOTrainer:
        """SDPO trainer for Goedel-Prover-V2-8B with Unsloth LoRA on A100-80GB."""
        
        model_name: str = modal.parameter(default="Goedel-LM/Goedel-Prover-V2-8B")
        lora_rank: int = modal.parameter(default=16)
        lora_alpha: int = modal.parameter(default=32)

        @modal.enter()
        def setup(self):
            # max_model_len must fit prompt + max_new_tokens (8192). 10240 leaves room for prompts.
            _setup_trainer(self, gpu_memory_utilization=0.35, max_model_len=10240, gpu_name="A100-80GB")
        
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
            """Run SDPO test-time RL on a single problem.

            Gradient accumulation: losses are accumulated over
            `gradient_accumulation_steps` iterations before calling
            optimizer.step(). LoRA weights are saved and reloaded
            into vLLM at each optimizer step boundary.
            """
            import torch
            import torch.nn.functional as F
            from datetime import datetime

            config = SDPOConfig(**config_dict)
            accum_steps = config.gradient_accumulation_steps

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

            # Student prompt (problem only) -- constant across all iterations
            base_prompt = self._create_base_prompt(config, problem)

            # Track how many valid losses have been accumulated in this window
            accum_count = 0
            optimizer.zero_grad()

            for iteration in range(config.max_iterations):
                iter_start = time.time()
                minibatch_pos = (accum_count % accum_steps) + 1
                print(f"\n--- Iteration {iteration + 1}/{config.max_iterations} "
                      f"[accum {minibatch_pos}/{accum_steps}, LoRA v{self.lora_version}] ---")

                # ---- Generate (student: problem only, via vLLM + LoRA) ----
                raw_output, generated_ids = self._generate_proof(config, base_prompt)
                print(f"  Generated {len(raw_output)} chars")

                # ---- Extract tactics and build full Lean code ----
                is_truncated = "<think>" in raw_output and "</think>" not in raw_output
                tactics = self._extract_proof_tactics(raw_output)
                lean4_code = self._get_field(problem, config.theorem_fields)
                header = self._get_field(problem, config.header_fields)
                full_code = self._create_full_lean_code(config, lean4_code, tactics, header)

                # ---- Verify ----
                verification = self._verify_with_retries(full_code)
                is_success = verification["success"] and verification["complete"]
                is_server_error = verification.get("is_server_error", False)

                # Treat bare "sorry" as failure
                is_sorry_tactic = tactics.strip().lower() == "sorry"
                if is_sorry_tactic:
                    is_success = False
                    verification["complete"] = False
                    verification["has_sorry"] = True
                    if is_truncated:
                        verification["feedback"] = (
                            "Output was truncated before completing. "
                            "Be more concise and output proof tactics sooner."
                        )
                        verification["is_truncated"] = True
                    else:
                        verification["feedback"] = (
                            "No valid proof tactics found. "
                            "Output actual Lean 4 tactics in a ```lean4 code block."
                        )

                self._print_verification_status(
                    is_success, is_server_error, is_truncated, is_sorry_tactic
                )

                # ---- Build iteration log ----
                current_teacher_prompt = (
                    self._create_feedback_prompt(config, problem, feedback_history)
                    if feedback_history else None
                )
                iter_log = {
                    "iteration": iteration + 1,
                    "lora_version": self.lora_version,
                    "student_prompt": base_prompt,
                    "teacher_prompt": current_teacher_prompt,
                    "raw_output": raw_output,
                    "extracted_tactics": tactics,
                    "full_code": full_code,
                    "verification": verification,
                    "success": is_success,
                }

                # ---- Success: stop early ----
                if is_success:
                    best_proof = tactics
                    self._append_null_metrics(iter_log, logs, metrics, iteration, iter_start)
                    metrics["rewards"][-1] = 1.0
                    print(f"  Proof found!")
                    break

                # ---- Server error: skip training, don't accumulate ----
                if is_server_error:
                    iter_log["server_error"] = True
                    self._append_null_metrics(iter_log, logs, metrics, iteration, iter_start)
                    print(f"  Skipping SDPO update due to server error")
                    continue

                # ---- Failed: compute SDPO loss and accumulate gradient ----
                feedback = self._extract_feedback(verification)
                feedback_history.append((feedback, tactics))

                teacher_prompt = self._create_feedback_prompt(
                    config, problem, feedback_history
                )
                print(f"  SDPO: Teacher sees {len(feedback_history)} feedback(s)")

                per_token_kl, reward, avg_kl, entropy = self._compute_sdpo_loss(
                    config, base_prompt, teacher_prompt, generated_ids
                )

                # Scale loss for gradient accumulation averaging
                loss = per_token_kl.mean() / accum_steps
                loss.backward()
                accum_count += 1

                loss_val = loss.item() * accum_steps  # Log the unscaled loss

                # ---- Optimizer step at accumulation boundary ----
                grad_norm = None
                is_last_iter = (iteration == config.max_iterations - 1)
                if accum_count % accum_steps == 0 or is_last_iter:
                    grad_norm = self._compute_grad_norm()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    self._save_lora_and_bump_version()
                    print(f"  Optimizer step (accumulated {min(accum_count, accum_steps)} gradients)")

                print(f"  Loss: {loss_val:.4f}, Reward: {reward:.4f}, KL: {avg_kl:.4f}")
                print(f"  Entropy: {entropy:.4f}" +
                      (f", Grad norm: {grad_norm:.4f}" if grad_norm is not None else ""))

                iter_log["teacher_prompt"] = teacher_prompt
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
                metrics["grad_norms"].append(grad_norm or 0.0)
                metrics["timestamps"].append(time.time() - iter_start)

            logs["end_time"] = datetime.now().isoformat()
            logs["success"] = best_proof is not None
            logs["best_proof"] = best_proof
            logs["metrics"] = metrics

            model_save_path = self._save_results(config, logs, metrics)
            logs["model_save_path"] = model_save_path

            return logs

        # --- Helpers for run_sdpo ---

        def _verify_with_retries(self, full_code: str, max_retries: int = 3) -> dict:
            """Verify Lean code with retry logic for transient server errors."""
            for attempt in range(max_retries):
                verification = LeanVerifier().verify.remote(full_code)
                if not verification.get("is_server_error", False):
                    return verification
                if attempt < max_retries - 1:
                    print(f"  Server error on verification, retrying ({attempt + 1}/{max_retries})...")
                    time.sleep(5)
            return verification

        @staticmethod
        def _print_verification_status(
            is_success: bool, is_server_error: bool,
            is_truncated: bool, is_sorry_tactic: bool,
        ):
            if is_server_error:
                print(f"  Verification: SERVER ERROR (will not count as failed proof)")
            elif is_truncated:
                print(f"  Verification: FAILED (output truncated)")
            elif is_sorry_tactic:
                print(f"  Verification: FAILED (no valid tactics extracted)")
            else:
                print(f"  Verification: {'SUCCESS' if is_success else 'FAILED'}")

        @staticmethod
        def _extract_feedback(verification: dict) -> str:
            if verification.get("has_sorry"):
                return (verification["feedback"] or
                        "Forbidden to output `sorry`. Provide actual proof tactics.")
            return verification["feedback"] or "Proof verification failed"

        @staticmethod
        def _append_null_metrics(iter_log, logs, metrics, iteration, iter_start):
            """Append placeholder metrics for iterations that skip training."""
            for key in ("loss", "reward", "kl_div", "entropy", "grad_norm"):
                iter_log[key] = None
            logs["iteration_logs"].append(iter_log)
            metrics["iterations"].append(iteration + 1)
            metrics["losses"].append(0.0)
            metrics["rewards"].append(0.0)
            metrics["kl_divs"].append(0.0)
            metrics["entropies"].append(0.0)
            metrics["grad_norms"].append(0.0)
            metrics["timestamps"].append(time.time() - iter_start)

        def _compute_grad_norm(self) -> float:
            """Compute L2 norm of gradients across all trainable parameters."""
            total = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total += p.grad.data.norm(2).item() ** 2
            return total ** 0.5
        
        def _create_base_prompt(self, config: SDPOConfig, theorem: dict) -> str:
            """Create STUDENT prompt (problem only, no feedback).
            
            Follows the Goedel-Prover-V2 prompt format:
            - User-only message (no system prompt)
            - Full formal statement (imports + theorem) in a lean4 code block
            - Instruction to provide proof plan before code
            """
            lean4_code = self._get_field(theorem, config.theorem_fields)
            header = self._get_field(theorem, config.header_fields)

            full_statement = f"{header}\n\n{lean4_code}" if header.strip() else lean4_code

            user_content = (
                f"Complete the following Lean 4 code:\n\n"
                f"```lean4\n{full_statement}\n```\n\n"
                f"Before producing the Lean 4 code to formally prove the given theorem, "
                f"provide a detailed proof plan outlining the main proof steps and strategies.\n"
                f"The plan should highlight key ideas, intermediate lemmas, and proof "
                f"structures that will guide the construction of the final formal proof."
            )

            messages = [{"role": "user", "content": user_content}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        def _create_feedback_prompt(
            self,
            config: SDPOConfig,
            theorem: dict,
            feedback_history: list[tuple[str, str]],
        ) -> str:
            """Create TEACHER prompt (problem + accumulated error feedback).
            
            Same Goedel format as the student prompt, but with an additional
            section listing previous errors so the teacher has "hindsight".
            """
            lean4_code = self._get_field(theorem, config.theorem_fields)
            header = self._get_field(theorem, config.header_fields)

            full_statement = f"{header}\n\n{lean4_code}" if header.strip() else lean4_code

            user_content = (
                f"Complete the following Lean 4 code:\n\n"
                f"```lean4\n{full_statement}\n```\n\n"
            )

            # Append error feedback from previous attempts
            if feedback_history:
                user_content += "Previous proof attempts produced these errors:\n"
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
                user_content += "\n\nAvoid these errors.\n\n"

            user_content += (
                "Before producing the Lean 4 code to formally prove the given theorem, "
                "provide a detailed proof plan outlining the main proof steps and strategies.\n"
                "The plan should highlight key ideas, intermediate lemmas, and proof "
                "structures that will guide the construction of the final formal proof."
            )

            messages = [{"role": "user", "content": user_content}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        def _generate_proof(self, config: SDPOConfig, prompt: str) -> tuple[str, "torch.Tensor"]:
            """Generate a proof using vLLM with current LoRA weights."""
            import torch
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest

            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_new_tokens,
                stop=config.stop_tokens,
            )

            # Apply current LoRA weights (None for base model at version 0)
            lora_request = None
            if self.lora_version > 0:
                lora_path = f"{LORA_WEIGHT_DIR}/v{self.lora_version}"
                lora_request = LoRARequest(
                    lora_name="sdpo_adapter",
                    lora_int_id=self.lora_version,
                    lora_path=lora_path,
                )

            outputs = self.vllm_engine.generate(
                [prompt], sampling_params, lora_request=lora_request
            )
            generated_text = outputs[0].outputs[0].text
            # Strip Goedel/Qwen special tokens that may appear in .text (defensive)
            generated_text = self._strip_special_tokens_from_generation(generated_text)

            generated_ids = self.tokenizer(
                generated_text, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]

            return generated_text, generated_ids

        def _save_lora_and_bump_version(self):
            """Save current LoRA weights to disk and increment version counter.
            
            Called after each optimizer.step(). The new version is picked up
            by vLLM on the next _generate_proof call via LoRARequest.
            """
            import os
            self.lora_version += 1
            save_path = f"{LORA_WEIGHT_DIR}/v{self.lora_version}"
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"  LoRA weights saved (v{self.lora_version}) -> {save_path}")
        
        @staticmethod
        def _strip_special_tokens_from_generation(text: str) -> str:
            """Strip Goedel/Qwen special tokens from vLLM output."""
            if not text:
                return text
            for tok in ("<|im_end|>", "<|endoftext|>"):
                if text.endswith(tok):
                    text = text[: -len(tok)].rstrip()
            if text.startswith("<|im_start|>assistant"):
                text = text[len("<|im_start|>assistant"):].lstrip("\n").lstrip()
            elif text.startswith("<|im_start|>"):
                text = text[len("<|im_start|>"):].lstrip("\n").lstrip()
            return text.strip()

        @staticmethod
        def _extract_tactics_from_code_block(block: str) -> str:
            """Delegate to module-level extraction."""
            return _extract_tactics_from_code_block(block)

        def _extract_proof_tactics(self, output: str) -> str:
            """Delegate to module-level extraction."""
            return _extract_proof_tactics(output)
        
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
            
            # Use model-generated imports; fall back to default header if none
            if model_imports:
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
            
            # Save final LoRA weights
            model_save_dir = run_dir / "final_lora"
            model_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving final LoRA weights to {model_save_dir}...")
            self.model.save_pretrained(model_save_dir)
            self.tokenizer.save_pretrained(model_save_dir)
            print(f"LoRA weights saved to {model_save_dir}")
            
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
    # Local Entrypoint
    # ========================================================================

    DEFAULT_THEOREM_FIELDS = [
        "lean4_code", "formal_statement", "lean4_statement",
        "statement", "code", "theorem", "problem_statement"
    ]
    DEFAULT_INFORMAL_FIELDS = [
        "informal_prefix", "informal_stmt", "problem", "informal_statement",
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
        model: str = "Goedel-LM/Goedel-Prover-V2-8B",
        dataset: str = "cat-searcher/minif2f-lean4",
        dataset_subset: str = "",
        dataset_split: str = "test",
        problem_idx: int = 0,
        max_iterations: int = 5,
        learning_rate: float = 1e-5,
        temperature: float = 0.6,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        gradient_accumulation_steps: int = 4,
        feedback_errors_only: bool = True,
        default_header: str = "",
        theorem_field: str = "",
        informal_field: str = "",
        header_field: str = "",
    ):
        """Run Goedel-8B SDPO with LoRA on Modal.

        Uses Unsloth for 4-bit LoRA training + vLLM for inference on A100-40GB.
        """
        from datasets import load_dataset

        print("=" * 60)
        print("Goedel-8B SDPO (Unsloth LoRA) on Modal")
        print("=" * 60)
        print(f"Model:          {model}")
        print(f"LoRA:           rank={lora_rank}, alpha={lora_alpha}")
        print(f"Grad accum:     {gradient_accumulation_steps}")
        print(f"Dataset:        {dataset}")
        if dataset_subset:
            print(f"Dataset subset: {dataset_subset}")
        print(f"Split:          {dataset_split}")
        print(f"Problem index:  {problem_idx}")
        print(f"Max iterations: {max_iterations}")
        print(f"Feedback mode:  {'errors only' if feedback_errors_only else 'errors + proofs'}")
        print("=" * 60)

        # ---- Load dataset ----
        print(f"\nLoading dataset {dataset}...")
        try:
            if dataset_subset:
                ds = load_dataset(dataset, dataset_subset, split=dataset_split)
            else:
                ds = load_dataset(dataset, split=dataset_split)
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            try:
                ds = load_dataset(dataset, split=dataset_split, trust_remote_code=True)
            except Exception:
                full_ds = load_dataset(dataset, trust_remote_code=True)
                if dataset_split in full_ds:
                    ds = full_ds[dataset_split]
                else:
                    available = list(full_ds.keys())
                    print(f"Split '{dataset_split}' not found. Available: {available}")
                    ds = full_ds[available[0]]

        print(f"Dataset loaded: {len(ds)} examples, columns: {ds.column_names}")

        if problem_idx >= len(ds):
            print(f"Index {problem_idx} out of range, using 0")
            problem_idx = 0

        problem = dict(ds[problem_idx])

        # ---- Resolve field names ----
        theorem_fields = ([theorem_field] if theorem_field else []) + DEFAULT_THEOREM_FIELDS
        informal_fields = ([informal_field] if informal_field else []) + DEFAULT_INFORMAL_FIELDS
        header_fields = ([header_field] if header_field else []) + DEFAULT_HEADER_FIELDS
        id_fields = DEFAULT_ID_FIELDS

        problem_id = get_field(problem, id_fields, f"problem_{problem_idx}")
        lean4_code = get_field(problem, theorem_fields)
        print(f"\nProblem {problem_idx} ({problem_id}):")
        if lean4_code:
            print(f"  Lean4: {lean4_code[:200]}...")
        else:
            print(f"  WARNING: No theorem code found! Fields: {list(problem.keys())}")

        # ---- Build config ----
        config_dict = {
            "model_name": model,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "dataset_name": dataset,
            "dataset_subset": dataset_subset or None,
            "dataset_split": dataset_split,
            "problem_idx": problem_idx,
            "max_iterations": max_iterations,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "feedback_include_failed_proof": not feedback_errors_only,
            "theorem_fields": theorem_fields,
            "informal_fields": informal_fields,
            "header_fields": header_fields,
            "id_fields": id_fields,
        }
        if default_header:
            config_dict["default_header"] = default_header

        # ---- Launch trainer ----
        trainer = SDPOTrainer(
            model_name=model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        print("\nStarting SDPO training on Modal...")
        results = trainer.run_sdpo.remote(config_dict, problem)

        # ---- Print results ----
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Success: {results['success']}")
        print(f"Iterations: {len(results['iteration_logs'])}")

        if results["success"]:
            print(f"Best proof: {results['best_proof'][:200]}...")

        if results["metrics"]["losses"]:
            print(f"\nFinal metrics:")
            print(f"  Loss:      {results['metrics']['losses'][-1]:.4f}")
            print(f"  Entropy:   {results['metrics']['entropies'][-1]:.4f}")
            print(f"  Grad norm: {results['metrics']['grad_norms'][-1]:.4f}")

        # ---- Save local copy ----
        import json
        from pathlib import Path
        from datetime import datetime

        local_output_dir = Path("sdpo_results") / "goedel_8b"
        local_output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = local_output_dir / f"run_{problem_idx}_{timestamp}"
        run_dir.mkdir(exist_ok=True)

        with open(run_dir / "logs.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        metrics = results.get("metrics", {})
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if metrics.get("iterations"):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Goedel-8B SDPO - Problem {problem_idx}", fontsize=12)

            for ax, key, color, label in [
                (axes[0, 0], "losses",    "b", "Loss"),
                (axes[0, 1], "grad_norms","r", "Gradient Norm"),
                (axes[1, 0], "entropies", "g", "Entropy"),
                (axes[1, 1], "kl_divs",   "m", "KL Divergence"),
            ]:
                ax.plot(metrics["iterations"], metrics[key],
                        f"{color}-o", linewidth=2, markersize=8)
                ax.set_xlabel("Iteration")
                ax.set_ylabel(label)
                ax.set_title(label)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(run_dir / "training_curves.png", dpi=150, bbox_inches="tight")
            plt.close()

        print(f"\nLocal results saved to: {run_dir}")
        print(f"Remote results in Modal volume 'sdpo-output' under '{SDPOConfig.output_dir}/'")
        print("=" * 60)

        return results

except ImportError:
    print("Modal not installed. Install with: pip install modal")

    def main():
        print("This script requires Modal. Install with: pip install modal")
        print("Then run: modal run lean_sdpo_goedel_8b_modal.py --problem-idx 0")


if __name__ == "__main__":
    import sys
    if "--test-extraction" in sys.argv:
        print("Running tactic extraction sanity tests...")
        _run_extraction_tests()
    else:
        main()
