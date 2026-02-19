"""
Test-Time RL for Lean Theorem Proving (SDPO) on Modal

This script runs SDPO (Self-Distilled Policy Optimization) on Modal with:
- Model inference on GPU
- Lean verification via Kimina server (also on Modal)
- Model-agnostic framework (any HF model)
- Dataset-agnostic (any HF dataset with lean4_code field)
- Single problem test-time RL
- Comprehensive logging and visualization

Usage:
    # Run on Modal with default settings
    modal run lean_sdpo_modal.py --model AI-MO/Kimina-Prover-RL-1.7B --problem-idx 0

    # Custom dataset and model
    modal run lean_sdpo_modal.py --model Goedel-LM/Goedel-Prover-V2-8B --dataset deepmind/math --problem-idx 5

    # More iterations
    modal run lean_sdpo_modal.py --max-iterations 10 --problem-idx 0
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
    
    # Generation settings
    max_new_tokens: int = 4096  # Increased for thinking models that need more tokens
    temperature: float = 0.6
    top_p: float = 0.95
    
    # Test-time RL settings
    max_iterations: int = 5
    learning_rate: float = 1e-5
    distillation_topk: int = 20
    
    # Feedback mode
    feedback_include_failed_proof: bool = False
    feedback_template: str = """
The previous proof attempt failed with the following Lean compiler error:

{feedback}

Failed proof:
{failed_proof}

Please analyze the error and provide a corrected proof.
"""
    feedback_template_errors_only: str = """
The previous proof attempt failed with the following Lean compiler error:

{feedback}

Please analyze the error and provide a corrected proof.
"""
    feedback_separator: str = "\n\n---\n\n"
    
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
    
    # Image for inference + training (GPU)
    inference_image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch",
            "transformers>=4.40.0",
            "accelerate",
            "sentencepiece",
            "protobuf",
            "datasets",
            "matplotlib",
        )
    )
    
    # Image for Lean verification with Kimina client + local Lean fallback
    lean_image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("curl", "git")
        .run_commands(
            "curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain leanprover/lean4:v4.8.0",
        )
        .env({"PATH": "/root/.elan/bin:$PATH"})
        .run_commands(
            "mkdir -p /lean_project && cd /lean_project && lake init lean_project math",
        )
        .run_commands(
            "cd /lean_project && lake update || true",
            "cd /lean_project && lake exe cache get || true",
        )
        .pip_install("kimina-client")
    )

    # ========================================================================
    # Lean Verification Service (Kimina server with local Lean fallback)
    # ========================================================================
    
    @app.cls(
        image=lean_image,
        timeout=600,
        scaledown_window=300,
    )
    class LeanVerifier:
        """Verify Lean 4 proofs using Kimina server, with local Lean fallback."""
        
        KIMINA_API_URL: str = "https://projectnumina.ai"
        LEAN_PROJECT_DIR: str = "/lean_project"
        
        @modal.enter()
        def setup(self):
            """Initialize Kimina client and verify local Lean is available."""
            import os
            import subprocess
            
            # Try to initialize Kimina client
            try:
                from kimina_client import KiminaClient
                api_url = os.environ.get("KIMINA_API_URL", self.KIMINA_API_URL)
                api_key = os.environ.get("KIMINA_API_KEY", os.environ.get("LEAN_SERVER_API_KEY", None))
                self.kimina_client = KiminaClient(api_url=api_url, api_key=api_key)
                print(f"Kimina client initialized (server: {api_url})")
            except Exception as e:
                print(f"Kimina client initialization failed: {e}")
                self.kimina_client = None
            
            # Verify local Lean is available as fallback
            env = {**os.environ, "PATH": "/root/.elan/bin:" + os.environ.get("PATH", "")}
            result = subprocess.run(["lean", "--version"], capture_output=True, text=True, env=env)
            print(f"Local Lean version: {result.stdout.strip()}")

        def _verify_with_kimina(self, lean_code: str) -> dict:
            """Try to verify with Kimina server."""
            import time
            
            if not self.kimina_client:
                return None
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    result = self.kimina_client.check(lean_code)
                    
                    # Parse response
                    if isinstance(result, dict):
                        messages = result.get('messages', []) or []
                        sorries = result.get('sorries', []) or []
                        error = result.get('error')
                        if error and ("Request failed" in str(error) or "405" in str(error)):
                            if attempt < max_retries - 1:
                                time.sleep(1)
                                continue
                            return None  # Fall back to local
                    elif hasattr(result, 'results') and result.results:
                        first_result = result.results[0]
                        if hasattr(first_result, 'error') and first_result.error:
                            error_str = str(first_result.error)
                            if "Request failed" in error_str or "405" in error_str:
                                if attempt < max_retries - 1:
                                    time.sleep(1)
                                    continue
                                return None
                        messages = getattr(first_result, 'messages', []) or []
                        sorries = getattr(first_result, 'sorries', []) or []
                    else:
                        messages = getattr(result, 'messages', []) or []
                        sorries = getattr(result, 'sorries', []) or []
                    
                    # Check for errors
                    errors = []
                    for msg in messages:
                        if isinstance(msg, dict) and msg.get('severity') == 'error':
                            errors.append(msg.get('data', str(msg)))
                        elif hasattr(msg, 'severity') and getattr(msg, 'severity') == 'error':
                            errors.append(getattr(msg, 'data', str(msg)))
                    
                    has_error = len(errors) > 0
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
                    }
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return None
            return None

        def _verify_with_local_lean(self, lean_code: str) -> dict:
            """Verify with local Lean compiler."""
            import subprocess
            import os
            
            env = {**os.environ, "PATH": "/root/.elan/bin:" + os.environ.get("PATH", "")}
            
            try:
                proof_path = os.path.join(self.LEAN_PROJECT_DIR, "Proof.lean")
                with open(proof_path, "w") as f:
                    f.write(lean_code)
                
                result = subprocess.run(
                    ["lake", "env", "lean", "Proof.lean"],
                    cwd=self.LEAN_PROJECT_DIR,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env,
                )
                
                stderr = result.stderr or ""
                stdout = result.stdout or ""
                has_error = result.returncode != 0
                
                # Collect all error-related output
                errors = []
                if has_error:
                    # Lean 4 errors appear in stderr with specific patterns
                    for line in stderr.split("\n"):
                        line = line.strip()
                        if not line or line.startswith("info:"):
                            continue
                        # Include lines with error indicators or file:line:col format
                        if "error" in line.lower() or "Proof.lean:" in line or "unexpected" in line.lower():
                            errors.append(line)
                    # Also check stdout for errors (some Lean versions output there)
                    for line in stdout.split("\n"):
                        line = line.strip()
                        if "error" in line.lower() or "Proof.lean:" in line:
                            errors.append(line)
                    # If still no errors captured, include raw stderr
                    if not errors:
                        combined = (stderr + "\n" + stdout).strip()
                        if combined:
                            errors = [combined[:1000]]
                        else:
                            errors = [f"Lean compilation failed with exit code {result.returncode}"]
                
                has_sorry = "sorry" in lean_code.lower()
                feedback = "\n".join(errors) if errors else ""
                
                return {
                    "success": not has_error,
                    "complete": not has_error and not has_sorry,
                    "has_sorry": has_sorry,
                    "feedback": feedback,
                    "errors": errors,
                    "messages": [],
                    "sorries": [],
                    "source": "local",
                    "stdout": stdout[:500] if stdout else "",
                    "stderr": stderr[:500] if stderr else "",
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "complete": False,
                    "has_sorry": "sorry" in lean_code.lower(),
                    "feedback": "Lean verification timed out (120s)",
                    "errors": ["Timeout"],
                    "messages": [],
                    "sorries": [],
                    "source": "local",
                }
            except Exception as e:
                return {
                    "success": False,
                    "complete": False,
                    "has_sorry": "sorry" in lean_code.lower(),
                    "feedback": f"Local verification error: {str(e)}",
                    "errors": [str(e)],
                    "messages": [],
                    "sorries": [],
                    "source": "local",
                }

        @modal.method()
        def verify(self, lean_code: str) -> dict:
            """Verify Lean code - try Kimina first, fall back to local Lean."""
            # Try Kimina first (faster if available)
            result = self._verify_with_kimina(lean_code)
            if result is not None:
                return result
            
            # Fall back to local Lean
            print("Kimina unavailable, using local Lean verification...")
            return self._verify_with_local_lean(lean_code)

    # ========================================================================
    # SDPO Training Service (GPU)
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
        """SDPO trainer running on Modal GPU."""
        
        model_name: str = modal.parameter(default="AI-MO/Kimina-Prover-RL-1.7B")
        
        @modal.enter()
        def setup(self):
            """Load model and tokenizer on container start."""
            import os
            os.environ["HF_HOME"] = "/cache"
            if os.environ.get("HF_TOKEN"):
                os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
            
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading model: {self.model_name}...")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.train()
            
            self.initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            print(f"Model loaded on {self.device}!")
        
        def reload_model(self):
            """Reload model to initial weights."""
            self.model.load_state_dict(self.initial_state)
        
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
            
            for iteration in range(config.max_iterations):
                iter_start = time.time()
                print(f"\n--- Iteration {iteration + 1}/{config.max_iterations} ---")
                
                base_prompt = self._create_base_prompt(config, problem)
                raw_output, generated_ids = self._generate_proof(config, base_prompt)
                
                print(f"  Generated {len(raw_output)} chars")
                
                tactics = self._extract_proof_tactics(raw_output)
                lean4_code = problem.get("lean4_code", problem.get("formal_statement", ""))
                header = problem.get("header", "")
                full_code = self._create_full_lean_code(lean4_code, tactics, header)
                
                verification = LeanVerifier().verify.remote(full_code)
                is_success = verification["success"] and verification["complete"]
                
                print(f"  Verification: {'SUCCESS' if is_success else 'FAILED'}")
                
                iter_log = {
                    "iteration": iteration + 1,
                    "prompt": base_prompt[:500] + "..." if len(base_prompt) > 500 else base_prompt,
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
                
                feedback = verification["feedback"] or "Proof verification failed"
                feedback_history.append((feedback, tactics))
                
                feedback_prompt = self._create_feedback_prompt(
                    config, problem, feedback_history
                )
                
                per_token_kl, reward, avg_kl, entropy = self._compute_sdpo_loss(
                    config, base_prompt, feedback_prompt, generated_ids
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
            
            # Save results, plots, and final model weights BEFORE reloading
            model_save_path = self._save_results(config, logs, metrics)
            logs["model_save_path"] = model_save_path
            
            self.reload_model()
            
            return logs
        
        def _create_base_prompt(self, config: SDPOConfig, theorem: dict) -> str:
            """Create base prompt WITHOUT feedback.
            
            Kimina-Prover is a thinking model - it outputs <think>...</think> then the answer.
            We allow this and extract tactics from the output.
            """
            lean4_code = theorem.get("lean4_code", theorem.get("formal_statement", ""))
            informal = theorem.get("informal_prefix", theorem.get("problem", ""))
            
            user_content = f"Prove the following Lean 4 theorem.\n\n"
            
            if informal:
                user_content += f"Problem: {informal}\n\n"
            
            user_content += f"```lean4\n{lean4_code}\n```\n\n"
            user_content += "After your reasoning, output ONLY the proof tactics (not the full theorem) in a ```lean4 code block. "
            user_content += "The tactics should replace `sorry`. Do NOT include `import`, `theorem`, or `:= sorry` in your final answer."
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": "You are an expert Lean 4 theorem prover. Output proof tactics that can replace `sorry`."},
                    {"role": "user", "content": user_content}
                ]
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except:
                    prompt = f"System: You are an expert Lean 4 theorem prover.\n\nUser: {user_content}\n\nAssistant:"
            else:
                prompt = f"System: You are an expert Lean 4 theorem prover.\n\nUser: {user_content}\n\nAssistant:"
            
            return prompt
        
        def _create_feedback_prompt(
            self,
            config: SDPOConfig,
            theorem: dict,
            feedback_history: list[tuple[str, str]],
        ) -> str:
            """Create prompt WITH accumulated feedback."""
            lean4_code = theorem.get("lean4_code", theorem.get("formal_statement", ""))
            informal = theorem.get("informal_prefix", theorem.get("problem", ""))
            
            user_content = f"Prove the following Lean 4 theorem. Output the proof tactics after `by`.\n\n"
            
            if informal:
                user_content += f"Problem: {informal}\n\n"
            
            user_content += f"```lean4\n{lean4_code}\n```\n\n"
            
            # Add feedback from previous attempts
            if config.feedback_include_failed_proof:
                blocks = [
                    config.feedback_template.format(feedback=f, failed_proof=p)
                    for f, p in feedback_history
                ]
            else:
                blocks = [
                    config.feedback_template_errors_only.format(feedback=f)
                    for f, p in feedback_history
                ]
            user_content += config.feedback_separator.join(blocks)
            user_content += "\n\nProvide corrected proof tactics:"
            
            if hasattr(self.tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": "You are an expert Lean 4 theorem prover. After reasoning, output the proof tactics in a ```lean4 code block."},
                    {"role": "user", "content": user_content}
                ]
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except:
                    prompt = f"System: You are an expert Lean 4 theorem prover.\n\nUser: {user_content}\n\nAssistant:"
            else:
                prompt = f"System: You are an expert Lean 4 theorem prover.\n\nUser: {user_content}\n\nAssistant:"
            
            return prompt
        
        def _generate_proof(self, config: SDPOConfig, prompt: str) -> tuple[str, "torch.Tensor"]:
            """Generate a proof."""
            import torch
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=2048
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
                
                prompt_len = inputs["input_ids"].shape[1]
                generated_ids = outputs.sequences[0, prompt_len:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # For thinking models, we want the full output including </think>
                # Only truncate at end-of-message tokens
                stop_strings = ["<|im_end|>", "<|endoftext|>"]
                for stop in stop_strings:
                    if stop in generated_text:
                        generated_text = generated_text.split(stop)[0]
                        break
            
            return generated_text, generated_ids
        
        def _extract_tactics_from_code_block(self, block: str) -> str:
            """Extract tactics from a single code block, filtering out non-tactic lines."""
            lines = []
            for line in block.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                # Skip imports
                if stripped.startswith("import "):
                    continue
                # Skip theorem/lemma declarations
                if stripped.startswith("theorem ") or stripped.startswith("lemma "):
                    continue
                # Skip open statements
                if stripped.startswith("open "):
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
            4. Fallback tactics for common proof patterns
            """
            output = output.strip()
            tactics = None
            
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
            
            # If no tactics extracted, try to infer from the reasoning
            if not tactics:
                # Look for specific patterns in the thinking that suggest tactics
                lower_output = output.lower()
                if "modulo" in lower_output or "mod " in lower_output or "≡" in output:
                    # Modular arithmetic problem - try decide or omega
                    tactics = "decide"
                elif "contradiction" in lower_output:
                    tactics = "contradiction"
                elif "induction" in lower_output:
                    tactics = "induction"
                elif "simp" in lower_output:
                    tactics = "simp"
            
            return tactics if tactics else "sorry"
        
        def _create_full_lean_code(self, theorem_code: str, proof_tactics: str, header: str = "") -> str:
            """Create full Lean 4 code by replacing sorry with proof tactics."""
            proof_lines = proof_tactics.split("\n")
            indented_proof = "\n  ".join(proof_lines)
            
            if ":= by sorry" in theorem_code:
                theorem_with_proof = theorem_code.replace(":= by sorry", f":= by\n  {indented_proof}")
            elif ":= by\n  sorry" in theorem_code:
                theorem_with_proof = theorem_code.replace("sorry", indented_proof)
            else:
                theorem_with_proof = theorem_code.replace("sorry", indented_proof)
            
            # Include header (imports) if provided
            if header:
                return f"{header}\n\n{theorem_with_proof}"
            return theorem_with_proof
        
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
            feedback_prompt: str,
            generated_ids: "torch.Tensor",
        ) -> tuple["torch.Tensor", float, float, float]:
            """Compute SDPO loss with entropy tracking."""
            import torch
            import torch.nn.functional as F
            
            K = config.distillation_topk
            model_device = next(self.model.parameters()).device
            
            base_prompt_ids = self.tokenizer(
                base_prompt, return_tensors="pt", truncation=True, max_length=2048
            ).input_ids.to(model_device)
            
            feedback_prompt_ids = self.tokenizer(
                feedback_prompt, return_tensors="pt", truncation=True, max_length=2048
            ).input_ids.to(model_device)
            
            response_ids = generated_ids.to(model_device)
            if response_ids.dim() == 1:
                response_ids = response_ids.unsqueeze(0)
            
            base_input_ids = torch.cat([base_prompt_ids, response_ids], dim=1)
            feedback_input_ids = torch.cat([feedback_prompt_ids, response_ids], dim=1)
            
            base_prompt_len = base_prompt_ids.shape[1]
            feedback_prompt_len = feedback_prompt_ids.shape[1]
            seq_len = response_ids.shape[1]
            
            student_logits = self.model(
                input_ids=base_input_ids,
            ).logits[0, base_prompt_len - 1 : base_prompt_len - 1 + seq_len]
            
            student_probs = F.softmax(student_logits, dim=-1)
            entropy = -(student_probs * torch.log(student_probs + 1e-10)).sum(dim=-1).mean().item()
            
            K_actual = min(K, student_logits.size(-1))
            student_topk_logits, topk_indices = torch.topk(student_logits, K_actual, dim=-1)
            student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
            student_topk_logps = student_topk_logits - student_logsumexp
            
            with torch.no_grad():
                teacher_logits = self.model(
                    input_ids=feedback_input_ids,
                ).logits[0, feedback_prompt_len - 1 : feedback_prompt_len - 1 + seq_len]
                
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
    # Local Entrypoint
    # ========================================================================
    
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
    ):
        """
        Run SDPO test-time RL on Modal.
        
        Args:
            model: HuggingFace model ID
            dataset: HuggingFace dataset ID
            dataset_subset: Dataset subset (if any)
            dataset_split: Dataset split (train/test/validation)
            problem_idx: Index of problem to solve
            max_iterations: Max RL iterations
            learning_rate: Learning rate for updates
            temperature: Sampling temperature
            feedback_errors_only: Only include error messages in feedback (not failed proofs)
        """
        from datasets import load_dataset
        
        print("="*60)
        print("SDPO Test-Time RL on Modal")
        print("="*60)
        print(f"Model: {model}")
        print(f"Dataset: {dataset}")
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
            ds = load_dataset(dataset, split=dataset_split, trust_remote_code=True)
        
        if problem_idx >= len(ds):
            print(f"Problem index {problem_idx} out of range (dataset has {len(ds)} examples)")
            problem_idx = 0
        
        problem = dict(ds[problem_idx])
        print(f"\nLoaded problem {problem_idx}:")
        problem_id = problem.get("problem_id", problem.get("name", f"problem_{problem_idx}"))
        print(f"  ID: {problem_id}")
        
        lean4_code = problem.get("lean4_code", problem.get("formal_statement", ""))
        print(f"  Lean4 code: {lean4_code[:200]}...")
        
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
        }
        
        print("\nStarting SDPO training on Modal...")
        trainer = SDPOTrainer(model_name=model)
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
