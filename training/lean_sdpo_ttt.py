"""
Test-Time RL for Lean Theorem Proving (SDPO)

This script implements Self-Distilled Policy Optimization (SDPO) for Lean 4 proofs.
The key insight is that the model distills knowledge from itself when given feedback.

SDPO Gradient (Proposition 2.1):
================================
∇L_SDPO(θ) = E[Σ_t log(π_θ(ŷ_t|x,y_{<t}) / π_θ(ŷ_t|x,f,y_{<t})) · ∇_θ log π_θ(ŷ_t|x,y_{<t})]

Where:
- π_θ(ŷ_t|x,y_{<t}): Model probability WITHOUT feedback (numerator)
- π_θ(ŷ_t|x,f,y_{<t}): Model probability WITH feedback f (denominator)
- The log ratio is the reward signal
- Same model, different prompts (self-distillation)

Usage:
    python lean_sdpo_ttt.py --model Qwen/Qwen3-1.6B --n_problems 10
    python lean_sdpo_ttt.py --model AI-MO/Kimina-Prover-RL-1.7B --max_iterations 5

Key Implementations:
===================

1. TEST-TIME RL LOOP (solve_theorem method):
   For each problem:
     For each iteration (until success or max_iterations):
       a) Generate ONE proof with model (prompt WITHOUT feedback)
       b) Verify with Lean compiler
       c) If correct: terminate, record steps needed
       d) If incorrect: extract error message f
       e) Compute reward: log π(y|x) - log π(y|x,f) for each token
       f) RL update using SDPO gradient
     After max_iterations: reload model weights

2. SDPO LOSS (compute_sdpo_loss method, Eq. 11):
   - Top-K + tail KL: KL(π_θ(·|x,y_{<t}) ‖ stopgrad(q_θ(·|x,f,y_{<t})))
   - Only top-K tokens from student + a tail bucket (never full vocab softmax)
   - Student = model without feedback, Teacher = model with feedback (stopgraded)

3. RL UPDATE (rl_update method):
   - Loss = mean per-token KL (direct gradient through student distribution)
   - Updates model weights
   - Model reloaded after max_iterations to prevent drift

4. INTUITION:
   - High KL: student and teacher distributions differ a lot (proof needs feedback)
   - Low KL: student already matches teacher (proof is good without feedback)
   - Minimizing KL pushes student toward teacher, internalizing the feedback
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plotting disabled.")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SDPOConfig:
    """Configuration for SDPO (Self-Distilled Policy Optimization)."""
    # Model settings (single model for self-distillation)
    model_name: str = "AI-MO/Kimina-Prover-RL-1.7B"
    kimina_server_url: str = "http://localhost:80"
    
    # Generation settings
    max_new_tokens: int = 4096  # Max tokens for proof generation
    temperature: float = 0.6
    top_p: float = 0.95
    
    # Test-time RL settings
    max_iterations: int = 5  # Max iterations per problem before model reload
    learning_rate: float = 1e-5  # Learning rate for RL updates
    distillation_topk: int = 20  # Top-K tokens for KL approximation (Eq. 11)
    
    # Logging settings
    log_dir: str = "logs/sdpo_runs"
    
    # Whether to use the model's chat template
    use_chat_template: bool = True
    
    # Feedback mode for RL denominator prompt:
    # - True: pass both past error messages AND failed proofs (full context)
    # - False: pass only past error messages (no failed proof text)
    feedback_include_failed_proof: bool = True
    # Template when including both error + failed proof
    feedback_template: str = """
The previous proof attempt failed with the following Lean compiler error:

{feedback}

Failed proof:
{failed_proof}

Please analyze the error and provide a corrected proof.
"""
    # Template when including only error messages (no failed proof)
    feedback_template_errors_only: str = """
The previous proof attempt failed with the following Lean compiler error:

{feedback}

Please analyze the error and provide a corrected proof.
"""
    # Separator between multiple feedback blocks in the prompt
    feedback_separator: str = "\n\n---\n\n"


# ============================================================================
# Metrics Tracker
# ============================================================================

class MetricsTracker:
    """Track and log training metrics for visualization."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            "iteration": [],
            "problem_id": [],
            "kl_divergence": [],
            "reward": [],
            "success": [],
            "cumulative_accuracy": [],
            "sample_idx": [],
        }
        
        # Per-problem tracking
        self.problem_metrics = {}
        
        # Global counters
        self.total_attempts = 0
        self.total_successes = 0
        self.global_step = 0
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup file and console logging."""
        self.logger = logging.getLogger(f"SDPO_{self.run_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # File handler - detailed logs
        fh = logging.FileHandler(self.run_dir / "training.log")
        fh.setLevel(logging.DEBUG)
        fh_format = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fh_format)
        self.logger.addHandler(fh)
        
        # Console handler - info level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_format = logging.Formatter('%(message)s')
        ch.setFormatter(ch_format)
        self.logger.addHandler(ch)
        
        self.logger.info(f"Logging to {self.run_dir}")
    
    def log_sample(
        self,
        problem_id: str,
        iteration: int,
        sample_idx: int,
        reward: float,
        kl_div: float,
        success: bool,
        proof: str = "",
        feedback: str = "",
    ):
        """Log metrics for a single sample."""
        self.global_step += 1
        self.total_attempts += 1
        if success:
            self.total_successes += 1
        
        cumulative_acc = self.total_successes / self.total_attempts
        
        # Store metrics
        self.metrics["iteration"].append(iteration)
        self.metrics["problem_id"].append(problem_id)
        self.metrics["kl_divergence"].append(kl_div)
        self.metrics["reward"].append(reward)
        self.metrics["success"].append(int(success))
        self.metrics["cumulative_accuracy"].append(cumulative_acc)
        self.metrics["sample_idx"].append(sample_idx)
        
        # Log to file
        self.logger.debug(
            f"Step {self.global_step} | Problem: {problem_id} | "
            f"Iter: {iteration} | Sample: {sample_idx} | "
            f"Reward: {reward:.2f} | KL: {kl_div:.4f} | "
            f"Success: {success} | CumAcc: {cumulative_acc:.3f}"
        )
        
        if feedback and not success:
            self.logger.debug(f"  Feedback: {feedback[:200]}...")
        if proof and success:
            self.logger.debug(f"  Proof: {proof[:200]}...")
    
    def log_problem_summary(
        self,
        problem_id: str,
        solved: bool,
        iterations_used: int,
        total_samples: int,
        best_proof: Optional[str] = None,
    ):
        """Log summary for a completed problem."""
        self.problem_metrics[problem_id] = {
            "solved": solved,
            "iterations": iterations_used,
            "samples": total_samples,
            "best_proof": best_proof,
        }
        
        self.logger.info(
            f"{'✓' if solved else '✗'} {problem_id}: "
            f"{'SOLVED' if solved else 'FAILED'} in {iterations_used} iterations, "
            f"{total_samples} samples"
        )
    
    def save_metrics(self):
        """Save all metrics to JSON."""
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        summary_path = self.run_dir / "problem_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.problem_metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to {self.run_dir}")
    
    def plot_metrics(self):
        """Generate and save metric plots."""
        if not HAS_MATPLOTLIB:
            self.logger.warning("matplotlib not available, skipping plots")
            return
        
        if len(self.metrics["iteration"]) == 0:
            self.logger.warning("No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"SDPO Training Metrics - Run {self.run_id}", fontsize=14)
        
        steps = list(range(len(self.metrics["reward"])))
        
        # Plot 1: Reward over steps
        ax1 = axes[0, 0]
        ax1.plot(steps, self.metrics["reward"], 'b-', alpha=0.6, label="Reward")
        # Add rolling average
        window = min(10, len(steps) // 3) if len(steps) > 3 else 1
        if window > 1:
            rolling_reward = self._rolling_average(self.metrics["reward"], window)
            ax1.plot(steps, rolling_reward, 'b-', linewidth=2, label=f"Rolling Avg (w={window})")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Reward")
        ax1.set_title("Reward per Sample")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        # Plot 2: KL Divergence over steps
        ax2 = axes[0, 1]
        ax2.plot(steps, self.metrics["kl_divergence"], 'r-', alpha=0.6, label="KL Divergence")
        if window > 1:
            rolling_kl = self._rolling_average(self.metrics["kl_divergence"], window)
            ax2.plot(steps, rolling_kl, 'r-', linewidth=2, label=f"Rolling Avg (w={window})")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("KL Divergence")
        ax2.set_title("KL Divergence (Policy vs Reference)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative Accuracy
        ax3 = axes[1, 0]
        ax3.plot(steps, self.metrics["cumulative_accuracy"], 'g-', linewidth=2)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Cumulative Accuracy")
        ax3.set_title("Cumulative Success Rate")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Plot 4: Success rate by iteration
        ax4 = axes[1, 1]
        iteration_success = {}
        for i, (it, succ) in enumerate(zip(self.metrics["iteration"], self.metrics["success"])):
            if it not in iteration_success:
                iteration_success[it] = {"total": 0, "success": 0}
            iteration_success[it]["total"] += 1
            iteration_success[it]["success"] += succ
        
        iters = sorted(iteration_success.keys())
        success_rates = [
            iteration_success[it]["success"] / iteration_success[it]["total"]
            for it in iters
        ]
        ax4.bar(iters, success_rates, color='purple', alpha=0.7)
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Success Rate")
        ax4.set_title("Success Rate by Iteration")
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.run_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plots saved to {plot_path}")
    
    def _rolling_average(self, values: list, window: int) -> list:
        """Compute rolling average."""
        result = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            result.append(sum(values[start:i+1]) / (i - start + 1))
        return result
    
    def print_final_summary(self):
        """Print final training summary."""
        n_problems = len(self.problem_metrics)
        n_solved = sum(1 for p in self.problem_metrics.values() if p["solved"])
        
        self.logger.info("\n" + "="*60)
        self.logger.info("FINAL SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Problems: {n_solved}/{n_problems} solved ({100*n_solved/n_problems:.1f}%)")
        self.logger.info(f"Total samples: {self.total_attempts}")
        self.logger.info(f"Total successes: {self.total_successes}")
        self.logger.info(f"Overall accuracy: {100*self.total_successes/self.total_attempts:.1f}%")
        self.logger.info(f"Logs saved to: {self.run_dir}")
        self.logger.info("="*60)


# ============================================================================
# Lean Verification via Kimina
# ============================================================================

class LeanVerifier:
    """Verify Lean 4 proofs using Kimina server."""
    
    def __init__(self, server_url: str = "http://localhost:80"):
        self.server_url = server_url
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                from kimina_client import KiminaClient
                self._client = KiminaClient(self.server_url)
            except ImportError:
                raise ImportError(
                    "kimina_client not installed. Install with: pip install kimina-client"
                )
        return self._client
    
    def verify(self, lean_code: str) -> dict:
        """
        Verify a Lean 4 proof and return result with feedback.
        
        Returns:
            dict with keys:
                - success: bool - whether proof compiles without errors
                - complete: bool - whether proof has no sorries
                - feedback: str - compiler error messages (empty if success)
                - errors: list[str] - individual error messages
        """
        try:
            check_response = self.client.check(lean_code)
            
            has_error = False
            has_sorry = False
            errors = []
            sorries = []
            
            for repl_resp in check_response.results:
                resp_dict = repl_resp.response
                if isinstance(resp_dict, dict):
                    messages = resp_dict.get("messages", [])
                    for msg in messages:
                        if msg.get("severity") == "error":
                            has_error = True
                            errors.append(msg.get("data", ""))
                    
                    if resp_dict.get("sorries"):
                        has_sorry = True
                        sorries.extend(resp_dict.get("sorries", []))
                elif hasattr(resp_dict, "model_dump"):
                    d = resp_dict.model_dump()
                    messages = d.get("messages", [])
                    for msg in messages:
                        if msg.get("severity") == "error":
                            has_error = True
                            errors.append(msg.get("data", ""))
                    
                    if d.get("sorries"):
                        has_sorry = True
                        sorries.extend(d.get("sorries", []))
            
            feedback = "\n".join(errors) if errors else ""
            
            return {
                "success": not has_error,
                "complete": not has_error and not has_sorry,
                "has_sorry": has_sorry,
                "feedback": feedback,
                "errors": errors,
                "sorries": sorries,
            }
            
        except Exception as e:
            return {
                "success": False,
                "complete": False,
                "has_sorry": False,
                "feedback": f"Verification error: {str(e)}",
                "errors": [str(e)],
                "sorries": [],
            }


# ============================================================================
# Proof Generation and Extraction
# ============================================================================

def extract_proof_tactics(output: str) -> str:
    """
    Extract proof tactics from model output.
    
    The model may output:
    1. Just tactics: "rw [h₁]\nnorm_num"
    2. Full theorem with tactics after ":= by"
    3. Code in markdown blocks
    4. Thinking blocks followed by code
    
    We need to extract ONLY the proof tactics (the lines after "by").
    """
    output = output.strip()
    
    # Handle Qwen3/Kimina thinking mode - extract content after </think>
    if "</think>" in output:
        output = output.split("</think>")[-1].strip()
    
    # If output is just a <think> block without closing, it's incomplete
    if output.startswith("<think>") and "</think>" not in output:
        return "sorry"
    
    # Remove markdown code blocks if present - extract the code inside
    if "```" in output:
        code_pattern = r"```(?:lean4?|lean)?\n?(.*?)```"
        matches = re.findall(code_pattern, output, re.DOTALL)
        if matches:
            # Take the last code block (usually the actual proof)
            output = matches[-1].strip()
    
    # If output contains ":= by", extract everything after the LAST ":= by"
    # This handles cases where model outputs full theorem
    if ":= by" in output:
        # Find the last occurrence of ":= by"
        last_by_idx = output.rfind(":= by")
        output = output[last_by_idx + 5:].strip()  # +5 for ":= by"
    elif " by\n" in output or " by " in output:
        # Handle "... by\n  tactics" format
        by_match = re.search(r'\bby\s*\n', output)
        if by_match:
            output = output[by_match.end():].strip()
    
    # Remove leading "by" if present (standalone)
    if output.lower().startswith("by\n") or output.lower().startswith("by "):
        output = output[2:].strip()
    elif output.lower() == "by":
        return "sorry"
    
    # Split into lines and filter out non-tactic lines
    lines = output.split('\n')
    
    # Lines to skip (not actual tactics)
    skip_patterns = [
        r'^import\s',           # import statements
        r'^open\s',             # open statements  
        r'^set_option\s',       # set_option directives
        r'^theorem\s',          # theorem declarations
        r'^lemma\s',            # lemma declarations
        r'^def\s',              # def declarations
        r'^/--',                # docstrings
        r'^--',                 # comments
        r'^\s*#',               # Lean commands like #check
        r'^namespace\s',        # namespace
        r'^section\s',          # section
        r'^variable\s',         # variable declarations
        r'^\(h[₀-₉]*\s*:',      # hypothesis declarations like (h₀ : ...)
        r'^where\s',            # where clauses
    ]
    
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        
        # Skip empty lines at the beginning
        if not cleaned_lines and not line_stripped:
            continue
        
        # Skip lines matching skip patterns
        should_skip = False
        for pattern in skip_patterns:
            if re.match(pattern, line_stripped):
                should_skip = True
                break
        
        if should_skip:
            continue
        
        # Skip lines that look like theorem signatures (contain ": ... :=")
        if re.match(r'.*:\s*\w+.*:=', line_stripped):
            continue
        
        # Skip docstring content (lines starting with description text)
        if line_stripped.startswith("The ") or line_stripped.startswith("This "):
            continue
        if line_stripped.startswith("where ") or line_stripped.startswith("Show that"):
            continue
            
        # Remove trailing comma if present
        if line_stripped.endswith(','):
            line_stripped = line_stripped[:-1].strip()
        
        if line_stripped:
            cleaned_lines.append(line_stripped)
    
    # Stop at double newline (often separates proof from explanation)
    result = "\n".join(cleaned_lines).strip()
    if "\n\n" in result:
        result = result.split("\n\n")[0].strip()
    
    # If we still have nothing useful, return sorry
    if not result or result.lower() in ["by", "sorry", ""]:
        return "sorry"
    
    return result


def create_full_lean_code(theorem_code: str, proof_tactics: str) -> str:
    """Create full Lean 4 code by replacing sorry with proof tactics."""
    proof_lines = proof_tactics.split("\n")
    indented_proof = "\n  ".join(proof_lines)
    
    if ":= by sorry" in theorem_code:
        return theorem_code.replace(":= by sorry", f":= by\n  {indented_proof}")
    elif ":= by\n  sorry" in theorem_code:
        return theorem_code.replace("sorry", indented_proof)
    else:
        return theorem_code.replace("sorry", indented_proof)


# ============================================================================
# SDPO Pipeline
# ============================================================================

class LeanSDPO:
    """
    Self-Distilled Policy Optimization (SDPO) for Lean theorem proving.
    
    KEY IMPLEMENTATION DETAILS:
    ==========================
    
    1. SINGLE MODEL (self-distillation):
       - Same model used with two different prompts
       - π(y|x): probability without feedback
       - π(y|x,f): probability with feedback f
       
    2. SDPO GRADIENT (Proposition 2.1):
       reward_t = log π(y_t|x,y_{<t}) - log π(y_t|x,f,y_{<t})
       loss = -Σ_t reward_t · log π(y_t|x,y_{<t})
       
    3. TEST-TIME RL LOOP:
       For each iteration:
         a) Generate proof with prompt (no feedback)
         b) Verify with Lean compiler
         c) If correct: terminate
         d) If incorrect: get error f, compute SDPO reward, update
       After max_iterations: reload model weights
    
    4. INTUITION:
       - High reward: proof is good without feedback (π(y|x) >> π(y|x,f))
       - Low reward: proof needs feedback to be corrected
       - Model learns to generate proofs that don't need correction
    """
    
    def __init__(self, config: SDPOConfig, metrics_tracker: Optional[MetricsTracker] = None):
        self.config = config
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Initialize metrics tracker
        self.metrics = metrics_tracker or MetricsTracker(config.log_dir)
        
        # Use float32 on MPS, bfloat16 on CUDA
        self.dtype = torch.float32 if self.device.type == "mps" else torch.bfloat16
        
        # Load model (single model for self-distillation)
        self.metrics.logger.info(f"Loading model: {config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=self.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.train()  # Enable gradients for RL updates
        
        # Store initial weights for reload
        self.initial_model_state = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        self.verifier = LeanVerifier(config.kimina_server_url)
        self.metrics.logger.info(f"Model loaded on {self.device}")
    
    def reload_model(self):
        """Reload model to initial weights after max_iterations."""
        self.metrics.logger.info("Reloading model to initial weights...")
        self.model.load_state_dict(self.initial_model_state)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def create_base_prompt(self, theorem: dict) -> str:
        """
        Create base prompt WITHOUT feedback.
        Used for: π(y|x) in SDPO formula (numerator)
        """
        lean4_code = theorem.get("lean4_code", theorem.get("formal_statement", ""))
        informal = theorem.get("informal_prefix", "")
        
        user_content = "Think about and solve the following problem step by step in Lean 4."
        
        if informal:
            user_content += f"\n# Problem: {informal}"
        
        user_content += f"\n# Formal statement:\n```lean4\n{lean4_code}\n```\n"
        
        if self.config.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are an expert in mathematics and proving theorems in Lean 4."},
                {"role": "user", "content": user_content}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"""<|im_start|>system
You are an expert in mathematics and proving theorems in Lean 4.
<|im_end|>
<|im_start|>user
{user_content}
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def create_feedback_prompt(
        self,
        theorem: dict,
        feedback: str = "",
        failed_proof: str = "",
        feedback_history: list[tuple[str, str]] | None = None,
    ) -> str:
        """
        Create prompt WITH feedback.
        Used for: π(y|x,f) in SDPO formula (denominator)
        
        Same model, but prompt includes error message(s) and optionally failed proof(s).
        If feedback_history is provided, all entries are appended (accumulated feedback).
        Otherwise feedback/failed_proof are used as a single block (backward compatible).
        When feedback_include_failed_proof is False, only error messages are included.
        """
        lean4_code = theorem.get("lean4_code", theorem.get("formal_statement", ""))
        informal = theorem.get("informal_prefix", "")
        include_proof = self.config.feedback_include_failed_proof
        template_full = self.config.feedback_template
        template_errors = self.config.feedback_template_errors_only

        user_content = "Think about and solve the following problem step by step in Lean 4."
        
        if informal:
            user_content += f"\n# Problem: {informal}"
        
        user_content += f"\n# Formal statement:\n```lean4\n{lean4_code}\n```\n"
        
        # Add feedback block(s): either accumulated history or single (feedback, failed_proof)
        if feedback_history:
            blocks = [
                template_full.format(feedback=f, failed_proof=p) if include_proof
                else template_errors.format(feedback=f)
                for f, p in feedback_history
            ]
            user_content += self.config.feedback_separator.join(blocks)
        else:
            if include_proof:
                user_content += template_full.format(
                    feedback=feedback,
                    failed_proof=failed_proof
                )
            else:
                user_content += template_errors.format(feedback=feedback)
        
        if self.config.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": "You are an expert in mathematics and proving theorems in Lean 4."},
                {"role": "user", "content": user_content}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"""<|im_start|>system
You are an expert in mathematics and proving theorems in Lean 4.
<|im_end|>
<|im_start|>user
{user_content}
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    @torch.no_grad()
    def generate_proof(self, prompt: str) -> tuple[str, torch.Tensor]:
        """
        Generate ONE proof with the model.
        
        Returns:
            tuple: (generated_text, generated_token_ids)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        
        # Get generated token ids (excluding prompt)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[0, prompt_len:]
        
        # Decode to text
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return generated_text, generated_ids
    
    @staticmethod
    def _add_tail(log_probs: torch.Tensor) -> torch.Tensor:
        """Append a tail bucket: log(1 - sum(exp(log_probs))).

        Matches SDPO repo core_algos.compute_self_distillation_loss.add_tail.
        """
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    def compute_sdpo_loss(
        self,
        base_prompt: str,
        feedback_prompt: str,
        generated_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """
        Compute SDPO loss via top-K + tail KL divergence (paper Eq. 11).

        L_SDPO(θ) = Σ_t KL(π_θ(·|x,y_{<t}) ‖ stopgrad(q_θ(·|x,f,y_{<t})))

        Approximated as:
          Σ_t [ Σ_{ŷ ∈ topK(π)} π(ŷ) log(π(ŷ)/sg(q(ŷ)))
                + p_tail · log(p_tail / sg(q_tail)) ]

        Implementation follows SDPO repo (dp_actor + core_algos):
        - Never materializes full (seq_len, vocab_size) log_softmax.
        - Computes logsumexp once, then topk_logits - logsumexp for top-K
          log probs.  Teacher reuses student's top-K indices.

        Args:
            base_prompt: Prompt WITHOUT feedback (x)
            feedback_prompt: Prompt WITH feedback (x, f)
            generated_ids: Token IDs from generation (reused for both passes)

        Returns:
            (per_token_kl, total_reward_scalar, avg_kl_scalar)
        """
        model_device = next(self.model.parameters()).device
        K = self.config.distillation_topk

        # --- Tokenize prompts separately, concat with same generated_ids ---
        base_prompt_ids = self.tokenizer(
            base_prompt, return_tensors="pt", truncation=True, max_length=2048,
        ).input_ids.to(model_device)
        feedback_prompt_ids = self.tokenizer(
            feedback_prompt, return_tensors="pt", truncation=True, max_length=2048,
        ).input_ids.to(model_device)

        response_ids = generated_ids.to(model_device)
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)

        base_input_ids = torch.cat([base_prompt_ids, response_ids], dim=1)
        feedback_input_ids = torch.cat([feedback_prompt_ids, response_ids], dim=1)

        base_prompt_len = base_prompt_ids.shape[1]
        feedback_prompt_len = feedback_prompt_ids.shape[1]
        seq_len = response_ids.shape[1]

        # --- Student forward (with gradients) ---
        # Logits at positions [prompt_len-1 .. prompt_len-1+seq_len) predict
        # the response tokens.
        student_logits = self.model(
            input_ids=base_input_ids,
        ).logits[0, base_prompt_len - 1 : base_prompt_len - 1 + seq_len]
        # (seq_len, vocab_size)

        # Top-K from student — avoids full log_softmax (repo pattern:
        # dp_actor.py lines 530-535).
        K_actual = min(K, student_logits.size(-1))
        student_topk_logits, topk_indices = torch.topk(
            student_logits, K_actual, dim=-1,
        )  # (seq_len, K)
        student_logsumexp = torch.logsumexp(
            student_logits, dim=-1, keepdim=True,
        )  # (seq_len, 1)
        student_topk_logps = student_topk_logits - student_logsumexp  # (seq_len, K)

        # --- Teacher forward (no gradients) ---
        with torch.no_grad():
            teacher_logits = self.model(
                input_ids=feedback_input_ids,
            ).logits[0, feedback_prompt_len - 1 : feedback_prompt_len - 1 + seq_len]

            # Gather teacher logits at student's top-K indices (repo pattern:
            # dp_actor.py line 533).
            teacher_topk_logits = torch.gather(
                teacher_logits, dim=-1, index=topk_indices,
            )
            teacher_logsumexp = torch.logsumexp(
                teacher_logits, dim=-1, keepdim=True,
            )
            teacher_topk_logps = teacher_topk_logits - teacher_logsumexp  # (seq_len, K)

        # --- Add tail bucket (Eq. 11 tail term) ---
        student_with_tail = self._add_tail(student_topk_logps)   # (seq_len, K+1)
        teacher_with_tail = self._add_tail(teacher_topk_logps)   # (seq_len, K+1)

        # --- Reverse KL: KL(π_θ ‖ stopgrad(q_θ)) ---
        # F.kl_div(input, target, log_target=True) = exp(target)*(target - input)
        # so input=teacher (stopgraded), target=student → KL(student ‖ teacher)
        kl_per_bucket = F.kl_div(
            teacher_with_tail.detach(),
            student_with_tail,
            reduction="none",
            log_target=True,
        )  # (seq_len, K+1)
        per_token_kl = kl_per_bucket.sum(dim=-1)  # (seq_len,)

        # Scalar metrics for logging
        with torch.no_grad():
            target_ids = response_ids[0]
            student_lp = -F.cross_entropy(student_logits.detach(), target_ids, reduction="none")
            teacher_lp = -F.cross_entropy(teacher_logits, target_ids, reduction="none")
            total_reward = (student_lp - teacher_lp).sum().item()
            avg_kl = per_token_kl.mean().item()

        return per_token_kl, total_reward, avg_kl
    
    def rl_update(self, per_token_kl: torch.Tensor) -> float:
        """
        Perform SDPO update using top-K + tail KL loss (Eq. 11).

        L = (1/T) Σ_t KL(π_θ(·|x,y_{<t}) ‖ stopgrad(q_θ(·|x,f,y_{<t})))

        The KL already carries correct gradients through the student
        distribution; no REINFORCE-style reward weighting needed.
        """
        loss = per_token_kl.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
    
    def solve_theorem(self, theorem: dict) -> dict:
        """
        Solve a theorem using SDPO (Self-Distilled Policy Optimization).
        
        SDPO LOOP:
        ==========
        
        For each iteration (until success or max_iterations):
          1. Generate ONE proof with base prompt (no feedback)
          2. Verify with Lean compiler
          3. If correct: terminate, record steps needed
          4. If incorrect: extract error message f
          5. Create feedback prompt (with error + failed proof)
          6. Compute SDPO reward: log π(y|x) - log π(y|x,f)
          7. RL update using SDPO gradient
        
        After max_iterations: reload model weights
        """
        problem_id = theorem.get("problem_id", theorem.get("name", "unknown"))
        self.metrics.logger.info(f"\n{'='*60}")
        self.metrics.logger.info(f"Solving: {problem_id}")
        self.metrics.logger.info(f"{'='*60}")
        
        lean4_code = theorem.get("lean4_code", theorem.get("formal_statement", ""))
        best_proof = None
        iteration_results = []
        feedback_history: list[tuple[str, str]] = []  # Accumulate (feedback, failed_proof) across iterations
        
        for iteration in range(self.config.max_iterations):
            self.metrics.logger.info(f"\n--- Iteration {iteration + 1}/{self.config.max_iterations} ---")
            
            # Step 1: Generate ONE proof with base prompt (no feedback)
            base_prompt = self.create_base_prompt(theorem)
            self.metrics.logger.info(f"  Generating proof...")
            raw_output, generated_ids = self.generate_proof(base_prompt)
            
            # Print raw model output for debugging
            self.metrics.logger.info(f"  [RAW OUTPUT] ({len(raw_output)} chars):")
            self.metrics.logger.info(f"  {raw_output[:500]}{'...' if len(raw_output) > 500 else ''}")
            
            # Extract tactics and create full Lean code
            tactics = extract_proof_tactics(raw_output)
            full_code = create_full_lean_code(lean4_code, tactics)
            
            # Print extracted tactics and full code
            self.metrics.logger.info(f"  [EXTRACTED TACTICS]:")
            self.metrics.logger.info(f"  {tactics[:300]}{'...' if len(tactics) > 300 else ''}")
            self.metrics.logger.info(f"  [FULL LEAN CODE]:")
            for line in full_code.split('\n')[:20]:
                self.metrics.logger.info(f"    {line}")
            if full_code.count('\n') > 20:
                self.metrics.logger.info(f"    ... ({full_code.count(chr(10)) - 20} more lines)")
            
            # Step 2: Verify with Lean compiler
            self.metrics.logger.info(f"  Verifying with Lean compiler...")
            result = self.verifier.verify(full_code)
            is_success = result["success"] and result["complete"]
            
            self.metrics.logger.info(f"  [VERIFICATION RESULT]: {'SUCCESS' if is_success else 'FAILED'}")
            
            # Step 3: If correct, terminate
            if is_success:
                best_proof = tactics
                self.metrics.logger.info(f"  ✓ Correct proof found in {iteration + 1} iterations!")
                
                # Log success metrics
                self.metrics.log_sample(
                    problem_id=problem_id,
                    iteration=iteration + 1,
                    sample_idx=0,
                    reward=0.0,
                    kl_div=0.0,
                    success=True,
                    proof=tactics,
                    feedback="",
                )
                
                iteration_results.append({
                    "iteration": iteration + 1,
                    "proof": tactics,
                    "success": True,
                    "feedback": None,
                    "reward": None,
                    "loss": None,
                })
                break
            
            # Step 4: Extract error message and add to accumulated feedback
            feedback = result["feedback"]
            if not feedback:
                feedback = "Proof verification failed (no specific error message)"
            
            self.metrics.logger.info(f"  Error: {feedback[:200]}...")
            
            feedback_history.append((feedback, tactics))
            
            # Step 5: Create feedback prompt (with all accumulated errors + failed proofs)
            feedback_prompt = self.create_feedback_prompt(
                theorem,
                feedback_history=feedback_history,
            )
            
            # Step 6: Compute SDPO loss via top-K + tail KL (Eq. 11)
            per_token_kl, reward_value, avg_kl = self.compute_sdpo_loss(
                base_prompt=base_prompt,
                feedback_prompt=feedback_prompt,
                generated_ids=generated_ids,
            )

            self.metrics.logger.info(f"  SDPO reward (total): {reward_value:.4f}")
            self.metrics.logger.info(f"  SDPO KL (avg/token): {avg_kl:.4f}")

            # Step 7: RL update using top-K + tail KL gradient
            loss = self.rl_update(per_token_kl)
            self.metrics.logger.info(f"  SDPO loss: {loss:.4f}")

            # Log metrics
            self.metrics.log_sample(
                problem_id=problem_id,
                iteration=iteration + 1,
                sample_idx=0,
                reward=reward_value,
                kl_div=avg_kl,
                success=False,
                proof=tactics,
                feedback=feedback,
            )

            iteration_results.append({
                "iteration": iteration + 1,
                "proof": tactics,
                "success": False,
                "feedback": feedback,
                "reward": reward_value,
                "avg_kl": avg_kl,
                "loss": loss,
            })
        
        # After max_iterations: reload model weights
        if best_proof is None:
            self.metrics.logger.info(f"\n  ✗ Failed after {self.config.max_iterations} iterations")
        
        self.reload_model()
        
        # Compute statistics
        iterations_used = len(iteration_results)
        successful = best_proof is not None
        
        # Log problem summary
        self.metrics.log_problem_summary(
            problem_id=problem_id,
            solved=successful,
            iterations_used=iterations_used,
            total_samples=iterations_used,  # One sample per iteration
            best_proof=best_proof,
        )
        
        return {
            "problem_id": problem_id,
            "success": successful,
            "best_proof": best_proof,
            "iterations_used": iterations_used,
            "iteration_results": iteration_results,
        }


# ============================================================================
# Data Loading
# ============================================================================

def load_minif2f_dataset(path: str, n_examples: Optional[int] = None) -> list[dict]:
    """Load MiniF2F dataset from JSONL file."""
    examples = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if n_examples and i >= n_examples:
                break
            examples.append(json.loads(line))
    return examples


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SDPO (Self-Distilled Policy Optimization) for Lean proofs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with 5 problems
  python lean_sdpo_ttt.py --n_problems 5

  # Custom model
  python lean_sdpo_ttt.py --model AI-MO/Kimina-Prover-RL-1.7B --n_problems 10

  # More iterations per problem
  python lean_sdpo_ttt.py --model Qwen/Qwen3-1.6B --max_iterations 5
        """
    )
    parser.add_argument("--model", type=str, default="AI-MO/Kimina-Prover-RL-1.7B",
                        help="Model for self-distillation (default: AI-MO/Kimina-Prover-RL-1.7B)")
    parser.add_argument("--dataset", type=str, 
                        default="Goedel-Prover-V2/dataset/minif2f.jsonl",
                        help="Path to MiniF2F dataset")
    parser.add_argument("--n_problems", type=int, default=5,
                        help="Number of problems to solve (default: 5)")
    parser.add_argument("--max_iterations", type=int, default=5,
                        help="Max iterations per problem before model reload (default: 5)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for RL updates (default: 1e-5)")
    parser.add_argument("--kimina_url", type=str, default="http://localhost:80",
                        help="Kimina server URL (default: http://localhost:80)")
    parser.add_argument("--output", type=str, default="results/sdpo_results.json",
                        help="Output file path")
    parser.add_argument("--log_dir", type=str, default="logs/sdpo_runs",
                        help="Directory for logs and plots (default: logs/sdpo_runs)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--feedback_errors_only", action="store_true",
                        help="Feedback mode: only pass past error messages (no failed proof text). Default: pass both errors and failed proofs.")
    args = parser.parse_args()
    
    # Create config
    config = SDPOConfig(
        model_name=args.model,
        kimina_server_url=args.kimina_url,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        log_dir=args.log_dir,
        feedback_include_failed_proof=not args.feedback_errors_only,
    )
    
    # Initialize metrics tracker
    metrics = MetricsTracker(config.log_dir)
    
    # Log configuration
    metrics.logger.info("="*60)
    metrics.logger.info("SDPO (Self-Distilled Policy Optimization) for Lean")
    metrics.logger.info("="*60)
    metrics.logger.info(f"Model: {config.model_name}")
    metrics.logger.info(f"Max iterations per problem: {config.max_iterations}")
    metrics.logger.info(f"Learning rate: {config.learning_rate}")
    metrics.logger.info(f"Temperature: {config.temperature}")
    metrics.logger.info(f"Kimina URL: {config.kimina_server_url}")
    metrics.logger.info(f"Feedback mode: {'errors only' if not config.feedback_include_failed_proof else 'errors + failed proofs'}")
    metrics.logger.info(f"Log directory: {metrics.run_dir}")
    metrics.logger.info("="*60)
    
    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).parent / dataset_path
    
    metrics.logger.info(f"Loading dataset from {dataset_path}")
    problems = load_minif2f_dataset(str(dataset_path), args.n_problems)
    metrics.logger.info(f"Loaded {len(problems)} problems")
    
    # Initialize SDPO with metrics tracker
    sdpo = LeanSDPO(config, metrics_tracker=metrics)
    
    # Solve problems
    results = []
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        metrics.logger.info(f"\n[{i+1}/{len(problems)}] Starting problem...")
        result = sdpo.solve_theorem(problem)
        results.append(result)
    
    elapsed = time.time() - start_time
    
    # Print final summary
    metrics.print_final_summary()
    metrics.logger.info(f"Time elapsed: {elapsed:.1f}s")
    
    # Save metrics and generate plots
    metrics.save_metrics()
    metrics.plot_metrics()
    
    # Save detailed results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare serializable results
    serializable_results = []
    for r in results:
        sr = {
            "problem_id": r["problem_id"],
            "success": r["success"],
            "best_proof": r["best_proof"],
            "iterations_used": r["iterations_used"],
        }
        serializable_results.append(sr)
    
    n_solved = sum(1 for r in results if r["success"])
    total_iterations = sum(r["iterations_used"] for r in results)
    
    output_data = {
        "config": {
            "model": config.model_name,
            "max_iterations": config.max_iterations,
            "learning_rate": config.learning_rate,
            "temperature": config.temperature,
        },
        "summary": {
            "n_problems": len(problems),
            "n_solved": n_solved,
            "solve_rate": n_solved / len(problems) if problems else 0,
            "total_iterations": total_iterations,
            "elapsed_seconds": elapsed,
        },
        "results": serializable_results,
        "log_dir": str(metrics.run_dir),
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    metrics.logger.info(f"\nResults saved to {output_path}")
    metrics.logger.info(f"Logs and plots saved to {metrics.run_dir}")


if __name__ == "__main__":
    main()
