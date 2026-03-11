"""
TLDR: Single source of truth for all SDPO hyperparameters.

SDPOConfig is the only dataclass; every module receives it rather than defining
its own constants. CLI flags in modal_app.py override defaults.

Notable differences from qwen_eval.EvalConfig:
  - Adds SDPO training params (max_iterations, learning_rate, distillation_topk)
  - Adds LoRA params (use_lora, lora_r, lora_alpha)
  - results_base_dir defaults to "sdpo_results"; model tag is appended at runtime
    → sdpo_results/Qwen3.5-9B/run_Qwen3.5-9B_{problem_idx}_{timestamp}/
  - problem_idx (single problem) rather than n_problems / pass_k (batch eval)
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

# Allowed values for teacher_response_mode (run dir uses these; payload uses "full" for full generation).
TeacherResponseMode = Literal["full_output", "answer_only", "code_only"]


@dataclass
class SDPOConfig:
    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model_name: str = "Qwen/Qwen3.5-4B"    # or "Qwen/Qwen3.5-9B"

    # When True, Qwen3.5 may emit <think>...</think> reasoning (tokenizer enable_thinking).
    # When False, student and teacher use enable_thinking=False for non-thinking mode.
    # Mirrors qwen_eval.EvalConfig.use_think_mode.
    use_think_mode: bool = True

    # LoRA: always enabled (QLoRA via bitsandbytes + peft).
    # target_modules covers both Qwen3_5GatedDeltaNet (24 linear_attention layers)
    # and Qwen3_5Attention (8 full_attention layers) and Qwen3_5MLP (all 32 layers).
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    dataset_name: str = "cat-searcher/minif2f-lean4"
    dataset_split: str = "test"
    problem_idx: int = 0                    # single problem index in the dataset

    # Field name priority lists (tried in order; first non-empty wins).
    # Covers common variants across HuggingFace dataset versions.
    theorem_fields: list = field(default_factory=lambda: [
        "lean4_code", "formal_statement", "lean4_statement",
        "statement", "code", "theorem", "problem_statement",
    ])
    informal_fields: list = field(default_factory=lambda: [
        "informal_prefix", "informal_stmt", "problem", "informal_statement",
        "natural_language", "description", "question", "informal",
    ])
    header_fields: list = field(default_factory=lambda: [
        "header", "imports", "preamble", "prefix",
    ])
    id_fields: list = field(default_factory=lambda: [
        "problem_id", "name", "id", "idx", "index",
    ])

    # -------------------------------------------------------------------------
    # Generation (same sampling params as qwen_eval for consistency)
    # -------------------------------------------------------------------------
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    # -------------------------------------------------------------------------
    # SDPO training
    # -------------------------------------------------------------------------
    max_iterations: int = 5
    learning_rate: float = 1e-5
    # Number of top-K token logits used in KL divergence computation.
    # A tail bucket captures remaining probability mass so KL covers full distribution.
    distillation_topk: int = 20
    # If True, teacher prompt receives only the Lean compiler error string.
    # If False, it also receives the full failed proof attempt.
    feedback_errors_only: bool = True
    # Which part of the generation the teacher (KL) uses: full_output (entire generation),
    # answer_only (tokens after </think>), or code_only (parsed lean4 block only).
    # Run directory is sdpo_results/{model_tag}/{teacher_response_mode}/...
    teacher_response_mode: TeacherResponseMode = "full_output"

    # -------------------------------------------------------------------------
    # Lean header fallback
    # Used only when the model's extracted block contains no `import` lines.
    # -------------------------------------------------------------------------
    default_header: str = (
        "import Mathlib\n"
        "set_option maxHeartbeats 400000\n"
        "open BigOperators Real Nat Topology Rat"
    )

    # -------------------------------------------------------------------------
    # Infrastructure
    # -------------------------------------------------------------------------
    # Set "A100-40GB" for 4B (fits comfortably, ~25GB used of 40GB).
    # Set "H100" for 9B (requires ~42GB, must have 80GB).
    gpu: str = "A100-40GB"   # A100-40GB for 4B (default); use H100 for 9B

    kimina_url: str = "http://localhost:8000"
    verify_timeout_s: int = 60
    verify_retries: int = 3
    verify_retry_wait_s: float = 3.0

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    # Model tag is appended at runtime:
    #   sdpo_results/Qwen3.5-9B/run_Qwen3.5-9B_{problem_idx}_{timestamp}/
    results_base_dir: str = "sdpo_results"
    # When set (by run_sdpo_batch), per-problem output goes under batch_run_dir/runs/problem_{idx}/
    # and manifest under batch_run_dir/manifest/. Single-problem runs ignore this.
    batch_run_dir: Optional[str] = None
