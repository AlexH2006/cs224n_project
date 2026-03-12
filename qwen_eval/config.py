"""
TLDR: Single source of truth for all eval parameters.

EvalConfig is the only dataclass; every other module receives it (or a subset)
rather than defining its own constants. CLI flags in modal_app.py override defaults.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalConfig:
    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model_name: str = "Qwen/Qwen3.5-4B"

    # When True, Qwen3.5 may emit <think>...</think> reasoning (via tokenizer enable_thinking).
    # When False, tokenizer.apply_chat_template(..., enable_thinking=False) disables it.
    use_think_mode: bool = False

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    dataset_name: str = "cat-searcher/minif2f-lean4"
    dataset_split: str = "test"

    # Dataset field name priority lists (tried in order; first non-empty wins).
    # Covers common variants across different HF dataset versions.
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
    # Eval scale
    # -------------------------------------------------------------------------
    n_problems: int = 20
    pass_k: int = 4
    seed: int = 42

    # Multi-turn correction: 0 = one-shot only; N = up to N+1 generations per attempt.
    num_correction_rounds: int = 0

    # -------------------------------------------------------------------------
    # Sampling parameters (passed directly to vLLM SamplingParams)
    # -------------------------------------------------------------------------
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 8192

    # Chunk size for vLLM generate() when processing many problems on one GPU.
    # None or 0 = pass all prompts in one call. Reduces peak memory for large runs.
    inference_batch_size: Optional[int] = 256

    # Save results after each batch of this many problems (generate -> parse -> verify -> save).
    # None or 0 = run all problems in one go (current behavior).
    generation_save_batch_size: Optional[int] = None

    # -------------------------------------------------------------------------
    # Lean header fallback
    # Used only when the model's extracted block contains no `import` lines.
    # Simple `import Mathlib` covers all Mathlib tactics for Lean 4.
    # -------------------------------------------------------------------------
    default_header: str = (
        "import Mathlib\n"
        "set_option maxHeartbeats 400000\n"
        "open BigOperators Real Nat Topology Rat"
    )

    # -------------------------------------------------------------------------
    # Infrastructure
    # -------------------------------------------------------------------------
    gpu: str = "H100"

    # Kimina Lean Server URL. Override with LEAN_VERIFY_KIMINA_URL env var.
    kimina_url: str = "http://localhost:8000"

    # Verification robustness
    verify_timeout_s: int = 60
    verify_retries: int = 3
    verify_retry_wait_s: float = 3.0

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    # Results are written to: {results_base_dir}/run_{model_safe}_{timestamp}/
    results_base_dir: str = "baseline"

    # Optional: run a specific subset by index (None = first n_problems from split)
    problem_indices: Optional[list] = None
