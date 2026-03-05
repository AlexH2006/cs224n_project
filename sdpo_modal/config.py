"""
SDPO and dataset field configuration.

TLDR: Single source of truth for SDPOConfig and default field name lists used
by dataset loading and prompt building. Used by: utils, prompts, sdpo_loss,
trainer_core, entrypoint.
"""

from dataclasses import dataclass, field
from typing import Optional


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
        "informal_prefix", "informal_stmt", "problem", "informal_statement",
        "natural_language", "description", "question", "informal"
    ])
    header_fields: list = field(default_factory=lambda: [
        "header", "imports", "preamble", "prefix"
    ])
    id_fields: list = field(default_factory=lambda: [
        "problem_id", "name", "id", "idx", "index"
    ])

    # Generation settings
    max_new_tokens: int = 8096
    temperature: float = 0.6
    top_p: float = 0.95
    stop_tokens: list = field(default_factory=lambda: [
        "<|im_end|>", "<|endoftext|>", "</s>", "<|end|>",
        "[/INST]", "<|eot_id|>"
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
    feedback_errors_only: bool = True
    feedback_include_failed_proof: bool = False
    feedback_attempt_template: str = """- Error: {feedback}
  Failed proof: {failed_proof}"""
    feedback_attempt_template_errors_only: str = """- {feedback}"""
    feedback_separator: str = "\n"

    # Output / infra
    output_dir: str = "kimina_2b"
    gpu: str = "A100-40GB"

    # Optional overrides for dataset field names (prepended to theorem_fields, etc.)
    theorem_field_override: Optional[str] = None
    informal_field_override: Optional[str] = None
    header_field_override: Optional[str] = None


# Default field name lists for dataset-agnostic loading (used by entrypoint and utils)
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
