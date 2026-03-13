"""
Student and teacher prompt construction for SDPO.

TLDR: Prompts match qwen_eval (system + user template). create_base_prompt builds
the student prompt; create_feedback_prompt builds the teacher prompt as the same
base content plus only Lean compiler errors (no previous attempt code). Used by:
entrypoint (imo_p6_experiment).
"""

from typing import Callable

from imo_p6_experiment.config import SDPOConfig


# Same as qwen_eval/prompts.py — single system for both student and teacher
_SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover. "
    "Output only valid Lean 4 code. "
    "Do not include explanations, reasoning, or commentary. "
    "Do not use `sorry`."
)

_USER_TEMPLATE = """\
Write a complete Lean 4 file that proves the theorem below.

# Informal problem:
{informal}

# Lean 4 theorem to prove:
```lean4
{header_and_theorem}
```
"""


def create_base_prompt(
    config: SDPOConfig,
    problem: dict,
    get_field: Callable[[dict, list, str], str],
    tokenizer,
) -> str:
    """Build student prompt: same format as qwen_eval (system + user template)."""
    lean4_code = get_field(problem, config.theorem_fields)
    informal = get_field(problem, config.informal_fields)
    header = get_field(problem, config.header_fields)

    effective_header = (
        header.strip() if header and header.strip() else config.default_header
    )
    header_and_theorem = f"{effective_header}\n{lean4_code}"

    user_content = _USER_TEMPLATE.format(
        informal=informal.strip() if informal else "(no informal description provided)",
        header_and_theorem=header_and_theorem,
    )

    system_prompt = (config.system_prompt or _SYSTEM_PROMPT).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            pass

    return f"System: {system_prompt}\n\nUser: {user_content}\n\nAssistant:"


def create_feedback_prompt(
    config: SDPOConfig,
    problem: dict,
    latest_feedback: tuple[str, str] | None = None,
    feedback_history: list[tuple[str, str]] | None = None,
    get_field: Callable[[dict, list, str], str] | None = None,
    tokenizer=None,
) -> str:
    """Build teacher prompt: same base as student + Lean compiler feedback only."""
    if get_field is None or tokenizer is None:
        raise TypeError("get_field and tokenizer are required")

    lean4_code = get_field(problem, config.theorem_fields)
    informal = get_field(problem, config.informal_fields)
    header = get_field(problem, config.header_fields)

    effective_header = (
        header.strip() if header and header.strip() else config.default_header
    )
    header_and_theorem = f"{effective_header}\n{lean4_code}"

    user_content = _USER_TEMPLATE.format(
        informal=informal.strip() if informal else "(no informal description provided)",
        header_and_theorem=header_and_theorem,
    )

    history = feedback_history if feedback_history is not None else []
    if latest_feedback is not None and not history:
        history = [latest_feedback]

    if history:
        user_content += "\n\nCompiler feedback from previous attempt:\n"
        for feedback_text, _ in history:
            user_content += config.feedback_attempt_template_errors_only.format(
                feedback=feedback_text
            )
            user_content += "\n"

    system_prompt = (config.system_prompt or _SYSTEM_PROMPT).strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            pass

    return f"System: {system_prompt}\n\nUser: {user_content}\n\nAssistant:"