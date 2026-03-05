"""
Student and teacher prompt construction for SDPO.

TLDR: create_base_prompt (problem only, for student) and create_feedback_prompt
(problem + single most recent compiler feedback, for teacher). Use
tokenizer.apply_chat_template when available. Used by: trainer_core, entrypoint.
"""

from typing import Callable

from sdpo_modal_local_verify_goedel.config import SDPOConfig


def create_base_prompt(
    config: SDPOConfig,
    problem: dict,
    get_field: Callable[[dict, list, str], str],
    tokenizer,
) -> str:
    """Create base prompt WITHOUT feedback (for STUDENT). Includes informal + formal statement."""
    lean4_code = get_field(problem, config.theorem_fields)
    informal = get_field(problem, config.informal_fields)

    user_content = "Think about and solve the following problem step by step in Lean 4."
    if informal:
        user_content += f"\n# Problem:{informal}"
    user_content += f"\n# Formal statement:\n```lean4\n{lean4_code}\n```\n"

    system_content = "You are an expert in mathematics and proving theorems in Lean 4."

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"System: {system_content}\n\nUser: {user_content}\n\nAssistant:"


def create_feedback_prompt(
    config: SDPOConfig,
    problem: dict,
    latest_feedback: tuple[str, str] | None,
    get_field: Callable[[dict, list, str], str],
    tokenizer,
) -> str:
    """Create prompt WITH the single most recent compiler feedback (for TEACHER)."""
    lean4_code = get_field(problem, config.theorem_fields)
    informal = get_field(problem, config.informal_fields)
    header = get_field(problem, config.header_fields)
    has_header = bool(header.strip())

    user_content = "Prove the following Lean 4 theorem.\n\n"
    if informal:
        user_content += f"Problem: {informal}\n\n"
    user_content += f"```lean4\n{lean4_code}\n```\n\n"

    if latest_feedback is not None:
        feedback_text, failed_proof = latest_feedback
        user_content += "Avoid the following error:\n"
        if config.feedback_include_failed_proof:
            user_content += config.feedback_attempt_template.format(
                feedback=feedback_text, failed_proof=failed_proof
            )
        else:
            user_content += config.feedback_attempt_template_errors_only.format(
                feedback=feedback_text
            )

    if has_header:
        user_content += "\n\nProvide corrected proof tactics:"
    else:
        user_content += "\n\nProvide corrected proof tactics (include any necessary imports):"

    feedback_system_prompt = (
        config.system_prompt
        + " After reasoning, output the proof tactics in a ```lean4 code block."
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": feedback_system_prompt},
            {"role": "user", "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"System: {feedback_system_prompt}\n\nUser: {user_content}\n\nAssistant:"
