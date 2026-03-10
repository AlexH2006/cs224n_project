"""
Student and teacher prompt construction for SDPO.

TLDR: create_base_prompt (problem only, for student) and create_feedback_prompt
(problem + compiler feedback for teacher). Teacher can receive a single latest
feedback or full history of previous attempts (code + errors only). Used by:
trainer_core, entrypoint.
"""

from typing import Callable

from sdpo_modal_local_verify_qwen.config import SDPOConfig


STUDENT_SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover.\n"
    "First reason about the proof strategy inside <think>...</think>.\n"
    "Keep the <think> section brief and focused on the proof strategy.\n"
    "Then output exactly one ```lean4 code block containing ONLY the Lean proof "
    "that replaces `sorry`. Do not include any text after the code block.\n"
    "Do not restate the theorem.\n\n"

    "Prove the theorem using Lean tactics. Carefully consider what tactic "
    "decomposition would make the proof easiest to verify in Lean.\n"
    "Break the argument into small steps and introduce helper lemmas when useful.\n"

    "Prefer explicit tactic proofs such as:\n"
    "intro, cases, induction, have, calc, simp, rw, exact, refine.\n\n"

    "Derive intermediate results from the given hypotheses rather than "
    "guessing closed forms. Ensure each step typechecks in Lean.\n"
    "Prefer Lean-checked simplification and rewriting over manual arithmetic."
)

TEACHER_SYSTEM_PROMPT = (
    "You are an expert Lean 4 theorem prover debugging a failed proof.\n"
    "First reason about how to repair the proof inside <think>...</think>.\n"
    "Keep the <think> section brief and focused on the repair strategy.\n"
    "Then output exactly one ```lean4 code block containing ONLY the corrected "
    "Lean proof that replaces `sorry`. Do not include any text after the code block.\n"
    "Do not restate the theorem.\n\n"

    "Your task is to FIX the proof using the compiler feedback.\n"
    "Identify the step that caused the error and repair it.\n"
    "Prefer minimal local fixes when possible.\n"
    "If the proof strategy is fundamentally wrong, restructure it.\n\n"

    "Break the repair into small Lean-verified steps and introduce helper lemmas when useful.\n"
    "Use explicit Lean tactics such as intro, cases, induction, have, calc, simp, rw, exact, refine.\n"
    "Derive intermediate results from the hypotheses rather than guessing closed forms.\n"
    "Prefer Lean-checked simplification and rewriting over manual arithmetic.\n"
    "Ensure every step typechecks.\n"
    "Do not output `sorry`."
)


def create_base_prompt(
    config: SDPOConfig,
    problem: dict,
    get_field: Callable[[dict, list, str], str],
    tokenizer,
) -> str:
    lean4_code = get_field(problem, config.theorem_fields)
    informal = get_field(problem, config.informal_fields)

    system_content = STUDENT_SYSTEM_PROMPT

    user_content = ""
    if informal:
        user_content += f"# Problem: {informal}\n\n"
    user_content += f"# Formal statement:\n```lean4\n{lean4_code}\n```"

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
    latest_feedback: tuple[str, str] | None = None,
    feedback_history: list[tuple[str, str]] | None = None,
    get_field: Callable[[dict, list, str], str] | None = None,
    tokenizer=None,
) -> str:
    if get_field is None or tokenizer is None:
        raise TypeError("get_field and tokenizer are required")

    lean4_code = get_field(problem, config.theorem_fields)
    informal = get_field(problem, config.informal_fields)
    header = get_field(problem, config.header_fields)
    has_header = bool(header.strip())

    system_content = TEACHER_SYSTEM_PROMPT

    user_content = ""
    if informal:
        user_content += f"# Problem: {informal}\n\n"
    user_content += f"# Formal statement:\n```lean4\n{lean4_code}\n```\n\n"

    history = feedback_history if feedback_history is not None else []
    if latest_feedback is not None and not history:
        history = [latest_feedback]

    if history:
        user_content += "Compiler feedback from previous attempts:\n"
        for i, (feedback_text, failed_proof) in enumerate(history, 1):
            if len(history) > 1:
                user_content += f"\nAttempt {i}:\n"
            if config.feedback_include_failed_proof:
                user_content += config.feedback_attempt_template.format(
                    feedback=feedback_text, failed_proof=failed_proof
                )
            else:
                user_content += config.feedback_attempt_template_errors_only.format(
                    feedback=feedback_text
                )
            if len(history) > 1:
                user_content += "\n"

    if has_header:
        user_content += "\n\nReturn the corrected proof."
    else:
        user_content += "\n\nReturn the corrected proof. Include any necessary imports."

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