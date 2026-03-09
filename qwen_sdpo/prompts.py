"""
TLDR: Build chat-formatted prompts for the SDPO student and teacher.

Prompts are built from templates; content is identical to qwen_eval's build_prompt()
for the student (zero-shot CoT) and a distinct teacher template with feedback. Both
share the same system prompt. Teacher prompt optionally disables Qwen3.5 thinking
mode when teacher_response_mode is answer_only/code_only and not truncated.

Used by: entrypoint.py (builds both prompts locally before sending to Modal).
"""

from qwen_sdpo.config import SDPOConfig


# Shared system prompt (student and teacher).
_SYSTEM_PROMPT = "You are an expert in mathematics and Lean 4 theorem proving."

# Student user template — zero-shot CoT, identical to qwen_eval's build_prompt() content.
_STUDENT_USER_TEMPLATE = """\
Reason step-by-step to solve the following Lean 4 problem.

# Informal problem:
{informal}

# Lean 4 theorem to prove:
```lean4
{header_and_theorem}
```

Instructions:
- Do NOT use `sorry`. 
- At the very end of your response, output your final answer as exactly one lean4 code block that is complete and self-contained: all imports, set_option lines, and the full theorem with proof
-Do NOT output any text after the closing ```\
"""

# Teacher user template — problem + raw feedback + closing instruction.
_TEACHER_USER_TEMPLATE = """\
Solve the following Lean 4 problem

# Informal problem:
{informal}

# Lean 4 theorem to prove:
```lean4
{header_and_theorem}
```

You MUST avoid the following errors of earlier attempts:
{feedback}

Correctly solve the original Lean 4 problem.
"""


def _header_and_theorem(theorem_code: str, header: str, default_header: str) -> str:
    """Build header_and_theorem string. Uses default_header if header is empty."""
    effective = header.strip() if header and header.strip() else default_header
    return f"{effective}\n{theorem_code}"


def _safe_informal(informal: str) -> str:
    """Normalize informal; fallback if empty."""
    return informal.strip() if informal else "(no informal description provided)"


def _build_chat_prompt(
    user_content: str,
    tokenizer,
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Format messages and apply chat template. Shared by student and teacher.

    Args:
        user_content: User message content.
        tokenizer: HuggingFace tokenizer with apply_chat_template.
        enable_thinking: When None, use tokenizer default (thinking on for Qwen3.5).
                         When False, disable Qwen3.5 thinking mode so model outputs answer directly.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def build_student_prompt(
    theorem_code: str,
    informal: str,
    header: str,
    tokenizer,
    cfg: SDPOConfig,
) -> str:
    """Build the student prompt — zero-shot CoT, identical to qwen_eval content.

    The student sees only the problem (no error feedback). Uses _STUDENT_USER_TEMPLATE
    so prompt format stays in sync with baseline eval.

    Args:
        theorem_code: Formal Lean 4 theorem (with `sorry` placeholder).
        informal:     Natural language problem description.
        header:       Dataset-provided header (imports + set_option). May be empty;
                      falls back to cfg.default_header.
        tokenizer:    HuggingFace tokenizer with apply_chat_template.
        cfg:          SDPOConfig (passes default_header; other fields unused here).

    Returns:
        A fully chat-formatted string ready for vLLM.
    """
    header_and_theorem = _header_and_theorem(theorem_code, header, cfg.default_header)
    user_content = _STUDENT_USER_TEMPLATE.format(
        informal=_safe_informal(informal),
        header_and_theorem=header_and_theorem,
    )
    return _build_chat_prompt(user_content, tokenizer)


def build_teacher_prompt(
    theorem_code: str,
    informal: str,
    header: str,
    feedback: str,
    tokenizer,
    cfg: SDPOConfig,
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Build the teacher prompt — problem + raw feedback + closing instruction.

    Uses _TEACHER_USER_TEMPLATE: same system prompt as student, but a distinct
    user message that includes raw feedback (compiler error, truncation, or
    parse failure) and a closing instruction.

    Args:
        theorem_code:   Formal Lean 4 theorem (with `sorry` placeholder).
        informal:       Natural language problem description.
        header:         Dataset-provided header. May be empty; falls back to cfg.default_header.
        feedback:       Raw feedback string (compiler error, truncation message, or parse failure).
        tokenizer:      HuggingFace tokenizer with apply_chat_template.
        cfg:            SDPOConfig (passes default_header).
        enable_thinking: When None, use tokenizer default. Caller should pass False for
                        answer_only/code_only when not truncated; True when truncated or full_output.

    Returns:
        A fully chat-formatted string ready for vLLM.
    """
    header_and_theorem = _header_and_theorem(theorem_code, header, cfg.default_header)
    user_content = _TEACHER_USER_TEMPLATE.format(
        informal=_safe_informal(informal),
        header_and_theorem=header_and_theorem,
        feedback=feedback.strip() if feedback else "Proof verification failed.",
    )
    return _build_chat_prompt(user_content, tokenizer, enable_thinking=enable_thinking)
