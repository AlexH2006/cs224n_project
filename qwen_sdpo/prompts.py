"""
TLDR: Build chat-formatted prompts for the SDPO student and teacher.

Four templates: student_prompt_thinking, student_prompt_no_thinking, teacher_prompt_thinking,
teacher_prompt_no_thinking. All share the same system prompt. build_student_prompt(cfg) selects
student template by cfg.use_think_mode; build_teacher_prompt uses teacher thinking;
build_teacher_prompt_with_full_generation uses teacher no-thinking (includes full_generation).

Used by: entrypoint.py (builds both prompts locally before sending to Modal).
"""

from qwen_sdpo.config import SDPOConfig


# Shared system prompt (student and teacher).
_SYSTEM_PROMPT = "You are an expert in mathematics and Lean 4 theorem proving."

# ---------------------------------------------------------------------------
# Four prompt templates: student/teacher x thinking/no_thinking
# Format keys: {informal}, {header_and_theorem}; teacher also {feedback}; no-thinking teacher also {full_generation}.
# ---------------------------------------------------------------------------

# Thinking-mode student: zero-shot CoT, identical to qwen_eval's build_prompt() content.
_STUDENT_PROMPT_THINKING = """\
Reason step-by-step to solve the following Lean 4 theorem without using `sorry`. At the very end of your response, output your complete, final solution as exactly one ```lean4``` code block.

# Informal problem:
{informal}

# Lean 4 theorem to prove:
```lean4
{header_and_theorem}
```
"""

# Non-thinking student: same content as thinking for now; can be edited independently (e.g. remove "Reason step-by-step").
_STUDENT_PROMPT_NO_THINKING = """\
Prove the following Lean 4 theorem without using `sorry`. At the very end of your response, output your complete, final solution as exactly one ```lean4``` code block.

# Informal problem:
{informal}

# Lean 4 theorem to prove:
```lean4
{header_and_theorem}
```
"""

# Thinking-mode teacher: problem + raw feedback + closing instruction (no generation in prompt).
_TEACHER_PROMPT_THINKING = """\
Reason step-by-step to solve the following Lean 4 theorem without using `sorry`. At the very end of your response, output your complete, final solution as exactly one ```lean4``` code block.

# Informal problem:
{informal}

# Lean 4 theorem to prove:
```lean4
{header_and_theorem}
```

You MUST avoid the following compiler errors of earlier attempts:
{feedback}
"""

# Non-thinking teacher: problem + full_generation (failed attempt) + feedback + instructions.
_TEACHER_PROMPT_NO_THINKING = """\
Prove the following Lean 4 theorem without using `sorry`. At the very end of your response, output your complete, final solution as exactly one ```lean4``` code block.

# Informal problem:
{informal}

# Lean 4 theorem to prove:
```lean4
{header_and_theorem}
```

You MUST avoid the following compiler errors of earlier attempts:
{feedback}
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
    """Build the student prompt. Template chosen by cfg.use_think_mode.

    Uses _STUDENT_PROMPT_THINKING when use_think_mode=True, _STUDENT_PROMPT_NO_THINKING
    when False. The student sees only the problem (no error feedback).

    Args:
        theorem_code: Formal Lean 4 theorem (with `sorry` placeholder).
        informal:     Natural language problem description.
        header:       Dataset-provided header (imports + set_option). May be empty;
                      falls back to cfg.default_header.
        tokenizer:    HuggingFace tokenizer with apply_chat_template.
        cfg:          SDPOConfig (default_header, use_think_mode).

    Returns:
        A fully chat-formatted string ready for vLLM.
    """
    header_and_theorem = _header_and_theorem(theorem_code, header, cfg.default_header)
    fmt_args = {"informal": _safe_informal(informal), "header_and_theorem": header_and_theorem}
    if cfg.use_think_mode:
        user_content = _STUDENT_PROMPT_THINKING.format(**fmt_args)
        return _build_chat_prompt(user_content, tokenizer, enable_thinking=True)
    user_content = _STUDENT_PROMPT_NO_THINKING.format(**fmt_args)
    return _build_chat_prompt(user_content, tokenizer, enable_thinking=False)


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
    """Build the teacher prompt (thinking mode). Uses _TEACHER_PROMPT_THINKING.

    Problem + raw feedback + closing instruction; no generation in the prompt.
    enable_thinking is set by caller (e.g. False for answer_only/code_only when not truncated).

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
    user_content = _TEACHER_PROMPT_THINKING.format(
        informal=_safe_informal(informal),
        header_and_theorem=header_and_theorem,
        feedback=feedback.strip() if feedback else "Proof verification failed.",
    )
    return _build_chat_prompt(user_content, tokenizer, enable_thinking=enable_thinking)


def build_teacher_prompt_with_full_generation(
    theorem_code: str,
    informal: str,
    header: str,
    feedback: str,
    full_generation: str,
    tokenizer,
    cfg: SDPOConfig,
) -> str:
    """Build the teacher prompt (non-thinking mode). Uses _TEACHER_PROMPT_NO_THINKING.

    User message includes problem, full_generation (the raw failed attempt), feedback, and
    instructions. Always uses enable_thinking=False. Used by entrypoint when cfg.use_think_mode is False.

    Args:
        theorem_code:   Formal Lean 4 theorem (with `sorry` placeholder).
        informal:       Natural language problem description.
        header:         Dataset-provided header. May be empty; falls back to cfg.default_header.
        feedback:       Raw feedback string (compiler error, truncation, or parse failure).
        full_generation: The entire raw model output from the failed attempt (appended to prompt).
        tokenizer:      HuggingFace tokenizer with apply_chat_template.
        cfg:            SDPOConfig (passes default_header).

    Returns:
        A fully chat-formatted string ready for vLLM.
    """
    header_and_theorem = _header_and_theorem(theorem_code, header, cfg.default_header)
    user_content = _TEACHER_PROMPT_NO_THINKING.format(
        informal=_safe_informal(informal),
        header_and_theorem=header_and_theorem,
        feedback=feedback.strip() if feedback else "Proof verification failed.",
        full_generation=full_generation.strip() if full_generation else "(no generation)",
    )
    return _build_chat_prompt(user_content, tokenizer, enable_thinking=False)
