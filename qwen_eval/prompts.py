"""
TLDR: Build the chat-formatted prompt for Qwen3.5-4B proof generation.

Strategy:
  - Zero-shot CoT: "Think step-by-step" triggers Qwen3.5's <think>...</think> reasoning.
  - Full-block output: model is instructed to emit a complete, self-contained lean4
    code block (imports + set_option + full theorem with proof) as the very last thing
    in its response, with nothing after the closing fence.
  - This "last block" placement makes parsing unambiguous: always take the last lean4 block.

Used by: modal_app.py (ProofGenerator.setup builds prompts before batching).
"""

from qwen_eval.config import EvalConfig


_SYSTEM_PROMPT = "You are an expert in mathematics and Lean 4 theorem proving."

_USER_TEMPLATE = """\
Reason step-by-step to prove the following Lean 4 theorem.

# Informal problem:
{informal}

# Lean 4 theorem to prove:
```lean4
{header_and_theorem}
```

Instructions:
- Do NOT use `sorry`
- At the very end of your response, output your final answer as exactly one lean4 code block that is complete and self-contained: all imports, set_option lines, and the full theorem with proof
- Do not output any text after the closing ```\
"""


def build_prompt(
    theorem_code: str,
    informal: str,
    header: str,
    tokenizer,
    cfg: EvalConfig,
) -> str:
    """
    Build a chat-formatted prompt string for one problem.

    Args:
        theorem_code: The formal Lean 4 theorem (with `sorry` placeholder).
        informal:     Natural language description of the problem.
        header:       Dataset-provided header (imports + set_option). May be empty;
                      falls back to cfg.default_header so the model sees correct imports.
        tokenizer:    HuggingFace tokenizer with apply_chat_template.
        cfg:          EvalConfig (unused currently but kept for future prompt variants).

    Returns:
        A fully formatted string ready to pass to vLLM as a prompt.
    """
    # Use dataset header if available, otherwise fall back to default so the model
    # sees the correct import context when deciding what to include in its output.
    effective_header = header.strip() if header and header.strip() else cfg.default_header
    header_and_theorem = f"{effective_header}\n{theorem_code}"

    user_content = _USER_TEMPLATE.format(
        informal=informal.strip() if informal else "(no informal description provided)",
        header_and_theorem=header_and_theorem,
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
