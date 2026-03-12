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
Prove the following Lean 4 theorem without using `sorry`. At the very end of your response, output your complete, final solution as exactly one ```lean4``` code block.

# Informal problem:
{informal}

# Lean 4 theorem to prove:
```lean4
{header_and_theorem}
```
"""


def _initial_messages(theorem_code: str, informal: str, header: str, cfg: EvalConfig) -> list[dict]:
    """Build the initial conversation messages (system + user) for one problem. Used by build_prompt and get_initial_messages."""
    effective_header = header.strip() if header and header.strip() else cfg.default_header
    header_and_theorem = f"{effective_header}\n{theorem_code}"
    user_content = _USER_TEMPLATE.format(
        informal=informal.strip() if informal else "(no informal description provided)",
        header_and_theorem=header_and_theorem,
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def messages_to_prompt(messages: list[dict], tokenizer, cfg: EvalConfig) -> str:
    """
    Convert a conversation messages list to a single prompt string for vLLM.
    Used for correction rounds when messages have been extended with assistant + user.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=cfg.use_think_mode,
    )


def get_initial_messages(problem: dict, cfg: EvalConfig) -> list[dict]:
    """Return the initial messages list for a problem (for correction-flow state init)."""
    return _initial_messages(
        problem["formal_statement"],
        problem.get("informal_stmt", ""),
        problem.get("header", ""),
        cfg,
    )


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
        cfg:          EvalConfig (use_think_mode passed to tokenizer as enable_thinking).

    Returns:
        A fully formatted string ready to pass to vLLM as a prompt.
    """
    messages = _initial_messages(theorem_code, informal, header, cfg)
    return messages_to_prompt(messages, tokenizer, cfg)
