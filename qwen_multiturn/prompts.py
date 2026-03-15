"""
TLDR: Build chat-formatted prompts for Qwen proof generation (round 0 and correction).

- use_think_mode (EvalConfig) is passed to the tokenizer as enable_thinking. Default is
  False (reasoning off). When False, the template signals the model not to emit <think>.
- Full-block output: model is instructed to emit one final ```lean4``` code block.
- Parsing takes the last lean4 block from the response.

Used by: modal_app.ProofGenerator (generate_all, generate_correction_round).
"""

from qwen_multiturn.config import EvalConfig


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


def get_initial_messages(
    theorem_code: str,
    informal: str,
    header: str,
    cfg: EvalConfig,
) -> list[dict]:
    """
    Build the initial conversation messages (system + user) for one problem.
    Used by build_prompt for round 0 and by build_correction_prompt to extend the conversation.
    """
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


_CORRECTION_USER_TEMPLATE = """\
The previous proof did not verify in Lean.

Here is the Lean/compiler feedback from your previous attempt:
<error>
{feedback}
</error>

Please revise your previous solution to fix these errors.

Requirements:
- Keep working on the same theorem.
- Use the verifier feedback above.
- At the very end of your response, output your complete corrected solution as exactly one ```lean4``` code block.
- Do not output multiple final Lean code blocks.
- Do not leave any `sorry` in the final code.
"""


def build_correction_prompt(
    problem: dict,
    previous_assistant_output: str,
    feedback: str,
    tokenizer,
    cfg: EvalConfig,
) -> str:
    """
    Build a chat-formatted prompt for a correction round: initial messages +
    previous assistant response + new USER message with only the latest verifier feedback.
    cfg.use_think_mode is passed to the tokenizer as enable_thinking.
    """
    messages = get_initial_messages(
        theorem_code=problem["formal_statement"],
        informal=problem["informal_stmt"],
        header=problem["header"],
        cfg=cfg,
    )
    correction_content = _CORRECTION_USER_TEMPLATE.format(
        feedback=feedback if feedback else "(no feedback provided)"
    )
    messages.append({"role": "assistant", "content": previous_assistant_output})
    messages.append({"role": "user", "content": correction_content})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=cfg.use_think_mode,
    )


def build_prompt(
    theorem_code: str,
    informal: str,
    header: str,
    tokenizer,
    cfg: EvalConfig,
) -> str:
    """
    Build a chat-formatted prompt string for one problem (round 0).

    Args:
        theorem_code: The formal Lean 4 theorem (with `sorry` placeholder).
        informal:     Natural language description of the problem.
        header:       Dataset-provided header (imports + set_option). May be empty;
                      falls back to cfg.default_header so the model sees correct imports.
        tokenizer:    HuggingFace tokenizer with apply_chat_template.
        cfg:          EvalConfig; cfg.use_think_mode is passed to tokenizer as enable_thinking.

    Returns:
        A fully formatted string ready to pass to vLLM as a prompt.
    """
    messages = get_initial_messages(theorem_code, informal, header, cfg)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=cfg.use_think_mode,
    )
