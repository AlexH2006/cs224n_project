"""
TLDR: Extract a complete Lean 4 code block from model raw output, then assemble
the final file string to submit to the verifier.

Full-block strategy (vs. tactic-level):
  The model is prompted to output a complete, self-contained lean4 block
  (imports + set_option + full theorem with proof). So we extract that block
  directly — no tactic stripping, no sorry-counting, no header surgery.

Two public functions:
  extract_full_lean_block(raw_output) -> str
      Returns the content of the last lean4/lean fenced block after </think>,
      or "sorry" on failure (incomplete reasoning or no block found).

  create_full_lean_code(theorem_code, extracted_block, default_header) -> str
      If the block contains `import` lines, return it as-is (model was self-contained).
      Otherwise prepend default_header (model omitted imports).
      If extracted_block is "sorry", fall back to default_header + theorem_code.

Used by: modal_app.py (post-generation, before verification).
"""

import re

# Matches ```lean4, ```lean, or ``` (no lang tag) then content until ```.
# We include the no-lang-tag variant as a graceful fallback in case the model
# omits the language identifier despite being instructed to use lean4.
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:lean4?|lean)?\s*\n(.*?)```",
    re.DOTALL,
)


def _has_incomplete_reasoning(text: str) -> bool:
    """True iff <think> is open but never closed — model was cut off mid-reasoning.

    NOTE: Only ~80% of outputs use the <think> wrapper at all; the rest reason
    inline without it. This is therefore a supplementary signal, not the primary one.
    Use is_token_limit_hit() as the primary truncation detector.
    """
    return "<think>" in text and "</think>" not in text


def is_token_limit_hit(finish_reason: str) -> bool:
    """True iff vLLM stopped because it hit max_tokens (finish_reason="length").

    This is the definitive truncation signal — set directly by vLLM when generation
    is forcibly stopped, regardless of output format or think-tag presence.

    finish_reason="stop" means the model produced an EOS token and finished naturally.
    finish_reason="length" means max_tokens was reached and the output was cut off.
    """
    return finish_reason == "length"


def _answer_region(text: str) -> str:
    """
    Return the region of text to search for code blocks.
    If </think> is present, use only the text after the last </think> (the answer).
    Otherwise search the whole output (model produced no think wrapper).
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


class ParseResult:
    """
    Structured result of extract_full_lean_block.

    Attributes:
        block:       The extracted Lean 4 code string, or "sorry" on failure.
        truncated:   True iff the model's reasoning was cut off (token limit hit).
                     Detected by an unclosed <think> tag.
        no_block:    True iff reasoning completed but no lean4 fenced block was found.
    """
    __slots__ = ("block", "truncated", "no_block")

    def __init__(self, block: str, truncated: bool = False, no_block: bool = False):
        self.block = block
        self.truncated = truncated
        self.no_block = no_block

    @property
    def failed(self) -> bool:
        """True iff parsing failed for any reason (block == "sorry")."""
        return self.block == "sorry"


def extract_full_lean_block(raw_output: str) -> str:
    """
    Extract the last lean4 code block from model output.

    Returns:
      - "sorry"  if output is empty, reasoning is incomplete, or no lean4 block found.
      - The content of the last lean4/lean fenced block otherwise.

    For structured access to the failure reason (truncated vs. no block found),
    use extract_full_lean_block_parsed() instead.
    """
    return extract_full_lean_block_parsed(raw_output).block


def extract_full_lean_block_parsed(
    raw_output: str,
    finish_reason: str = "",
) -> ParseResult:
    """
    Extract the last lean4 code block from model output, with structured failure info.

    Truncation detection uses two complementary signals (both set truncated=True):
      1. PRIMARY:   finish_reason="length" — set directly by vLLM when max_tokens is
                    hit and generation is forcibly stopped. Pass this from the vLLM
                    completion object. Reliable for all output formats.
      2. SECONDARY: unclosed <think> tag — fallback for post-hoc log analysis when
                    finish_reason is unavailable. Misses the ~16% of outputs that
                    reason inline without a <think> wrapper.

    Returns a ParseResult with:
      - block="sorry", truncated=True   if finish_reason="length" or <think> unclosed
      - block="sorry", no_block=True    if reasoning completed but no lean4 block found
      - block=<code>                    on success
    """
    if not raw_output or not raw_output.strip():
        return ParseResult("sorry", no_block=True)

    text = raw_output.strip()

    # Primary check: vLLM finish_reason (authoritative, format-agnostic).
    if is_token_limit_hit(finish_reason):
        return ParseResult("sorry", truncated=True)

    # Secondary check: unclosed think tag (post-hoc fallback only).
    if _has_incomplete_reasoning(text):
        return ParseResult("sorry", truncated=True)

    region = _answer_region(text)
    matches = _CODE_BLOCK_PATTERN.findall(region)
    if not matches:
        return ParseResult("sorry", no_block=True)

    content = matches[-1].strip()
    if not content:
        return ParseResult("sorry", no_block=True)
    return ParseResult(content)


def create_full_lean_code(
    theorem_code: str,
    extracted_block: str,
    default_header: str,
) -> str:
    """
    Assemble the final Lean 4 source string to submit to the verifier.

    Three cases:
      1. extracted_block has `import` lines → return as-is (model was self-contained).
      2. extracted_block has no `import` but is non-trivial → prepend default_header.
      3. extracted_block is "sorry" (parse failed) → fallback: default_header + theorem_code.

    Args:
        theorem_code:     The raw theorem from the dataset (may contain `sorry`).
        extracted_block:  Result of extract_full_lean_block(); may be "sorry".
        default_header:   Fallback imports+set_option string from EvalConfig.

    Returns:
        A complete Lean 4 source string ready for verification.
    """
    if extracted_block == "sorry":
        # Parse failed entirely — submit the original theorem with the default header.
        # This will almost certainly fail verification, but gives the verifier something
        # concrete to report on rather than an empty submission.
        return f"{default_header}\n\n{theorem_code}"

    if "import" in extracted_block:
        # Model produced a self-contained file — trust it completely.
        return extracted_block

    # Model produced a block but omitted import lines — prepend the default header.
    return f"{default_header}\n\n{extracted_block}"
