"""
TLDR: Self-contained parsing for SDPO — Lean block extraction + COT/answer split.

Provides extract_full_lean_block_parsed (with optional code-block character span for
code_only teacher mode), create_full_lean_code, split_cot_and_answer,
get_answer_token_slice, and get_code_token_slice. Token alignment includes
verification and fallback to avoid character-to-token mismatches.

Used by: entrypoint (parse + payload), modal_trainer indirectly via payload.
"""

import logging
import re
from typing import Optional

_LOG = logging.getLogger(__name__)

# Matches ```lean4, ```lean, or ``` (no lang tag) then content until ```.
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:lean4?|lean)?\s*\n(.*?)```",
    re.DOTALL,
)

_THINK_CLOSE = "</think>"


def _has_incomplete_reasoning(text: str) -> bool:
    """True iff <think> is open but never closed — model was cut off mid-reasoning."""
    return "<think>" in text and _THINK_CLOSE not in text


def is_token_limit_hit(finish_reason: str) -> bool:
    """True iff vLLM stopped because it hit max_tokens (finish_reason="length")."""
    return finish_reason == "length"


def _answer_region(text: str) -> str:
    """
    Return the region of text to search for code blocks.
    If </think> is present, use only the text after the last </think> (the answer).
    Otherwise search the whole output (model produced no think wrapper).
    """
    if _THINK_CLOSE in text:
        return text.split(_THINK_CLOSE)[-1].strip()
    return text


class ParseResult:
    """
    Structured result of extract_full_lean_block_parsed.

    Attributes:
        block:       The extracted Lean 4 code string, or "sorry" on failure.
        truncated:   True iff the model's reasoning was cut off (token limit hit).
        no_block:    True iff reasoning completed but no lean4 fenced block was found.
        code_block_start_char: Start offset of the code block content in raw_output (None if failed).
        code_block_end_char:   End offset (exclusive) of the code block content in raw_output (None if failed).
    """
    __slots__ = ("block", "truncated", "no_block", "code_block_start_char", "code_block_end_char")

    def __init__(
        self,
        block: str,
        truncated: bool = False,
        no_block: bool = False,
        code_block_start_char: Optional[int] = None,
        code_block_end_char: Optional[int] = None,
    ):
        self.block = block
        self.truncated = truncated
        self.no_block = no_block
        self.code_block_start_char = code_block_start_char
        self.code_block_end_char = code_block_end_char

    @property
    def failed(self) -> bool:
        """True iff parsing failed for any reason (block == "sorry")."""
        return self.block == "sorry"


def extract_full_lean_block(raw_output: str) -> str:
    """Extract the last lean4 code block from model output. Returns "sorry" on failure."""
    return extract_full_lean_block_parsed(raw_output).block


def extract_full_lean_block_parsed(
    raw_output: str,
    finish_reason: str = "",
) -> ParseResult:
    """
    Extract the last lean4 code block from model output, with structured failure info.

    Truncation: finish_reason="length" (primary) or unclosed <think> tag (secondary).
    Returns ParseResult(block, truncated, no_block, code_block_start_char, code_block_end_char).
    The character span is set only when a block was successfully extracted (for code_only teacher mode).
    """
    if not raw_output or not raw_output.strip():
        return ParseResult("sorry", no_block=True)

    text = raw_output.strip()

    if is_token_limit_hit(finish_reason):
        return ParseResult("sorry", truncated=True)

    if _has_incomplete_reasoning(text):
        return ParseResult("sorry", truncated=True)

    region = _answer_region(text)
    matches_list = list(_CODE_BLOCK_PATTERN.finditer(region))
    if not matches_list:
        return ParseResult("sorry", no_block=True)

    match = matches_list[-1]
    content = match.group(1).strip()
    if not content:
        return ParseResult("sorry", no_block=True)

    # Map match span (in region) to character offsets in raw_output.
    if _THINK_CLOSE in text:
        suffix = text.split(_THINK_CLOSE)[-1]
        region_start_in_text = (
            text.rfind(_THINK_CLOSE) + len(_THINK_CLOSE) + (len(suffix) - len(suffix.lstrip()))
        )
    else:
        region_start_in_text = 0

    code_start_in_text = region_start_in_text + match.start(1)
    code_end_in_text = region_start_in_text + match.end(1)
    leading_ws = len(raw_output) - len(raw_output.lstrip())
    code_start_in_raw = leading_ws + code_start_in_text
    code_end_in_raw = leading_ws + code_end_in_text

    return ParseResult(
        content,
        code_block_start_char=code_start_in_raw,
        code_block_end_char=code_end_in_raw,
    )


def create_full_lean_code(
    theorem_code: str,
    extracted_block: str,
    default_header: str,
) -> str:
    """
    Assemble the final Lean 4 source string to submit to the verifier.

    Three cases: block has import (return as-is), block non-trivial (prepend header),
    block "sorry" (default_header + theorem_code).
    """
    if extracted_block == "sorry":
        return f"{default_header}\n\n{theorem_code}"

    if "import" in extracted_block:
        return extracted_block

    return f"{default_header}\n\n{extracted_block}"


# -----------------------------------------------------------------------------
# COT / Answer split (for teacher-ignores-COT mode)
# -----------------------------------------------------------------------------


def split_cot_and_answer(raw_output: str) -> tuple[str, str]:
    """
    Split model output into COT (before last </think>) and answer (after last </think>).

    Returns:
        (cot_text, answer_text). If no </think>, cot_text="", answer_text=raw_output.
    """
    if not raw_output:
        return ("", "")
    text = raw_output.strip()
    if _THINK_CLOSE not in text:
        return ("", text)
    last_idx = text.rfind(_THINK_CLOSE)
    cot = text[:last_idx].strip()
    answer = text[last_idx + len(_THINK_CLOSE) :].strip()
    return (cot, answer)


def get_answer_token_slice(
    raw_output: str,
    generated_ids: list[int],
    tokenizer,
) -> tuple[list[int], int]:
    """
    Return (answer_ids, cot_len) where answer_ids = generated_ids[cot_len:].

    Maps character offset (after </think>) to token boundary. Uses skip_special_tokens=False
    for consistency with generation tokenization. On verification failure, falls back
    to (generated_ids, 0) to avoid corrupting training with a wrong slice.
    """
    # Edge: empty output
    if not generated_ids or not raw_output.strip():
        return (list(generated_ids), 0)

    # Answer start character: after last </think>, or 0 if none
    if _THINK_CLOSE in raw_output:
        answer_start_char = raw_output.rfind(_THINK_CLOSE) + len(_THINK_CLOSE)
    else:
        answer_start_char = 0

    # No COT — entire output is answer
    if answer_start_char == 0:
        return (list(generated_ids), 0)

    # Optional: sanity check decode round-trip
    try:
        full_decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
        if abs(len(full_decoded) - len(raw_output)) > 10:
            _LOG.warning(
                "get_answer_token_slice: raw_output len=%d vs decode len=%d; falling back to full",
                len(raw_output),
                len(full_decoded),
            )
            return (list(generated_ids), 0)
    except Exception:
        pass

    # Find smallest i where decode(generated_ids[:i]) length >= answer_start_char
    cot_len = 0
    for i in range(1, len(generated_ids) + 1):
        prefix = tokenizer.decode(generated_ids[:i], skip_special_tokens=False)
        if len(prefix) >= answer_start_char:
            cot_len = i
            break

    # No valid i — fallback
    if cot_len == 0 and answer_start_char > 0:
        _LOG.warning(
            "get_answer_token_slice: could not find token boundary at char %d; falling back to full",
            answer_start_char,
        )
        return (list(generated_ids), 0)

    # Verification: decoded prefix should match raw_output[:answer_start_char]
    expected_prefix = raw_output[:answer_start_char]
    decoded_prefix = tokenizer.decode(generated_ids[:cot_len], skip_special_tokens=False)

    # Allow small whitespace/normalization differences; strict mismatch → fallback
    if abs(len(decoded_prefix) - len(expected_prefix)) > 5:
        _LOG.warning(
            "get_answer_token_slice: prefix len mismatch (decoded=%d, expected=%d); falling back to full",
            len(decoded_prefix),
            len(expected_prefix),
        )
        return (list(generated_ids), 0)

    if decoded_prefix.rstrip() != expected_prefix.rstrip():
        # Soft check: at least lengths should be similar
        if abs(len(decoded_prefix.rstrip()) - len(expected_prefix.rstrip())) > 10:
            _LOG.warning(
                "get_answer_token_slice: prefix content mismatch; falling back to full"
            )
            return (list(generated_ids), 0)

    answer_ids = generated_ids[cot_len:]
    return (answer_ids, cot_len)


def get_code_token_slice(
    raw_output: str,
    generated_ids: list[int],
    tokenizer,
    code_start_char: int,
    code_end_char: int,
) -> tuple[list[int], int]:
    """
    Return (code_ids, code_start_token_index) where code_ids is the slice of
    generated_ids that corresponds to the character span [code_start_char, code_end_char]
    in raw_output.

    Used for code_only teacher mode: the slice is the parsed lean4 block's tokens.
    On edge cases (empty ids, decode mismatch, invalid span), falls back to
    (generated_ids, 0) so training is not corrupted.
    """
    if not generated_ids:
        return (list(generated_ids), 0)

    if code_start_char >= code_end_char or code_end_char > len(raw_output):
        _LOG.warning(
            "get_code_token_slice: invalid span [%d, %d] for raw_output len=%d; falling back to full",
            code_start_char, code_end_char, len(raw_output),
        )
        return (list(generated_ids), 0)

    try:
        full_decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
        if abs(len(full_decoded) - len(raw_output)) > 10:
            _LOG.warning(
                "get_code_token_slice: raw_output len=%d vs decode len=%d; falling back to full",
                len(raw_output), len(full_decoded),
            )
            return (list(generated_ids), 0)
    except Exception:
        return (list(generated_ids), 0)

    # Find smallest i such that decode(ids[:i]) length >= code_start_char
    code_start_token = 0
    for i in range(1, len(generated_ids) + 1):
        prefix = tokenizer.decode(generated_ids[:i], skip_special_tokens=False)
        if len(prefix) >= code_start_char:
            code_start_token = i
            break

    # Find smallest j >= code_start_token such that decode(ids[:j]) length >= code_end_char
    code_end_token = code_start_token
    for j in range(code_start_token, len(generated_ids) + 1):
        prefix = tokenizer.decode(generated_ids[:j], skip_special_tokens=False)
        if len(prefix) >= code_end_char:
            code_end_token = j
            break

    if code_end_token <= code_start_token:
        _LOG.warning(
            "get_code_token_slice: could not find token span for [%d, %d]; falling back to full",
            code_start_char, code_end_char,
        )
        return (list(generated_ids), 0)

    code_ids = generated_ids[code_start_token:code_end_token]
    return (code_ids, code_start_token)
