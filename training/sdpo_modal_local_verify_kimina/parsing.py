"""
TLDR: Single place for extracting a full Lean 4 code block from model raw_output.
Used by the SDPO local-verify pipeline (utils, entrypoint, trainer_core). Two outcomes:
(1) Incomplete (<think> without </think>) or no/empty block → return "sorry";
(2) Otherwise → return the entire content of the last ```lean4/lean/tactics block.
No tactic filtering; no dependency on debug/.
"""

import re

# Matches ```lean4, ```lean, or ```tactics (optional) then newline then content until ```.
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:lean4?|lean|tactics)?\s*\n(.*?)```",
    re.DOTALL,
)


def _has_incomplete_reasoning(text: str) -> bool:
    """True iff <think> is present and </think> is absent (reasoning cut off)."""
    return "<think>" in text and "</think>" not in text


def _search_region(text: str) -> str:
    """
    Return the region to search for code blocks.
    If </think> is present, use only the text after the last </think> (the answer).
    Otherwise search the whole text (e.g. no think wrapper).
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def extract_full_lean_block(raw_output: str) -> str:
    """
    Extract the full content of one Lean code block from model output.

    Only two possibilities:
      - Incomplete: <think> present but </think> absent → return "sorry".
      - Complete: parse and return the entire code block (last ```lean4/lean/tactics
        block after </think>, or in whole output if no think wrapper). No block or empty
        block → "sorry".

    Returns:
      The block content as a string, or "sorry" if incomplete (think tokens) or no block.
    """
    if not raw_output or not raw_output.strip():
        return "sorry"

    text = raw_output.strip()
    if _has_incomplete_reasoning(text):
        return "sorry"

    region = _search_region(text)
    matches = _CODE_BLOCK_PATTERN.findall(region)
    if not matches:
        return "sorry"

    content = matches[-1].strip()
    return content if content else "sorry"
