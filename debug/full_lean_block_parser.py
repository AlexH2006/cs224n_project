"""
TLDR: Extract the entire content of one Lean 4 code block from model raw_output.
Only two outcomes: (1) Generation incomplete (<think> present but </think> absent) → "sorry".
(2) Generation complete → parse and return the entire code block. No other constraints.
No tactic filtering. Used by test_full_lean_block_parser.py and pipelines that need
full-block extraction from SDPO-style logs.
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
