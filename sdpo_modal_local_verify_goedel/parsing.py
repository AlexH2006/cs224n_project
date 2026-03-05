"""
TLDR: Extract the last lean4 fenced code block from Goedel model raw_output.
Goedel output has multiple sections (Detailed Proof, Abstract Plan, have Statements,
Complete Lean 4 Proof); the final proof is in the last ```lean4 block only.
(1) Incomplete (<think> without </think>) or no/empty block → return "sorry";
(2) Last ```lean4 fence opener has no closing ``` (truncated output) → return "sorry";
(3) Otherwise → return the entire content of the last ```lean4 block (no lean/tactics).
"""

import re

# Goedel: only match ```lean4 blocks; take the last one (the complete proof).
_CODE_BLOCK_PATTERN = re.compile(
    r"```lean4\s*\n(.*?)```",
    re.DOTALL,
)


def _has_incomplete_reasoning(text: str) -> bool:
    """True iff <think> is present and </think> is absent (reasoning cut off)."""
    return "<think>" in text and "</think>" not in text


def _has_truncated_final_block(text: str) -> bool:
    """True iff the last ```lean4 opener in the text has no matching closing ```.

    Goedel outputs multiple lean4 blocks; when generation is cut off mid-proof
    the final block is unclosed. In that case the regex falls back to an earlier
    intermediate block (e.g. the "have Statements" sketch), which is wrong.
    Detecting truncation here lets us return "sorry" instead.
    """
    last_open = text.rfind("```lean4")
    if last_open == -1:
        return False
    after_open = text[last_open + len("```lean4"):]
    return "```" not in after_open


def _search_region(text: str) -> str:
    """
    Return the region to search for code blocks.
    If </think> is present, use only the text after the last </think> (the answer).
    Otherwise search the whole text (e.g. no think wrapper).
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def is_truncated_output(raw_output: str) -> bool:
    """True iff the model output is truncated (reasoning or final code block cut off).

    Covers two cases:
      1. <think> present but </think> absent (think-model reasoning cut off).
      2. The last ```lean4 block has no closing ``` (Goedel-style output cut off
         mid-proof, no <think> tags used).

    Use this alongside extract_full_lean_block in trainer_core to set is_truncated.
    """
    if not raw_output or not raw_output.strip():
        return False
    text = raw_output.strip()
    if _has_incomplete_reasoning(text):
        return True
    region = _search_region(text)
    return _has_truncated_final_block(region)


def extract_full_lean_block(raw_output: str) -> str:
    """
    Extract the last lean4 fenced block from Goedel model output.

    Returns "sorry" in any of these cases:
      - Empty output.
      - <think> present but </think> absent (reasoning cut off).
      - The last ```lean4 block is unclosed (generation truncated mid-proof).
      - No ```lean4 block found, or the last block is empty.

    Otherwise returns the entire content of the last closed ```lean4 block.
    Goedel puts the complete proof in the final lean4 block.
    """
    if not raw_output or not raw_output.strip():
        return "sorry"

    text = raw_output.strip()
    if _has_incomplete_reasoning(text):
        return "sorry"

    region = _search_region(text)

    if _has_truncated_final_block(region):
        return "sorry"

    matches = _CODE_BLOCK_PATTERN.findall(region)
    if not matches:
        return "sorry"

    content = matches[-1].strip()
    return content if content else "sorry"
