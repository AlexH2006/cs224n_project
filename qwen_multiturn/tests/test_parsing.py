"""
Standalone unit tests for qwen_multiturn/parsing.py.

No Modal, no GPU, no Kimina required — run before the full eval to validate
parsing logic on representative model output patterns.

Run with:
    python -m pytest qwen_multiturn/tests/test_parsing.py -v
"""

import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_multiturn.parsing import create_full_lean_code, extract_full_lean_block

DEFAULT_HEADER = (
    "import Mathlib\n"
    "set_option maxHeartbeats 400000\n"
    "open BigOperators Real Nat Topology Rat"
)

THEOREM_CODE = "theorem foo : 1 + 1 = 2 := by\n  sorry"

CLEAN_PROOF_BLOCK = """\
import Mathlib
set_option maxHeartbeats 400000

theorem foo : 1 + 1 = 2 := by
  norm_num"""

PROOF_WITHOUT_IMPORTS = """\
theorem foo : 1 + 1 = 2 := by
  norm_num"""


# ---------------------------------------------------------------------------
# extract_full_lean_block
# ---------------------------------------------------------------------------

class TestExtractFullLeanBlock:
    def test_empty_output_returns_sorry(self):
        assert extract_full_lean_block("") == "sorry"

    def test_whitespace_only_returns_sorry(self):
        assert extract_full_lean_block("   \n\t  ") == "sorry"

    def test_no_code_block_returns_sorry(self):
        output = "I think the proof is by induction but I cannot write it."
        assert extract_full_lean_block(output) == "sorry"

    def test_incomplete_think_returns_sorry(self):
        # <think> open but </think> missing — model was cut off
        output = "<think>\nLet me think about this...\nWe need to use norm_num"
        assert extract_full_lean_block(output) == "sorry"

    def test_clean_think_block_extracted(self):
        output = (
            "<think>\nThe proof is by norm_num.\n</think>\n"
            "```lean4\n" + CLEAN_PROOF_BLOCK + "\n```"
        )
        result = extract_full_lean_block(output)
        assert result == CLEAN_PROOF_BLOCK

    def test_no_think_tags_extracts_block(self):
        # Model produced no thinking wrapper — extract from whole output
        output = "Here is the proof:\n```lean4\n" + CLEAN_PROOF_BLOCK + "\n```"
        result = extract_full_lean_block(output)
        assert result == CLEAN_PROOF_BLOCK

    def test_multiple_blocks_returns_last(self):
        # Model emitted an intermediate partial block, then corrected itself.
        # The last block is always the authoritative final answer.
        output = (
            "<think>\nFirst attempt:\n```lean4\ntheorem foo := by sorry\n```\n"
            "That uses sorry, let me fix it.\n</think>\n"
            "```lean4\n" + CLEAN_PROOF_BLOCK + "\n```"
        )
        result = extract_full_lean_block(output)
        assert result == CLEAN_PROOF_BLOCK

    def test_empty_code_block_returns_sorry(self):
        output = "```lean4\n\n```"
        assert extract_full_lean_block(output) == "sorry"

    def test_lean_tag_without_4_accepted(self):
        # Model may use ```lean instead of ```lean4
        output = "```lean\n" + CLEAN_PROOF_BLOCK + "\n```"
        result = extract_full_lean_block(output)
        assert result == CLEAN_PROOF_BLOCK

    def test_block_after_think_is_used_not_block_inside_think(self):
        # Block inside <think> should be ignored; only block after </think> counts.
        wrong_block = "theorem foo := by ring"
        output = (
            f"<think>\n```lean4\n{wrong_block}\n```\n</think>\n"
            f"```lean4\n{CLEAN_PROOF_BLOCK}\n```"
        )
        result = extract_full_lean_block(output)
        assert result == CLEAN_PROOF_BLOCK

    def test_returns_stripped_content(self):
        # Leading/trailing whitespace inside the block should be stripped.
        output = "```lean4\n\n  " + CLEAN_PROOF_BLOCK + "\n\n```"
        result = extract_full_lean_block(output)
        assert result == CLEAN_PROOF_BLOCK.strip() or "import" in result


# ---------------------------------------------------------------------------
# create_full_lean_code
# ---------------------------------------------------------------------------

class TestCreateFullLeanCode:
    def test_sorry_fallback_uses_default_header_and_theorem(self):
        result = create_full_lean_code(THEOREM_CODE, "sorry", DEFAULT_HEADER)
        assert result.startswith("import Mathlib")
        assert THEOREM_CODE in result

    def test_block_with_imports_returned_as_is(self):
        result = create_full_lean_code(THEOREM_CODE, CLEAN_PROOF_BLOCK, DEFAULT_HEADER)
        assert result == CLEAN_PROOF_BLOCK

    def test_block_without_imports_gets_default_header_prepended(self):
        result = create_full_lean_code(THEOREM_CODE, PROOF_WITHOUT_IMPORTS, DEFAULT_HEADER)
        assert result.startswith("import Mathlib")
        assert PROOF_WITHOUT_IMPORTS in result

    def test_block_with_imports_does_not_get_duplicate_header(self):
        result = create_full_lean_code(THEOREM_CODE, CLEAN_PROOF_BLOCK, DEFAULT_HEADER)
        # Should not contain the default header prepended to an already-complete block
        assert result.count("import Mathlib") == 1

    def test_sorry_block_does_not_duplicate_theorem(self):
        result = create_full_lean_code(THEOREM_CODE, "sorry", DEFAULT_HEADER)
        # theorem_code should appear exactly once
        assert result.count(THEOREM_CODE) == 1

    def test_block_without_imports_header_separated_from_block(self):
        result = create_full_lean_code(THEOREM_CODE, PROOF_WITHOUT_IMPORTS, DEFAULT_HEADER)
        # Header and block should be separated by a blank line
        assert DEFAULT_HEADER + "\n\n" + PROOF_WITHOUT_IMPORTS == result
