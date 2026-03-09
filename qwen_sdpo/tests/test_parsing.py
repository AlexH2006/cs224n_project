"""
TLDR: Unit tests for qwen_sdpo.parsing — get_answer_token_slice and related functions.

Covers happy path, no think tag, multiple think tags, empty output, and round-trip.
"""

import pytest

from qwen_sdpo.parsing import (
    create_full_lean_code,
    extract_full_lean_block_parsed,
    get_answer_token_slice,
    get_code_token_slice,
    split_cot_and_answer,
)


class TestSplitCotAndAnswer:
    """Tests for split_cot_and_answer."""

    def test_no_think_tag(self):
        text = "Here is the proof:\n```lean4\nimport Mathlib\n```"
        cot, answer = split_cot_and_answer(text)
        assert cot == ""
        assert answer == text

    def test_with_think_tag(self):
        text = "<think>Let me think...</think>\n```lean4\nimport Mathlib\n```"
        cot, answer = split_cot_and_answer(text)
        assert cot == "<think>Let me think..."
        assert "import Mathlib" in answer

    def test_empty(self):
        cot, answer = split_cot_and_answer("")
        assert cot == ""
        assert answer == ""

    def test_multiple_think_tags(self):
        text = "<think>first</think>\n<think>second</think>\nanswer"
        cot, answer = split_cot_and_answer(text)
        assert "first" in cot and "second" in cot
        assert answer == "answer"


class TestGetAnswerTokenSlice:
    """Tests for get_answer_token_slice — token alignment."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)

    def test_no_think_tag(self, tokenizer):
        raw = "Here is the lean4 block.\n```lean4\nimport Mathlib\n```"
        ids = tokenizer.encode(raw, add_special_tokens=False)
        answer_ids, cot_len = get_answer_token_slice(raw, ids, tokenizer)
        assert cot_len == 0
        assert answer_ids == ids

    def test_with_think_tag(self, tokenizer):
        raw = "<think>reasoning</think>\n```lean4\nimport Mathlib\n```"
        ids = tokenizer.encode(raw, add_special_tokens=False)
        answer_ids, cot_len = get_answer_token_slice(raw, ids, tokenizer)
        assert cot_len > 0
        assert len(answer_ids) + cot_len == len(ids)
        decoded_answer = tokenizer.decode(answer_ids, skip_special_tokens=False)
        assert "import Mathlib" in decoded_answer
        assert "<think>" not in decoded_answer or "</think>" in decoded_answer

    def test_empty_output(self, tokenizer):
        answer_ids, cot_len = get_answer_token_slice("", [], tokenizer)
        assert answer_ids == []
        assert cot_len == 0

    def test_round_trip(self, tokenizer):
        raw = "<think>step 1</think>\nanswer text"
        ids = tokenizer.encode(raw, add_special_tokens=False)
        answer_ids, cot_len = get_answer_token_slice(raw, ids, tokenizer)
        prefix_decoded = tokenizer.decode(ids[:cot_len], skip_special_tokens=False)
        expected_prefix = raw[: raw.rfind("</think>") + len("</think>")]
        assert len(prefix_decoded) >= len(expected_prefix) - 5


class TestExtractFullLeanBlockParsed:
    """Tests for extract_full_lean_block_parsed (copied from qwen_eval)."""

    def test_success(self):
        raw = "<think>...</think>\n```lean4\nimport Mathlib\n\ntheorem x : True := by trivial\n```"
        r = extract_full_lean_block_parsed(raw)
        assert not r.failed
        assert "import Mathlib" in r.block
        assert r.truncated is False
        assert r.no_block is False
        assert r.code_block_start_char is not None
        assert r.code_block_end_char is not None
        assert r.code_block_start_char < r.code_block_end_char
        assert raw[r.code_block_start_char:r.code_block_end_char].strip() == r.block.strip()

    def test_no_block(self):
        raw = "<think>...</think>\nNo code here."
        r = extract_full_lean_block_parsed(raw)
        assert r.failed
        assert r.no_block
        assert r.block == "sorry"

    def test_truncated_by_finish_reason(self):
        raw = "<think>incomplete"
        r = extract_full_lean_block_parsed(raw, finish_reason="length")
        assert r.failed
        assert r.truncated
        assert r.block == "sorry"


class TestCreateFullLeanCode:
    """Tests for create_full_lean_code."""

    def test_sorry_fallback(self):
        code = create_full_lean_code("theorem x : True := sorry", "sorry", "import Mathlib")
        assert "import Mathlib" in code
        assert "theorem x" in code

    def test_self_contained(self):
        block = "import Mathlib\n\ntheorem x : True := by trivial"
        code = create_full_lean_code("theorem x : True := sorry", block, "import Other")
        assert code == block


class TestGetCodeTokenSlice:
    """Tests for get_code_token_slice — code-only teacher mode."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)

    def test_success_returns_slice(self, tokenizer):
        raw = "<think>...</think>\n```lean4\nimport Mathlib\n```"
        r = extract_full_lean_block_parsed(raw)
        assert r.code_block_start_char is not None and r.code_block_end_char is not None
        ids = tokenizer.encode(raw, add_special_tokens=False)
        code_ids, start = get_code_token_slice(
            raw, ids, tokenizer, r.code_block_start_char, r.code_block_end_char
        )
        assert start >= 0
        assert len(code_ids) + start <= len(ids)
        decoded = tokenizer.decode(code_ids, skip_special_tokens=False)
        assert "import Mathlib" in decoded

    def test_fallback_on_invalid_span(self, tokenizer):
        raw = "some text"
        ids = tokenizer.encode(raw, add_special_tokens=False)
        code_ids, start = get_code_token_slice(raw, ids, tokenizer, 0, 0)
        assert start == 0
        assert code_ids == ids
