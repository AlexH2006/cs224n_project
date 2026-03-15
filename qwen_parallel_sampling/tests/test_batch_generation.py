"""
Unit tests for qwen_eval/batch_generation.py.

Validates flat prompt building and unflatten logic without Modal or vLLM.
Run with:
    python -m pytest qwen_eval/tests/test_batch_generation.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from qwen_eval.batch_generation import build_flat_prompts_and_meta, unflatten_results
from qwen_eval.config import EvalConfig
from qwen_eval.parsing import create_full_lean_code, extract_full_lean_block_parsed


def _fake_builder(p: dict) -> str:
    """Returns a deterministic prompt string for testing."""
    return f"prompt_for_idx_{p['problem_idx']}"


# ---------------------------------------------------------------------------
# build_flat_prompts_and_meta
# ---------------------------------------------------------------------------


class TestBuildFlatPromptsAndMeta:
    def test_empty_problems(self):
        flat, meta = build_flat_prompts_and_meta([], 4, _fake_builder)
        assert flat == []
        assert meta == []

    def test_single_problem_pass_k_1(self):
        problems = [{"problem_idx": 0, "formal_statement": "x", "informal_stmt": "y", "header": ""}]
        flat, meta = build_flat_prompts_and_meta(problems, 1, _fake_builder)
        assert flat == ["prompt_for_idx_0"]
        assert meta == [(0, "prompt_for_idx_0")]

    def test_single_problem_pass_k_4(self):
        problems = [{"problem_idx": 0, "formal_statement": "x", "informal_stmt": "y", "header": ""}]
        flat, meta = build_flat_prompts_and_meta(problems, 4, _fake_builder)
        assert flat == ["prompt_for_idx_0"] * 4
        assert meta == [(0, "prompt_for_idx_0")] * 4

    def test_two_problems_pass_k_2(self):
        problems = [
            {"problem_idx": 0, "formal_statement": "a", "informal_stmt": "b", "header": ""},
            {"problem_idx": 1, "formal_statement": "c", "informal_stmt": "d", "header": ""},
        ]
        flat, meta = build_flat_prompts_and_meta(problems, 2, _fake_builder)
        assert flat == ["prompt_for_idx_0", "prompt_for_idx_0", "prompt_for_idx_1", "prompt_for_idx_1"]
        assert meta == [
            (0, "prompt_for_idx_0"),
            (0, "prompt_for_idx_0"),
            (1, "prompt_for_idx_1"),
            (1, "prompt_for_idx_1"),
        ]

    def test_meta_same_length_as_flat(self):
        problems = [
            {"problem_idx": i, "formal_statement": "", "informal_stmt": "", "header": ""}
            for i in range(5)
        ]
        flat, meta = build_flat_prompts_and_meta(problems, 3, _fake_builder)
        assert len(flat) == 5 * 3
        assert len(meta) == len(flat)


# ---------------------------------------------------------------------------
# unflatten_results
# ---------------------------------------------------------------------------


class TestUnflattenResults:
    def test_empty(self):
        result = unflatten_results([], [], [])
        assert result == []

    def test_single_problem_pass_k_2(self):
        prompt_meta = [(0, "p0"), (0, "p0")]
        flat_outputs = [("out0", "stop"), ("out1", "length")]
        problems = [{"problem_idx": 0}]
        result = unflatten_results(prompt_meta, flat_outputs, problems)
        assert len(result) == 1
        assert result[0] == ["p0", ("out0", "stop"), ("out1", "length")]

    def test_two_problems_order_preserved(self):
        # problem_idx 1 first, then 0 — problems list order is [p1, p0]
        prompt_meta = [(1, "p1"), (1, "p1"), (0, "p0"), (0, "p0")]
        flat_outputs = [
            ("a", "stop"),
            ("b", "stop"),
            ("c", "stop"),
            ("d", "stop"),
        ]
        problems = [
            {"problem_idx": 1},
            {"problem_idx": 0},
        ]
        result = unflatten_results(prompt_meta, flat_outputs, problems)
        assert len(result) == 2
        # Result order must match problems order: first problem_idx 1, then 0.
        assert result[0] == ["p1", ("a", "stop"), ("b", "stop")]
        assert result[1] == ["p0", ("c", "stop"), ("d", "stop")]

    def test_length_mismatch_raises(self):
        prompt_meta = [(0, "p")] * 3
        flat_outputs = [("x", "stop"), ("y", "stop")]  # length 2
        problems = [{"problem_idx": 0}]
        with pytest.raises(ValueError, match="must match"):
            unflatten_results(prompt_meta, flat_outputs, problems)


# ---------------------------------------------------------------------------
# Round-trip: build_flat + unflatten
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_shape_matches_driver_contract(self):
        """After unflatten, result[i] has shape [prompt, (raw_0, r0), ..., (raw_{k-1}, r_{k-1})]."""
        problems = [
            {"problem_idx": 0, "formal_statement": "a", "informal_stmt": "b", "header": ""},
            {"problem_idx": 1, "formal_statement": "c", "informal_stmt": "d", "header": ""},
        ]
        pass_k = 3
        flat, meta = build_flat_prompts_and_meta(problems, pass_k, _fake_builder)
        # Simulate vLLM outputs (same order as flat).
        flat_outputs = [(f"raw_{i}", "stop") for i in range(len(flat))]
        raw_results = unflatten_results(meta, flat_outputs, problems)

        assert len(raw_results) == 2
        for i, res in enumerate(raw_results):
            assert len(res) == 1 + pass_k  # prompt + pass_k (raw, reason) pairs
            assert res[0] == f"prompt_for_idx_{i}"
            for j in range(pass_k):
                assert res[1 + j] == (f"raw_{i * pass_k + j}", "stop")


# ---------------------------------------------------------------------------
# Driver contract: raw_results shape consumed by modal_app parsing loop
# ---------------------------------------------------------------------------


class TestDriverContract:
    """Ensure unflatten_results output matches what run_eval's parsing loop expects."""

    def test_raw_result_shape_parsed_by_driver_loop(self):
        """Fake raw_results in generate_all shape; run same parse logic as modal_app."""
        cfg = EvalConfig()
        problems = [
            {
                "problem_idx": 0,
                "formal_statement": "theorem t : 1 = 1 := by sorry",
                "informal_stmt": "trivial",
                "header": "import Mathlib",
            },
            {
                "problem_idx": 1,
                "formal_statement": "theorem u : 2 = 2 := by sorry",
                "informal_stmt": "trivial",
                "header": "import Mathlib",
            },
        ]
        pass_k = 2
        raw_results = []
        for i in range(2):
            prompt = f"prompt_{i}"
            raw_ok = "<think>\nOk.\n</think>\n```lean4\ntheorem t : 1 = 1 := by norm_num\n```"
            raw_results.append([prompt, (raw_ok, "stop"), (raw_ok, "stop")])

        all_attempts = []
        for problem, result in zip(problems, raw_results):
            prompt = result[0]
            raw_pairs = result[1:]
            problem_attempts = []
            for attempt_idx, (raw_output, finish_reason) in enumerate(raw_pairs):
                parse_result = extract_full_lean_block_parsed(
                    raw_output, finish_reason=finish_reason
                )
                full_code = create_full_lean_code(
                    theorem_code=problem["formal_statement"],
                    extracted_block=parse_result.block,
                    default_header=cfg.default_header,
                )
                problem_attempts.append({
                    "problem_idx": problem["problem_idx"],
                    "attempt": attempt_idx,
                    "prompt": prompt,
                    "full_code": full_code,
                })
            all_attempts.append(problem_attempts)

        assert len(all_attempts) == 2
        assert len(all_attempts[0]) == pass_k
        assert all_attempts[0][0]["prompt"] == "prompt_0"
        assert "norm_num" in all_attempts[0][0]["full_code"]
