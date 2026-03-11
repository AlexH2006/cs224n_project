"""
Unit tests for qwen_sdpo/prompts.py.

TLDR: Validates that student and teacher prompts are correctly formatted
without requiring GPU or network access. Uses a minimal mock tokenizer that
echoes apply_chat_template inputs as plain text.

Run with:
    python -m pytest qwen_sdpo/tests/test_prompts.py -v
    # or directly:
    python qwen_sdpo/tests/test_prompts.py
"""

import sys
from pathlib import Path

# Ensure the project root is on the path when run directly.
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qwen_sdpo.config import SDPOConfig
from qwen_sdpo.prompts import (
    _STUDENT_PROMPT_NO_THINKING,
    _STUDENT_PROMPT_THINKING,
    _TEACHER_PROMPT_NO_THINKING,
    _TEACHER_PROMPT_THINKING,
    build_student_prompt,
    build_teacher_prompt,
    build_teacher_prompt_with_full_generation,
)


class _MockTokenizer:
    """Minimal tokenizer stub that formats chat messages as plain text."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        parts = []
        for m in messages:
            parts.append(f"[{m['role'].upper()}]\n{m['content']}")
        if add_generation_prompt:
            parts.append("[ASSISTANT]")
        return "\n---\n".join(parts)


_THEOREM_CODE = "theorem mathd_algebra_001 (n : ℕ) : n + 0 = n := by\n  sorry"
_INFORMAL = "Show that n + 0 = n for any natural number n."
_HEADER = "import Mathlib\nset_option maxHeartbeats 400000"
_ERROR = "unknown tactic 'exact n'\n  expected type: n + 0 = n"
_FAILED_PROOF = "theorem mathd_algebra_001 (n : ℕ) : n + 0 = n := by\n  ring_nf; simp [Nat.add_zero_unique_marker]"


def test_student_prompt_contains_theorem():
    cfg = SDPOConfig()
    tok = _MockTokenizer()
    prompt = build_student_prompt(_THEOREM_CODE, _INFORMAL, _HEADER, tok, cfg)
    assert "mathd_algebra_001" in prompt, "Theorem name should appear in student prompt"
    assert "sorry" in prompt, "Theorem with sorry should appear in student prompt"
    assert "Reason step-by-step" in prompt or "step-by-step" in prompt, "CoT instruction should be present"
    assert "lean4" in prompt, "lean4 fence should be in prompt"


def test_student_prompt_no_feedback():
    cfg = SDPOConfig()
    tok = _MockTokenizer()
    prompt = build_student_prompt(_THEOREM_CODE, _INFORMAL, _HEADER, tok, cfg)
    assert "INCORRECT" not in prompt, "Student prompt must not contain teacher feedback"
    assert "compiler returned" not in prompt, "Student prompt must not contain error block"


def test_student_prompt_with_use_think_mode_false():
    """build_student_prompt with use_think_mode=False still produces a valid prompt."""
    cfg = SDPOConfig(use_think_mode=False)
    tok = _MockTokenizer()
    prompt = build_student_prompt(_THEOREM_CODE, _INFORMAL, _HEADER, tok, cfg)
    assert "mathd_algebra_001" in prompt
    assert "sorry" in prompt
    assert "step-by-step" in prompt or "Reason" in prompt
    assert "[ASSISTANT]" in prompt, "Should end with generation prompt"


def test_teacher_prompt_with_full_generation_contains_generation_and_feedback():
    """build_teacher_prompt_with_full_generation embeds full_generation and feedback."""
    cfg = SDPOConfig()
    tok = _MockTokenizer()
    full_gen = "First I tried X. Then ```lean4\n  sorry\n```"
    prompt = build_teacher_prompt_with_full_generation(
        _THEOREM_CODE, _INFORMAL, _HEADER, _ERROR, full_gen, tok, cfg
    )
    assert full_gen in prompt, "Full generation must appear in teacher prompt"
    assert _ERROR in prompt, "Feedback must appear in teacher prompt"
    assert "mathd_algebra_001" in prompt
    assert "Your previous attempt" in prompt or "full_generation" in prompt or "Errors to avoid" in prompt


def test_teacher_prompt_structure():
    """Teacher format: Solve + informal + theorem + feedback + closing instruction."""
    cfg = SDPOConfig()
    tok = _MockTokenizer()
    prompt = build_teacher_prompt(
        _THEOREM_CODE, _INFORMAL, _HEADER,
        _ERROR, tok, cfg,
    )
    assert "Solve the following Lean 4 problem" in prompt
    assert "You MUST avoid the following errors" in prompt or "feedback" in prompt.lower()
    assert "Do NOT use" in prompt or "lean4 code block" in prompt, "Closing instruction present"


def test_teacher_prompt_contains_error():
    cfg = SDPOConfig()
    tok = _MockTokenizer()
    prompt = build_teacher_prompt(
        _THEOREM_CODE, _INFORMAL, _HEADER,
        _ERROR, tok, cfg,
    )
    assert "unknown tactic" in prompt, "Teacher prompt should include the raw error text"


def test_teacher_prompt_contains_required_sections():
    """Teacher prompt contains problem, feedback section, and closing instruction."""
    cfg = SDPOConfig()
    tok = _MockTokenizer()
    teacher = build_teacher_prompt(
        _THEOREM_CODE, _INFORMAL, _HEADER,
        _ERROR, tok, cfg,
    )
    assert "Solve the following Lean 4 problem" in teacher
    assert "You MUST avoid the following errors" in teacher or "feedback" in teacher.lower()
    assert "Do NOT use" in teacher or "lean4" in teacher, "Closing instruction present"
    assert "mathd_algebra_001" in teacher


def test_default_header_fallback():
    """When no header is provided, the default_header from config is used."""
    cfg = SDPOConfig(default_header="import Mathlib\nset_option maxHeartbeats 400000")
    tok = _MockTokenizer()
    prompt = build_student_prompt(_THEOREM_CODE, _INFORMAL, "", tok, cfg)
    assert "import Mathlib" in prompt, "Default header should appear when no dataset header provided"


def test_informal_fallback():
    """Missing informal description uses a placeholder."""
    cfg = SDPOConfig()
    tok = _MockTokenizer()
    prompt = build_student_prompt(_THEOREM_CODE, "", _HEADER, tok, cfg)
    assert "no informal description" in prompt, "Should use placeholder when informal is empty"


def test_config_teacher_response_mode():
    """teacher_response_mode accepts full_output, answer_only, code_only; default is full_output."""
    cfg = SDPOConfig()
    assert cfg.teacher_response_mode == "full_output"
    for mode in ("full_output", "answer_only", "code_only"):
        c = SDPOConfig(teacher_response_mode=mode)
        assert c.teacher_response_mode == mode


def test_config_use_think_mode():
    """use_think_mode defaults to True; accepts True/False for non-thinking mode."""
    cfg = SDPOConfig()
    assert cfg.use_think_mode is True
    cfg_no_think = SDPOConfig(use_think_mode=False)
    assert cfg_no_think.use_think_mode is False
    cfg_think = SDPOConfig(use_think_mode=True)
    assert cfg_think.use_think_mode is True


def test_four_templates_exist_with_expected_placeholders():
    """The four prompt templates exist and contain required format keys."""
    assert "{informal}" in _STUDENT_PROMPT_THINKING and "{header_and_theorem}" in _STUDENT_PROMPT_THINKING
    assert "{informal}" in _STUDENT_PROMPT_NO_THINKING and "{header_and_theorem}" in _STUDENT_PROMPT_NO_THINKING
    assert "{informal}" in _TEACHER_PROMPT_THINKING and "{header_and_theorem}" in _TEACHER_PROMPT_THINKING
    assert "{feedback}" in _TEACHER_PROMPT_THINKING
    assert "{informal}" in _TEACHER_PROMPT_NO_THINKING and "{header_and_theorem}" in _TEACHER_PROMPT_NO_THINKING
    assert "{feedback}" in _TEACHER_PROMPT_NO_THINKING and "{full_generation}" in _TEACHER_PROMPT_NO_THINKING


def test_teacher_prompt_enable_thinking_param():
    """build_teacher_prompt accepts enable_thinking and passes it through without error."""
    cfg = SDPOConfig()
    tok = _MockTokenizer()
    prompt = build_teacher_prompt(
        _THEOREM_CODE, _INFORMAL, _HEADER, _ERROR, tok, cfg,
        enable_thinking=False,
    )
    assert "Solve the following Lean 4 problem" in prompt
    assert "unknown tactic" in prompt


def test_make_run_dir_uses_teacher_response_mode():
    """make_run_dir puts runs under sdpo_results/{model_tag}/{teacher_response_mode}/."""
    from qwen_sdpo.results import make_run_dir

    for mode in ("full_output", "answer_only", "code_only"):
        cfg = SDPOConfig(model_name="Qwen/Qwen3.5-4B", teacher_response_mode=mode)
        model_tag, run_dir = make_run_dir(cfg, problem_idx=0)
        assert model_tag == "Qwen3.5-4B"
        assert mode in str(run_dir), f"run_dir should contain {mode!r}"


if __name__ == "__main__":
    tests = [
        test_student_prompt_contains_theorem,
        test_student_prompt_no_feedback,
        test_student_prompt_with_use_think_mode_false,
        test_teacher_prompt_with_full_generation_contains_generation_and_feedback,
        test_teacher_prompt_structure,
        test_teacher_prompt_contains_error,
        test_teacher_prompt_contains_required_sections,
        test_default_header_fallback,
        test_informal_fallback,
        test_teacher_prompt_enable_thinking_param,
        test_config_teacher_response_mode,
        test_config_use_think_mode,
        test_four_templates_exist_with_expected_placeholders,
        test_make_run_dir_uses_teacher_response_mode,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
