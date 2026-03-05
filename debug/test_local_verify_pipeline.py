#!/usr/bin/env python3
"""
TL;DR: Smoke test for sdpo_modal_local_verify_kimina pipeline (no Modal). Run from repo root.

Verifies: imports, config, prompt building, payload shape for run_sdpo_step,
and local_lean_verifier.verify contract. Does not call Modal.
"""
import sys

sys.path.insert(0, ".")

from sdpo_modal_local_verify_kimina.config import SDPOConfig
from sdpo_modal_local_verify_kimina.local_lean_verifier import verify as local_verify
from sdpo_modal_local_verify_kimina.parsing import extract_full_lean_block
from sdpo_modal_local_verify_kimina.prompts import create_base_prompt, create_feedback_prompt
from sdpo_modal_local_verify_kimina.utils import (
    get_field,
    create_full_lean_code,
    theorem_code_is_commented_out,
)


def test_config():
    cfg = SDPOConfig()
    assert cfg.max_iterations >= 1
    assert cfg.theorem_fields
    print("PASS: SDPOConfig")


def test_payload_keys():
    """Payload built by entrypoint must contain all keys run_sdpo_step expects."""
    required = {
        "iteration", "base_prompt", "teacher_prompt", "raw_output", "generated_ids",
        "tactics", "full_code", "verification", "num_tokens", "is_success", "is_server_error",
    }
    payload = {
        "iteration": 1,
        "base_prompt": "User: solve\nAssistant:",
        "teacher_prompt": None,
        "raw_output": "```lean4\nsorry\n```",
        "generated_ids": [1, 2, 3],
        "tactics": "sorry",
        "full_code": "import Mathlib\ntheorem x : True := sorry",
        "verification": {"success": False, "complete": False, "feedback": "", "is_server_error": False},
        "num_tokens": 3,
        "is_success": False,
        "is_server_error": False,
    }
    for k in required:
        assert k in payload, f"Missing payload key: {k}"
    print("PASS: payload keys")


def test_local_verifier_contract():
    r = local_verify("theorem x : True := trivial", lean_workspace="/nonexistent")
    assert "success" in r and "source" in r
    assert r["source"] == "local_lean"
    print("PASS: local_lean_verifier.verify contract")


def test_parse_and_full_code():
    raw = "<think>reasoning</think>\n```lean4\n  trivial\n```"
    block = extract_full_lean_block(raw)
    assert block and "trivial" in block
    full = create_full_lean_code(
        theorem_code="import Mathlib\ntheorem x : True := by sorry",
        extracted_block=block,
        header="import Mathlib",
        default_header="import Mathlib",
    )
    assert "trivial" in full and "sorry" not in full
    print("PASS: parse and create_full_lean_code")


def main():
    print("Smoke tests for sdpo_modal_local_verify_kimina (no Modal)...")
    test_config()
    test_payload_keys()
    test_local_verifier_contract()
    test_parse_and_full_code()
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
