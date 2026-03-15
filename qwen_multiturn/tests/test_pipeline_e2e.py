"""
End-to-end pipeline smoke test (no Modal, no GPU required).

Tests the full parse → assemble → verify loop against a live Kimina server.
Loads 1 real problem from cat-searcher/minif2f-lean4, synthesises 4 simulated
raw model outputs (covering the main parsing cases), then runs each through
the full pipeline: extract_full_lean_block → create_full_lean_code → verify().

Run with:
    python3 qwen_multiturn/tests/test_pipeline_e2e.py

Kimina Docker must be running locally:
    docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
"""

import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_multiturn.config import EvalConfig
from qwen_multiturn.dataset import get_field, load_problems
from qwen_multiturn.local_lean_verifier import verify
from qwen_multiturn.parsing import create_full_lean_code, extract_full_lean_block
from qwen_multiturn.results import build_problem_log, make_run_dir, save_results


def make_simulated_outputs(header: str, theorem_code: str) -> list[dict]:
    """
    Return 4 simulated raw model outputs covering the main parsing scenarios.

    Each dict has: label, raw_output, expected_block_non_sorry (bool)
    """
    # A correct proof for a simple norm_num theorem (may or may not match the
    # actual problem — point is to test the parsing and verification plumbing).
    correct_block = f"""{header}

{theorem_code.replace("sorry", "norm_num")}"""

    # A block that omits imports (model forgot to include them).
    no_imports_block = theorem_code.replace("sorry", "norm_num")

    return [
        {
            "label": "correct_with_think_tags",
            "raw_output": (
                "<think>\n"
                "The theorem looks like it can be solved by norm_num.\n"
                "Let me write the complete proof.\n"
                "</think>\n"
                "```lean4\n"
                + correct_block
                + "\n```"
            ),
            "expected_non_sorry": True,
        },
        {
            "label": "block_without_imports",
            "raw_output": (
                "<think>\nI'll write the proof tactics.\n</think>\n"
                "```lean4\n"
                + no_imports_block
                + "\n```"
            ),
            "expected_non_sorry": True,
        },
        {
            "label": "incomplete_think_no_close",
            "raw_output": (
                "<think>\nI'm thinking about this problem and..."
                # No </think> — simulates generation being cut off
            ),
            "expected_non_sorry": False,  # should parse to "sorry"
        },
        {
            "label": "multiple_blocks_takes_last",
            "raw_output": (
                "<think>\n"
                "First attempt:\n"
                "```lean4\n"
                f"{header}\n{theorem_code}\n"  # has sorry — intermediate block
                "```\n"
                "That has sorry, let me fix it.\n"
                "</think>\n"
                "```lean4\n"
                + correct_block
                + "\n```"
            ),
            "expected_non_sorry": True,
        },
    ]


def run_test():
    cfg = EvalConfig(n_problems=1, pass_k=4)

    print("=" * 60)
    print("qwen_multiturn end-to-end pipeline smoke test")
    print(f"  dataset: {cfg.dataset_name}  split: {cfg.dataset_split}")
    print(f"  kimina:  {cfg.kimina_url}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load one real problem
    # ------------------------------------------------------------------
    print("\n[1/3] Loading 1 problem from dataset...")
    problems = load_problems(cfg)
    problem = problems[0]
    print(f"  problem_id:  {problem['problem_id']}")
    print(f"  formal_stmt: {problem['formal_statement'][:80]}...")
    print(f"  header:      {problem['header'][:60]}...")

    # ------------------------------------------------------------------
    # 2. Parse each simulated raw output
    # ------------------------------------------------------------------
    print("\n[2/3] Parsing 4 simulated model outputs...")
    simulated = make_simulated_outputs(problem["header"], problem["formal_statement"])

    parsed_attempts = []
    for i, sim in enumerate(simulated):
        extracted = extract_full_lean_block(sim["raw_output"])
        full_code = create_full_lean_code(
            theorem_code=problem["formal_statement"],
            extracted_block=extracted,
            default_header=cfg.default_header,
        )
        is_sorry = extracted == "sorry"
        expected_non_sorry = sim["expected_non_sorry"]
        parse_ok = is_sorry != expected_non_sorry  # XOR: passes if expectation matches

        print(f"\n  Attempt {i} [{sim['label']}]")
        print(f"    extracted_block (first 80 chars): {extracted[:80].replace(chr(10), ' ')!r}")
        print(f"    is_sorry: {is_sorry}  |  expected_non_sorry: {expected_non_sorry}  |  parse_ok: {parse_ok}")
        if not parse_ok:
            print(f"    !! PARSE MISMATCH for {sim['label']}")

        parsed_attempts.append({
            "attempt": i,
            "label": sim["label"],
            "prompt": "(simulated)",
            "raw_output": sim["raw_output"],
            "extracted_block": extracted,
            "full_code": full_code,
            "num_tokens": len(sim["raw_output"].split()),
            "parse_ok": parse_ok,
        })

    # ------------------------------------------------------------------
    # 3. Verify each attempt with Kimina
    # ------------------------------------------------------------------
    print("\n[3/3] Verifying with Kimina...")
    attempt_logs = []
    for att in parsed_attempts:
        print(f"\n  Attempt {att['attempt']} [{att['label']}]")
        result = verify(att["full_code"], kimina_url=cfg.kimina_url, timeout=cfg.verify_timeout_s)
        att["verification"] = result
        att["success"] = result.get("success", False) and result.get("complete", False) and not result.get("has_sorry", True)
        print(f"    success:  {result.get('success')}")
        print(f"    complete: {result.get('complete')}")
        print(f"    has_sorry:{result.get('has_sorry')}")
        if result.get("errors"):
            print(f"    errors:   {result['errors'][0][:120]}")
        if result.get("is_server_error"):
            print(f"    !! SERVER ERROR — is Kimina Docker running at {cfg.kimina_url}?")
        attempt_logs.append(att)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    print("\n[4/4] Saving results...")
    problem_log = build_problem_log(problem, attempt_logs, cfg)
    run_dir = make_run_dir(cfg)
    save_results(run_dir, cfg, [problem_log])

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    parse_failures = [a for a in attempt_logs if not a.get("parse_ok", True)]
    server_errors  = [a for a in attempt_logs if a["verification"].get("is_server_error")]

    print(f"  Parse checks:  {len(attempt_logs) - len(parse_failures)}/{len(attempt_logs)} passed")
    print(f"  Server errors: {len(server_errors)}")
    print(f"  Verified OK:   {sum(1 for a in attempt_logs if a['success'])}/{len(attempt_logs)} succeeded")
    print(f"  Results saved: {run_dir}")

    if parse_failures:
        print("\n!! Parse failures:")
        for a in parse_failures:
            print(f"   - {a['label']}")
        sys.exit(1)

    if server_errors:
        print("\n!! Kimina server errors — check Docker is running.")
        sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    run_test()
