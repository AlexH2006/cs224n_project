"""
Verify generated Lean proofs using the Kimina Lean Server.

This script:
1. Loads evaluation results from minif2f_qwen3_8b_eval.json
2. Loads original problems from MiniF2F dataset
3. Constructs full Lean code with generated proofs
4. Verifies each proof using the Kimina server
5. Updates the results with verification status

Prerequisites:
    - Kimina server running locally (docker compose up)
    - pip install kimina-client

Usage:
    python verify_proofs_kimina.py [--server-url URL] [--input FILE] [--output FILE]
"""

import json
import re
import time
import argparse
from pathlib import Path
from typing import Optional


def load_minif2f_problems(dataset_path: Path, n: int = 10) -> dict:
    """Load MiniF2F problems into a dict keyed by problem_id."""
    problems = {}
    with open(dataset_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            problem = json.loads(line)
            problems[problem["problem_id"]] = problem
    return problems


def extract_lean_tactics(model_output: str) -> str:
    """
    Extract Lean tactics from model output.
    
    The model may output:
    - Raw tactics
    - Tactics wrapped in <think>...</think> reasoning
    - Tactics in markdown code blocks
    """
    output = model_output.strip()
    
    # Remove <think>...</think> blocks (chain-of-thought reasoning)
    # Keep only content after </think> if present
    if "</think>" in output:
        parts = output.split("</think>")
        output = parts[-1].strip()
    
    # If still has <think> tag at start, it's incomplete reasoning - use sorry
    if output.startswith("<think>"):
        return "sorry"
    
    # Remove markdown code blocks
    if "```" in output:
        # Try to extract content from code blocks
        code_block_pattern = r"```(?:lean4?|lean)?\n?(.*?)```"
        matches = re.findall(code_block_pattern, output, re.DOTALL)
        if matches:
            output = matches[0].strip()
        else:
            # Remove backticks
            output = output.replace("```lean4", "").replace("```lean", "").replace("```", "")
    
    # Clean up
    output = output.strip()
    
    # Remove "by" prefix if present (we add it back when constructing full code)
    if output.lower().startswith("by"):
        output = output[2:].strip()
    
    # If empty or just whitespace, return sorry
    if not output or output.isspace():
        return "sorry"
    
    # If it looks like reasoning text rather than tactics, return sorry
    if output.startswith("Okay") or output.startswith("Let me") or output.startswith("First"):
        return "sorry"
    
    return output


def construct_full_lean_code(problem: dict, tactics: str) -> str:
    """
    Construct full Lean 4 code by replacing 'sorry' with generated tactics.
    """
    lean4_code = problem["lean4_code"]
    
    # Clean up tactics - ensure proper indentation
    tactics_lines = tactics.split("\n")
    indented_tactics = "\n  ".join(line for line in tactics_lines if line.strip())
    
    # Replace "by sorry" with the generated proof
    if ":= by sorry" in lean4_code:
        full_code = lean4_code.replace(":= by sorry", f":= by\n  {indented_tactics}")
    elif "by sorry" in lean4_code:
        full_code = lean4_code.replace("by sorry", f"by\n  {indented_tactics}")
    elif "sorry" in lean4_code:
        full_code = lean4_code.replace("sorry", indented_tactics)
    else:
        # Fallback: append tactics
        full_code = lean4_code + f"\n  {indented_tactics}"
    
    return full_code


def verify_with_kimina(
    codes: list[dict],
    server_url: str = "http://localhost:8000",
    batch_size: int = 5,
    max_workers: int = 4,
) -> list[dict]:
    """
    Verify Lean proofs using Kimina server.
    
    Args:
        codes: List of dicts with 'problem_id' and 'full_code' keys
        server_url: Kimina server URL
        batch_size: Number of proofs to send per batch
        max_workers: Number of concurrent workers
    
    Returns:
        List of verification results
    """
    try:
        from kimina_client import KiminaClient
        from kimina_client.models import SnippetStatus
    except ImportError:
        print("ERROR: kimina_client not installed. Install with: pip install kimina-client")
        return [{"pass": False, "complete": False, "error": "kimina_client not installed"} 
                for _ in codes]
    
    client = KiminaClient(server_url, http_timeout=120)
    results = []
    
    print(f"\nVerifying {len(codes)} proofs with Kimina server at {server_url}...")
    
    for i, item in enumerate(codes):
        problem_id = item["problem_id"]
        full_code = item["full_code"]
        
        print(f"  [{i+1}/{len(codes)}] Verifying {problem_id}...", end=" ", flush=True)
        
        try:
            start_time = time.time()
            check_result = client.check(full_code, timeout=60, show_progress=False)
            elapsed = time.time() - start_time
            
            # Analyze the result
            if check_result.results:
                repl_response = check_result.results[0]
                analysis = repl_response.analyze()
                
                # SnippetStatus: valid, sorry, lean_error, repl_error, timeout_error, server_error
                is_valid = analysis.status == SnippetStatus.valid
                has_sorry = analysis.status == SnippetStatus.sorry
                is_error = analysis.status in (
                    SnippetStatus.lean_error, 
                    SnippetStatus.repl_error,
                    SnippetStatus.timeout_error,
                    SnippetStatus.server_error
                )
                
                verification = {
                    "pass": is_valid or has_sorry,  # Pass if code compiles (no errors)
                    "complete": is_valid,  # Complete if valid (no sorry, no errors)
                    "has_sorry": has_sorry,
                    "status": str(analysis.status.value),
                    "verify_time": round(elapsed, 2),
                    "server_time": analysis.time,
                }
                
                # Get error details from response
                if repl_response.response:
                    resp = repl_response.response
                    if hasattr(resp, 'messages') and resp.messages:
                        errors = [m.data for m in resp.messages if m.severity == 'error']
                        warnings = [m.data for m in resp.messages if m.severity == 'warning']
                        if errors:
                            verification["errors"] = errors[:3]
                        if warnings:
                            verification["warnings_count"] = len(warnings)
                    if hasattr(resp, 'sorries') and resp.sorries:
                        verification["sorries_count"] = len(resp.sorries)
            else:
                verification = {
                    "pass": False,
                    "complete": False,
                    "error": "No results returned",
                    "verify_time": round(elapsed, 2),
                }
            
            if verification.get("complete"):
                status = "✓ COMPLETE"
            elif verification.get("pass"):
                status = "○ COMPILES (has sorry)"
            else:
                status = f"✗ FAIL ({verification.get('status', 'unknown')})"
            print(f"{status} ({elapsed:.1f}s)")
            
            if verification.get("errors"):
                error_msg = str(verification["errors"][0])[:80]
                print(f"       Error: {error_msg}...")
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            verification = {
                "pass": False,
                "complete": False,
                "error": str(e),
            }
        
        results.append(verification)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify Lean proofs with Kimina server")
    parser.add_argument(
        "--server-url", 
        type=str, 
        default="http://localhost:8000",
        help="Kimina server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="results/minif2f_qwen3_8b_eval.json",
        help="Input evaluation results file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/minif2f_qwen3_8b_verified.json",
        help="Output file with verification results"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="Goedel-Prover-V2/dataset/minif2f.jsonl",
        help="MiniF2F dataset path"
    )
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent
    input_path = base_dir / args.input
    output_path = base_dir / args.output
    dataset_path = base_dir / args.dataset
    
    # Load evaluation results
    print(f"Loading evaluation results from {input_path}...")
    with open(input_path, "r") as f:
        eval_data = json.load(f)
    
    # Load original problems
    n_examples = eval_data.get("n_examples", 10)
    print(f"Loading {n_examples} problems from {dataset_path}...")
    problems = load_minif2f_problems(dataset_path, n_examples)
    
    # Prepare codes for verification
    codes_to_verify = []
    for result in eval_data["results"]:
        problem_id = result["problem_id"]
        generated_proof = result.get("generated_proof", "sorry")
        
        # Get original problem
        if problem_id not in problems:
            print(f"  Warning: Problem {problem_id} not found in dataset")
            continue
        
        problem = problems[problem_id]
        
        # Extract actual Lean tactics from model output
        tactics = extract_lean_tactics(generated_proof)
        
        # Construct full code
        full_code = construct_full_lean_code(problem, tactics)
        
        codes_to_verify.append({
            "problem_id": problem_id,
            "name": result.get("name", problem_id),
            "tactics": tactics,
            "full_code": full_code,
        })
        
        print(f"  {problem_id}: extracted tactics = '{tactics[:50]}...' " 
              if len(tactics) > 50 else f"  {problem_id}: extracted tactics = '{tactics}'")
    
    # Verify with Kimina
    verification_results = verify_with_kimina(codes_to_verify, args.server_url)
    
    # Update results
    for i, result in enumerate(eval_data["results"]):
        if i < len(verification_results):
            result["verification"] = verification_results[i]
            result["extracted_tactics"] = codes_to_verify[i]["tactics"]
            result["full_code"] = codes_to_verify[i]["full_code"]
    
    # Compute metrics
    passed = sum(1 for r in eval_data["results"] if r.get("verification", {}).get("pass", False))
    complete = sum(1 for r in eval_data["results"] if r.get("verification", {}).get("complete", False))
    total = len(eval_data["results"])
    
    eval_data["passed"] = passed
    eval_data["complete"] = complete
    
    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Total examples: {total}")
    print(f"Passed (no errors): {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"Complete (no sorry): {complete}/{total} ({100*complete/total:.1f}%)")
    
    print("\nDetailed Results:")
    for i, result in enumerate(eval_data["results"]):
        v = result.get("verification", {})
        if v.get("complete"):
            status = "✓ COMPLETE"
        elif v.get("pass"):
            status = "○ COMPILES (has sorry)"
        elif v.get("errors"):
            status = f"✗ ERROR: {str(v.get('errors')[0])[:40]}"
        else:
            status = f"✗ FAIL ({v.get('status', 'unknown')})"
        print(f"  {i+1}. {result['name']}: {status}")
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(eval_data, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
