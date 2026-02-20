#!/usr/bin/env python3
"""
Verify a single Lean proof using the local Lean REPL.

Usage:
    python verify_single_proof.py --code "LEAN_CODE_STRING"
    python verify_single_proof.py --file path/to/logs.json --iteration 3
    python verify_single_proof.py --file path/to/logs.json  # uses last iteration
"""

import json
import argparse
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from lean_compiler.repl_scheduler import scheduler


def extract_code_from_logs(logs_path: str, iteration: int = None) -> str:
    """Extract full_code from a logs.json file at a specific iteration."""
    with open(logs_path, 'r') as f:
        logs = json.load(f)
    
    iteration_logs = logs.get("iteration_logs", [])
    if not iteration_logs:
        raise ValueError("No iteration_logs found in the logs file")
    
    if iteration is None:
        iteration = len(iteration_logs)
    
    for log in iteration_logs:
        if log.get("iteration") == iteration:
            full_code = log.get("full_code")
            if full_code:
                return full_code
            raise ValueError(f"No full_code found at iteration {iteration}")
    
    raise ValueError(f"Iteration {iteration} not found. Available: {[l['iteration'] for l in iteration_logs]}")


def verify_lean_code(code: str, num_workers: int = 1) -> dict:
    """Verify Lean code using the local REPL scheduler."""
    proofs = [{"name": "proof_to_verify", "code": code}]
    results = scheduler(proofs, num_workers=num_workers)
    
    if results:
        return results[0]
    return {"error": "No results returned from scheduler"}


def main():
    parser = argparse.ArgumentParser(description="Verify Lean proof using local REPL")
    parser.add_argument("--code", type=str, help="Lean code string to verify")
    parser.add_argument("--file", type=str, help="Path to logs.json file")
    parser.add_argument("--iteration", type=int, help="Iteration number to extract (default: last)")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (default: 1)")
    parser.add_argument("--show-code", action="store_true", help="Print the code being verified")
    
    args = parser.parse_args()
    
    if not args.code and not args.file:
        parser.error("Either --code or --file must be provided")
    
    if args.code:
        code = args.code
    else:
        code = extract_code_from_logs(args.file, args.iteration)
    
    if args.show_code:
        print("=" * 60)
        print("CODE TO VERIFY:")
        print("=" * 60)
        print(code)
        print("=" * 60)
        print()
    
    print("Verifying Lean proof...")
    result = verify_lean_code(code, num_workers=args.workers)
    
    print("\n" + "=" * 60)
    print("VERIFICATION RESULT:")
    print("=" * 60)
    
    compilation = result.get("compilation_result", {})
    
    passed = compilation.get("pass", False)
    complete = compilation.get("complete", False)
    errors = compilation.get("errors", [])
    sorries = compilation.get("sorries", [])
    warnings = compilation.get("warnings", [])
    system_errors = compilation.get("system_errors")
    
    if complete:
        print("✓ COMPLETE - Proof verified successfully!")
    elif passed:
        print("○ COMPILES - Code compiles but has sorry or warnings")
    else:
        print("✗ FAILED - Proof has errors")
    
    print(f"\nPass: {passed}")
    print(f"Complete: {complete}")
    print(f"Verify time: {result.get('verify_time', 'N/A')}s")
    
    if system_errors:
        print(f"\nSystem Errors: {system_errors}")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - {err.get('data', err)}")
    
    if sorries:
        print(f"\nSorries ({len(sorries)}):")
        for sorry in sorries:
            print(f"  - Line {sorry.get('pos', {}).get('line', '?')}: {sorry.get('goal', 'N/A')[:100]}")
    
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for warn in warnings[:5]:
            print(f"  - {warn.get('data', warn)[:100]}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")
    
    print("\n" + "=" * 60)
    
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
