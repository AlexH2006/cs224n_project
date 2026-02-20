"""
Evaluate Qwen3-8B on first 10 examples of MiniF2F.
- vLLM inference runs on Modal (cloud GPU)
- Lean proof verification via Kimina server (if available) or local REPL

Usage:
    # Run with Modal for inference, Kimina for verification
    python eval_minif2f_qwen.py
    
    # Or run as Modal app
    modal run eval_minif2f_qwen.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import modal

# Paths
CURRENT_DIR = Path(__file__).parent
DATASET_PATH = CURRENT_DIR / "Goedel-Prover-V2" / "dataset" / "minif2f.jsonl"
RESULTS_DIR = CURRENT_DIR / "results"

# Modal app setup
app = modal.App("qwen3-8b-minif2f-eval")

# Modal image with vLLM
vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm>=0.6.0",
        "torch",
        "transformers",
        "huggingface_hub",
    )
)

MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_DIR = "/model"

# Volume to cache model weights
model_volume = modal.Volume.from_name("qwen3-8b-model-cache", create_if_missing=True)


@app.cls(
    image=vllm_image,
    gpu="A100",
    timeout=1800,
    volumes={MODEL_DIR: model_volume},
)
class QwenInference:
    """vLLM inference class running on Modal."""
    
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams
        
        print(f"Loading model {MODEL_NAME}...")
        self.llm = LLM(
            model=MODEL_NAME,
            download_dir=MODEL_DIR,
            trust_remote_code=True,
            max_model_len=4096,
            tensor_parallel_size=1,
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=2048,
        )
        print("Model loaded successfully!")
    
    @modal.method()
    def generate_proofs(self, prompts: list[str]) -> list[str]:
        """Generate proofs for a batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]


def load_minif2f_dataset(n_examples: int = 10) -> list[dict]:
    """Load first n examples from MiniF2F dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    examples = []
    with open(DATASET_PATH, "r") as f:
        for i, line in enumerate(f):
            if i >= n_examples:
                break
            examples.append(json.loads(line))
    
    return examples


def create_prompt(example: dict, enable_thinking: bool = False) -> str:
    """Create a prompt for Lean 4 proof generation.
    
    Args:
        example: MiniF2F problem dict
        enable_thinking: If False, disable Qwen3's thinking mode with /no_think
    """
    lean4_code = example["lean4_code"]
    informal = example.get("informal_prefix", "")
    
    # Extract just the theorem statement (after imports)
    lines = lean4_code.split("\n")
    theorem_lines = []
    in_theorem = False
    for line in lines:
        if line.strip().startswith("theorem") or line.strip().startswith("lemma"):
            in_theorem = True
        if in_theorem:
            theorem_lines.append(line)
    theorem_stmt = "\n".join(theorem_lines) if theorem_lines else lean4_code
    
    if enable_thinking:
        # Allow thinking mode (default Qwen3 behavior)
        prompt = f"""<|im_start|>user
Complete the following Lean 4 proof. Replace the `sorry` with valid proof tactics.

{informal}

```lean4
{lean4_code}
```

Provide ONLY the proof tactics that should replace `sorry`. Do not include imports, theorem statement, or markdown formatting.
<|im_end|>
<|im_start|>assistant
"""
    else:
        # Disable thinking mode with /no_think suffix
        # Explicitly instruct Lean 4 syntax
        prompt = f"""<|im_start|>user
You are a Lean 4 theorem prover. Complete the proof by providing ONLY the tactics to replace `sorry`.

IMPORTANT: Use Lean 4 syntax, NOT Lean 3:
- Separate tactics with newlines or semicolons (;), NOT commas
- Use `simp [h]` not `simp [h],`
- Use `rw [h₁]; rw [h₂]` or put each tactic on a new line

Problem:
{informal}

Theorem (Lean 4):
```lean4
{theorem_stmt}
```

Provide ONLY the Lean 4 proof tactics. Each tactic on a new line. No explanations. /no_think
<|im_end|>
<|im_start|>assistant
"""
    return prompt


def extract_proof(output: str) -> str:
    """Extract proof tactics from model output."""
    import re
    
    output = output.strip()
    
    # Handle Qwen3 thinking mode - extract content after </think> if present
    if "</think>" in output:
        output = output.split("</think>")[-1].strip()
    
    # If output is just a <think> block without closing, it's incomplete
    if output.startswith("<think>"):
        return "sorry"
    
    # Remove markdown code blocks if present
    if "```" in output:
        # Find content between code blocks
        code_pattern = r"```(?:lean4?|lean)?\n?(.*?)```"
        matches = re.findall(code_pattern, output, re.DOTALL)
        if matches:
            output = matches[0].strip()
        else:
            # Fallback: split by backticks
            parts = output.split("```")
            for part in parts:
                # Skip language tags
                if part.startswith("lean") or part.startswith("lean4"):
                    part = "\n".join(part.split("\n")[1:])
                part = part.strip()
                if part and not part.startswith("import") and "theorem" not in part[:50]:
                    output = part
                    break
    
    # Remove any leading "by" if present (we'll add it back)
    output = output.strip()
    if output.lower().startswith("by"):
        output = output[2:].strip()
    
    # === Lean 3 to Lean 4 syntax conversion ===
    # Convert comma-separated tactics to newline-separated (Lean 3 -> Lean 4)
    # Pattern: tactic, tactic -> tactic\ntactic
    # But be careful not to break things like `rw [a, b]` or `simp [a, b]`
    
    # First, handle comma-separated tactics that are NOT inside brackets
    # Split by commas that are followed by a tactic keyword or identifier
    lines = []
    current_line = ""
    bracket_depth = 0
    
    i = 0
    while i < len(output):
        char = output[i]
        
        if char in '([{':
            bracket_depth += 1
            current_line += char
        elif char in ')]}':
            bracket_depth -= 1
            current_line += char
        elif char == ',' and bracket_depth == 0:
            # This is a tactic separator, convert to newline
            current_line = current_line.strip()
            if current_line:
                lines.append(current_line)
            current_line = ""
        elif char == '\n':
            current_line = current_line.strip()
            if current_line:
                lines.append(current_line)
            current_line = ""
        else:
            current_line += char
        i += 1
    
    # Don't forget the last line
    current_line = current_line.strip()
    if current_line:
        lines.append(current_line)
    
    # Clean up each line
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines at start
        if not cleaned_lines and not line:
            continue
        # Skip lines that look like theorem declarations or imports
        if line.startswith("theorem") or line.startswith("import"):
            continue
        # Skip explanation text
        if line.startswith("This") or line.startswith("The "):
            continue
        # Remove trailing commas (Lean 3 style)
        if line.endswith(','):
            line = line[:-1].strip()
        if line:
            cleaned_lines.append(line)
    
    output = "\n".join(cleaned_lines).strip()
    
    # Final cleanup - remove trailing explanation
    if "\n\n" in output:
        # Keep only first block (likely the tactics)
        output = output.split("\n\n")[0].strip()
    
    return output if output else "sorry"


def create_full_code(example: dict, proof: str) -> str:
    """Create full Lean 4 code with generated proof."""
    lean4_code = example["lean4_code"]
    
    # Ensure proof is properly indented
    proof_lines = proof.split("\n")
    indented_proof = "\n  ".join(proof_lines)
    
    # Replace sorry with proof
    if ":= by sorry" in lean4_code:
        full_code = lean4_code.replace(":= by sorry", f":= by\n  {indented_proof}")
    elif ":= by\n  sorry" in lean4_code:
        full_code = lean4_code.replace("sorry", indented_proof)
    else:
        # Fallback
        full_code = lean4_code.replace("sorry", indented_proof)
    
    return full_code


def verify_with_kimina(codes: list[dict], server_url: str = "http://localhost:8000") -> list[dict]:
    """Verify proofs using Kimina server."""
    try:
        from kimina_client import KiminaClient
    except ImportError:
        print("kimina_client not installed. Install with: pip install kimina-client")
        return [{"pass": False, "complete": False, "error": "kimina_client not installed"} for _ in codes]
    
    client = KiminaClient(server_url)
    results = []
    
    for item in codes:
        try:
            check_response = client.check(item["full_code"])
            
            has_error = False
            has_sorry = False
            errors = []
            sorries = []
            
            for repl_resp in check_response.results:
                resp_dict = repl_resp.response
                if isinstance(resp_dict, dict):
                    messages = resp_dict.get("messages", [])
                    for msg in messages:
                        if msg.get("severity") == "error":
                            has_error = True
                            errors.append(msg.get("data", ""))
                    
                    if resp_dict.get("sorries"):
                        has_sorry = True
                        sorries.extend(resp_dict.get("sorries", []))
                elif hasattr(resp_dict, "model_dump"):
                    d = resp_dict.model_dump()
                    messages = d.get("messages", [])
                    for msg in messages:
                        if msg.get("severity") == "error":
                            has_error = True
                            errors.append(msg.get("data", ""))
                    
                    if d.get("sorries"):
                        has_sorry = True
                        sorries.extend(d.get("sorries", []))
            
            results.append({
                "pass": not has_error,
                "complete": not has_error and not has_sorry,
                "has_sorry": has_sorry,
                "errors": errors,
                "sorries": sorries,
            })
        except Exception as e:
            results.append({
                "pass": False,
                "complete": False,
                "error": str(e),
            })
    
    return results


def verify_with_local_repl(codes: list[dict], num_workers: int = 4) -> list[dict]:
    """Verify proofs using local Lean REPL scheduler."""
    sys.path.insert(0, str(CURRENT_DIR / "Goedel-Prover-V2" / "lean_compiler"))
    
    try:
        from repl_scheduler import scheduler
    except ImportError as e:
        print(f"Could not import repl_scheduler: {e}")
        print("Make sure you have the Lean REPL set up in Goedel-Prover-V2/")
        return [{"pass": False, "complete": False, "error": "repl_scheduler not available"} for _ in codes]
    
    # Prepare proofs for scheduler
    proofs = [{"name": item["problem_id"], "code": item["full_code"]} for item in codes]
    
    print(f"\nVerifying {len(proofs)} proofs with {num_workers} workers...")
    try:
        results = scheduler(proofs, num_workers=num_workers, timeout=120, imports="")
    except Exception as e:
        print(f"Verification failed: {e}")
        return [{"pass": False, "complete": False, "error": str(e)} for _ in codes]
    
    # Map results back
    results_by_name = {r["name"]: r["compilation_result"] for r in results}
    return [results_by_name.get(item["problem_id"], {"pass": False, "complete": False}) for item in codes]


def run_evaluation(
    n_examples: int = 10,
    verify: bool = True,
    kimina_url: Optional[str] = None,
    num_workers: int = 4,
    enable_thinking: bool = False,
):
    """
    Run the full evaluation pipeline.
    
    Args:
        n_examples: Number of MiniF2F examples to evaluate
        verify: Whether to verify proofs (requires Kimina server or local REPL)
        kimina_url: Kimina server URL (if None, uses local REPL)
        num_workers: Number of workers for local REPL verification
        enable_thinking: Whether to enable Qwen3's thinking mode (default: False)
    """
    print("="*60)
    print("MiniF2F Evaluation with Qwen3-8B")
    print("="*60)
    
    # Load dataset
    print(f"\n[1/4] Loading first {n_examples} examples from MiniF2F...")
    examples = load_minif2f_dataset(n_examples)
    print(f"      Loaded {len(examples)} examples")
    
    # Create prompts
    print("\n[2/4] Creating prompts...")
    print(f"      Thinking mode: {'enabled' if enable_thinking else 'disabled'}")
    prompts = [create_prompt(ex, enable_thinking=enable_thinking) for ex in examples]
    
    # Run inference on Modal
    print("\n[3/4] Running vLLM inference on Modal...")
    print("      (This may take a few minutes on first run to download the model)")
    
    start_time = time.time()
    with app.run():
        qwen = QwenInference()
        outputs = qwen.generate_proofs.remote(prompts)
    inference_time = time.time() - start_time
    
    print(f"      Generated {len(outputs)} outputs in {inference_time:.1f}s")
    
    # Process outputs
    results = []
    for i, (example, output) in enumerate(zip(examples, outputs)):
        proof = extract_proof(output)
        full_code = create_full_code(example, proof)
        
        results.append({
            "problem_id": example["problem_id"],
            "name": example["name"],
            "formal_statement": example.get("formal_statement", ""),
            "informal_prefix": example.get("informal_prefix", ""),
            "generated_proof": proof,
            "full_code": full_code,
            "model_output": output,
        })
        
        print(f"\n      [{i+1}] {example['name']}")
        proof_preview = proof[:100].replace("\n", " ")
        print(f"          Proof: {proof_preview}...")
    
    # Verify proofs
    if verify:
        print("\n[4/4] Verifying proofs...")
        
        if kimina_url:
            print(f"      Using Kimina server at {kimina_url}")
            verifications = verify_with_kimina(results, kimina_url)
        else:
            print("      Using local Lean REPL (requires Mathlib setup)")
            verifications = verify_with_local_repl(results, num_workers)
        
        # Merge verification results
        for result, verification in zip(results, verifications):
            result["verification"] = verification
    else:
        print("\n[4/4] Skipping verification (verify=False)")
        for result in results:
            result["verification"] = {"pass": None, "complete": None, "skipped": True}
    
    # Compute metrics
    if verify:
        passed = sum(1 for r in results if r["verification"].get("pass", False))
        complete = sum(1 for r in results if r["verification"].get("complete", False))
    else:
        passed = complete = 0
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Examples: {len(examples)}")
    print(f"Inference time: {inference_time:.1f}s")
    
    if verify:
        print(f"\nVerification Results:")
        print(f"  Passed (no errors): {passed}/{len(examples)} ({100*passed/len(examples):.1f}%)")
        print(f"  Complete (no sorry): {complete}/{len(examples)} ({100*complete/len(examples):.1f}%)")
        
        print("\nDetailed Results:")
        for i, result in enumerate(results):
            v = result["verification"]
            if v.get("complete"):
                status = "✓ COMPLETE"
            elif v.get("pass"):
                status = "○ PASSED (has sorry)"
            elif v.get("error"):
                status = f"✗ ERROR: {v['error'][:50]}"
            else:
                status = "✗ FAILED"
            print(f"  {i+1}. {result['name']}: {status}")
    
    # Save results to two separate files
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get current timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Save generated solutions (proofs only)
    solutions_path = RESULTS_DIR / "minif2f_qwen3_8b_solutions.json"
    solutions_data = {
        "_metadata": {
            "description": "Generated Lean 4 proof solutions from Qwen3-8B on MiniF2F dataset",
            "date": timestamp,
            "dataset": "MiniF2F (Lean 4)",
            "model": MODEL_NAME,
            "notes": "Solutions contain extracted tactics from model output, may not be valid proofs"
        },
        "model": MODEL_NAME,
        "n_examples": n_examples,
        "inference_time_seconds": inference_time,
        "thinking_mode": enable_thinking,
        "solutions": [
            {
                "problem_id": r["problem_id"],
                "name": r["name"],
                "formal_statement": r.get("formal_statement", ""),
                "informal_statement": r.get("informal_prefix", ""),
                "generated_proof": r["generated_proof"],
                "full_code": r.get("full_code", ""),
            }
            for r in results
        ],
    }
    
    with open(solutions_path, "w") as f:
        json.dump(solutions_data, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\nSolutions saved to: {solutions_path}")
    
    # 2. Save verification log (detailed results)
    log_path = RESULTS_DIR / "minif2f_qwen3_8b_verification_log.json"
    log_data = {
        "_metadata": {
            "description": "Verification results for Qwen3-8B generated proofs on MiniF2F",
            "date": timestamp,
            "dataset": "MiniF2F (Lean 4)",
            "model": MODEL_NAME,
            "verifier": "Kimina Lean Server",
            "notes": "pass=True means code compiles, complete=True means no sorry tactics used"
        },
        "model": MODEL_NAME,
        "n_examples": n_examples,
        "inference_time_seconds": inference_time,
        "thinking_mode": enable_thinking,
        "summary": {
            "total": len(examples),
            "passed": passed,
            "complete": complete,
            "pass_rate": f"{100*passed/len(examples):.1f}%" if examples else "0%",
            "complete_rate": f"{100*complete/len(examples):.1f}%" if examples else "0%",
        },
        "verification_results": [
            {
                "problem_id": r["problem_id"],
                "name": r["name"],
                "formal_statement": r.get("formal_statement", ""),
                "informal_statement": r.get("informal_prefix", ""),
                "extracted_tactics": r.get("generated_proof", ""),
                "verification": r["verification"],
                "full_code": r.get("full_code", ""),
            }
            for r in results
        ],
    }
    
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"Verification log saved to: {log_path}")
    
    # Also save combined results for backward compatibility
    combined_path = RESULTS_DIR / "minif2f_qwen3_8b_eval.json"
    combined_data = {
        "_metadata": {
            "description": "Combined evaluation results for Qwen3-8B on MiniF2F",
            "date": timestamp,
            "dataset": "MiniF2F (Lean 4)",
            "model": MODEL_NAME,
        },
        "model": MODEL_NAME,
        "n_examples": n_examples,
        "inference_time_seconds": inference_time,
        "passed": passed,
        "complete": complete,
        "results": [
            {
                "problem_id": r["problem_id"],
                "name": r["name"],
                "formal_statement": r.get("formal_statement", ""),
                "generated_proof": r["generated_proof"],
                "verification": r["verification"],
            }
            for r in results
        ],
    }
    
    with open(combined_path, "w") as f:
        json.dump(combined_data, f, indent=2, default=str, ensure_ascii=False)
    
    return results


@app.local_entrypoint()
def main():
    """Modal entry point."""
    # Try Kimina first, fall back to skipping verification
    try:
        from kimina_client import KiminaClient
        client = KiminaClient("http://localhost:8000")
        client.check("#check Nat")  # Quick test
        kimina_url = "http://localhost:8000"
        print("Kimina server detected!")
    except:
        kimina_url = None
        print("Kimina server not available, will skip verification")
    
    run_evaluation(
        n_examples=10,
        verify=(kimina_url is not None),
        kimina_url=kimina_url,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-8B on MiniF2F")
    parser.add_argument("-n", "--n-examples", type=int, default=10, help="Number of examples")
    parser.add_argument("--no-verify", action="store_true", help="Skip proof verification")
    parser.add_argument("--kimina-url", type=str, default=None, help="Kimina server URL")
    parser.add_argument("--workers", type=int, default=4, help="Workers for local REPL")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen3 thinking mode")
    
    args = parser.parse_args()
    
    run_evaluation(
        n_examples=args.n_examples,
        verify=not args.no_verify,
        kimina_url=args.kimina_url,
        num_workers=args.workers,
        enable_thinking=args.enable_thinking,
    )
