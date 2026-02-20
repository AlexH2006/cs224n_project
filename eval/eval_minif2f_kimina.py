"""
Evaluate Kimina Prover 1.7B on MiniF2F examples.
- vLLM inference runs on Modal (cloud GPU)
- Lean proof verification via Kimina server

Usage:
    python eval_minif2f_kimina.py -n 5 --kimina-url http://localhost:80
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional
from datetime import datetime

import modal

# Paths
CURRENT_DIR = Path(__file__).parent
DATASET_PATH = CURRENT_DIR / "Goedel-Prover-V2" / "dataset" / "minif2f.jsonl"
RESULTS_DIR = CURRENT_DIR / "results"

# Modal app setup
app = modal.App("kimina-prover-minif2f-eval")

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

MODEL_NAME = "AI-MO/Kimina-Prover-Preview-Distill-1.5B"
MODEL_DIR = "/model"

# Volume to cache model weights
model_volume = modal.Volume.from_name("kimina-prover-model-cache", create_if_missing=True)


@app.cls(
    image=vllm_image,
    gpu="A10G",  # Smaller GPU sufficient for 1.5B model
    timeout=1800,
    volumes={MODEL_DIR: model_volume},
)
class KiminaInference:
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


def create_prompt(example: dict) -> str:
    """Create a prompt for Kimina Prover.
    
    Kimina Prover is specifically trained for Lean 4 theorem proving,
    so we use a straightforward prompt format.
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
            if ":= by" in line or line.strip().endswith(":= by"):
                break
    
    theorem_stmt = "\n".join(theorem_lines)
    
    # Kimina Prover prompt format - straightforward theorem proving
    prompt = f"""Complete the following Lean 4 theorem proof. Provide ONLY the tactic proof (the part after `:= by`).

{informal}

```lean4
{theorem_stmt}
```

Provide the tactics to complete this proof:"""
    
    return prompt


def extract_proof(output: str) -> str:
    """Extract Lean 4 tactics from model output."""
    text = output.strip()
    
    # Remove any thinking tags if present
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + 8:].strip()
        else:
            return "sorry"
    
    # Try to extract from code blocks
    code_patterns = [
        r"```lean4?\s*([\s\S]*?)```",
        r"```\s*([\s\S]*?)```",
    ]
    
    for pattern in code_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            text = matches[0].strip()
            break
    
    # If the model repeated the theorem signature, extract only the tactics after := by
    if ":= by" in text:
        parts = text.split(":= by")
        if len(parts) > 1:
            # Take the last part (the actual tactics)
            text = parts[-1].strip()
    
    # Clean up the extracted text
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines, comments, and non-tactic content
        if not stripped:
            continue
        if stripped.startswith("--"):
            continue
        if stripped.startswith("import") or stripped.startswith("open"):
            continue
        if stripped.startswith("theorem") or stripped.startswith("lemma"):
            continue
        if stripped.startswith("```"):
            continue
        # Remove leading "by" if present
        if stripped.lower() == "by":
            continue
        # Skip lines that look like hypothesis declarations (part of theorem signature)
        if re.match(r"^\(h\d*\s*:", stripped):
            continue
        # Skip lines that look like type annotations
        if re.match(r"^:\s*\w+", stripped):
            continue
        # Skip "tactics" as a standalone word (model artifact)
        if stripped.lower() == "tactics":
            continue
        clean_lines.append(line)
    
    result = "\n".join(clean_lines).strip()
    
    if not result:
        return "sorry"
    
    return result


def create_full_code(example: dict, tactics: str) -> str:
    """Create full Lean code by inserting tactics into the theorem."""
    lean4_code = example["lean4_code"]
    
    # Find where to insert tactics (after := by)
    if ":= by" in lean4_code:
        parts = lean4_code.split(":= by", 1)
        if len(parts) == 2:
            # Get the part before := by and add our tactics
            before = parts[0]
            # Indent tactics properly
            indented_tactics = "\n".join("  " + line if line.strip() else line 
                                         for line in tactics.split("\n"))
            full_code = f"{before}:= by\n{indented_tactics}"
            return full_code
    
    return lean4_code.replace("sorry", tactics)


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


def run_evaluation(
    n_examples: int = 5,
    verify: bool = True,
    kimina_url: Optional[str] = None,
):
    """Run the full evaluation pipeline."""
    
    print("\n" + "="*60)
    print("MiniF2F Evaluation with Kimina Prover 1.5B")
    print("="*60)
    
    # Load dataset
    print(f"\n[1/4] Loading first {n_examples} examples from MiniF2F...")
    examples = load_minif2f_dataset(n_examples)
    print(f"      Loaded {len(examples)} examples")
    
    # Create prompts
    print(f"\n[2/4] Creating prompts...")
    prompts = [create_prompt(ex) for ex in examples]
    
    # Run inference on Modal
    print(f"\n[3/4] Running vLLM inference on Modal...")
    print(f"      (This may take a few minutes on first run to download the model)")
    
    start_time = time.time()
    
    with app.run():
        inference = KiminaInference()
        outputs = inference.generate_proofs.remote(prompts)
    
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
    if verify and kimina_url:
        print(f"\n[4/4] Verifying proofs...")
        print(f"      Using Kimina server at {kimina_url}")
        
        verification_results = verify_with_kimina(
            [{"full_code": r["full_code"]} for r in results],
            server_url=kimina_url
        )
        
        for result, v_result in zip(results, verification_results):
            result["verification"] = v_result
    else:
        print(f"\n[4/4] Skipping verification (no Kimina URL provided)")
        for result in results:
            result["verification"] = {"pass": None, "complete": None, "skipped": True}
    
    # Compute metrics
    if verify and kimina_url:
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
    
    if verify and kimina_url:
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
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save solutions
    solutions_path = RESULTS_DIR / "minif2f_kimina_prover_solutions.json"
    solutions_data = {
        "_metadata": {
            "description": "Generated Lean 4 proof solutions from Kimina Prover 1.5B on MiniF2F dataset",
            "date": timestamp,
            "dataset": "MiniF2F (Lean 4)",
            "model": MODEL_NAME,
            "notes": "Kimina Prover is specifically trained for Lean 4 theorem proving"
        },
        "model": MODEL_NAME,
        "n_examples": n_examples,
        "inference_time_seconds": inference_time,
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
    
    # Save verification log
    log_path = RESULTS_DIR / "minif2f_kimina_prover_verification_log.json"
    log_data = {
        "_metadata": {
            "description": "Verification results for Kimina Prover 1.5B generated proofs on MiniF2F",
            "date": timestamp,
            "dataset": "MiniF2F (Lean 4)",
            "model": MODEL_NAME,
            "verifier": "Kimina Lean Server",
            "notes": "pass=True means code compiles, complete=True means no sorry tactics used"
        },
        "model": MODEL_NAME,
        "n_examples": n_examples,
        "inference_time_seconds": inference_time,
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
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Kimina Prover 1.5B on MiniF2F")
    parser.add_argument("-n", "--n-examples", type=int, default=5, help="Number of examples")
    parser.add_argument("--no-verify", action="store_true", help="Skip proof verification")
    parser.add_argument("--kimina-url", type=str, default=None, help="Kimina server URL")
    
    args = parser.parse_args()
    
    run_evaluation(
        n_examples=args.n_examples,
        verify=not args.no_verify,
        kimina_url=args.kimina_url,
    )
