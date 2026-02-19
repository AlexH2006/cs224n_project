"""
Model-agnostic evaluation on MATH dataset.

Supports local inference or Modal cloud inference.

Usage:
    # Local inference
    python eval_compare_kimina_qwen.py --model Qwen/Qwen3-1.7B --n-examples 20
    
    # Modal inference
    python eval_compare_kimina_qwen.py --model Qwen/Qwen3-1.7B --n-examples 20 --modal
    
    # Compare two models
    python eval_compare_kimina_qwen.py --model Qwen/Qwen3-1.7B --model2 AI-MO/Kimina-Prover-RL-1.7B
"""

import json
import re
import gc
import time
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

CURRENT_DIR = Path(__file__).parent
OUTPUT_DIR = CURRENT_DIR / "eval_nl_MATH"


@dataclass
class ModelConfig:
    name: str
    model_id: str
    max_tokens: int
    temperature: float


def load_math_dataset(
    n_examples: int = 20, 
    split: str = "test",
    subset: str = "algebra",
) -> list[dict]:
    """Load MATH dataset from HuggingFace.
    
    Args:
        n_examples: Number of examples to load
        split: Dataset split (train/test)
        subset: MATH subset (algebra, counting_and_probability, geometry, 
                intermediate_algebra, number_theory, prealgebra, precalculus)
    """
    from datasets import load_dataset
    
    print(f"Loading MATH {split} split (subset: {subset})...")
    
    # Primary: EleutherAI/hendrycks_math (working source)
    try:
        dataset = load_dataset("EleutherAI/hendrycks_math", subset, split=split)
        print(f"Loaded from EleutherAI/hendrycks_math/{subset}")
    except Exception as e:
        print(f"EleutherAI/hendrycks_math failed: {e}")
        # Fallback to GSM8K
        print("Falling back to GSM8K...")
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        examples = []
        for i, item in enumerate(dataset):
            if i >= n_examples:
                break
            examples.append({
                "problem": item["question"],
                "solution": item["answer"],
                "level": "",
                "type": "arithmetic",
            })
        return examples
    
    examples = []
    for i, item in enumerate(dataset):
        if i >= n_examples:
            break
        examples.append({
            "problem": item["problem"],
            "solution": item["solution"],
            "level": item.get("level", ""),
            "type": item.get("type", subset),
        })
    
    return examples


def extract_ground_truth(solution: str) -> str:
    """Extract the final answer from MATH solution (\\boxed{answer} format)."""
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(boxed_pattern, solution)
    if matches:
        return matches[-1].strip()
    return solution.strip()


def create_few_shot_prompt(problem: str) -> str:
    """Create a few-shot prompt matching MATH dataset format with \\boxed{{}} answers."""
    prompt = f"""Solve the following math problems. Show your work and put your final answer in \\boxed{{}}.

Problem: What is the positive difference between $120\\%$ of 30 and $130\\%$ of 20?
Solution: One hundred twenty percent of 30 is $120\\cdot30\\cdot\\frac{{1}}{{100}}=36$, and $130\\%$ of 20 is $130\\cdot 20\\cdot\\frac{{1}}{{100}}=26$. The difference between 36 and 26 is $\\boxed{{10}}$.

Problem: How many vertical asymptotes does the graph of $y=\\frac{{2}}{{x^2+x-6}}$ have?
Solution: The denominator of the rational function factors into $x^2+x-6=(x-2)(x+3)$. Since the numerator is always nonzero, there is a vertical asymptote whenever the denominator is $0$, which occurs for $x = 2$ and $x = -3$. Therefore, the graph has $\\boxed{{2}}$ vertical asymptotes.

Problem: {problem}
Solution:"""
    
    return prompt


def truncate_to_first_solution(output: str) -> str:
    """Keep only the model's first solution; drop repeated prompt/examples."""
    text = output.strip()
    # Stop at repetition of prompt format (next "Problem:") or at double newline + "Problem"
    stop = "\n\nProblem:"
    idx = text.find(stop)
    if idx != -1:
        return text[:idx].strip()
    return text


def extract_answer(output: str) -> str:
    """Extract the answer from model output. Uses first \\boxed{} (model's actual answer)."""
    text = output.strip()
    
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + 8:].strip()
    
    # Only consider the first solution (before model repeats prompt/examples)
    text = truncate_to_first_solution(text)
    
    # Primary: look for \boxed{} (MATH dataset standard format); use FIRST match
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    matches = re.findall(boxed_pattern, text)
    if matches:
        return matches[0].strip()
    
    # Fallback patterns for other formats
    answer_patterns = [
        r"[Aa]nswer:\s*\$?([\d,\.\-\/]+)",
        r"[Aa]nswer is[:\s]+\$?([\d,\.\-\/]+)",
        r"= \$?([\d,\.\-\/]+)\s*$",
        r"####\s*([\d,\.\-\/]+)",  # GSM8K format
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(",", "").strip()
    
    return ""


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    ans = answer.strip()
    ans = ans.replace("\\frac", "frac")
    ans = ans.replace("\\", "")
    ans = ans.replace(" ", "")
    ans = ans.lower()
    return ans


def check_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if pred_norm == gt_norm:
        return True
    
    try:
        pred_num = float(eval(predicted.replace("^", "**")))
        gt_num = float(eval(ground_truth.replace("^", "**")))
        return abs(pred_num - gt_num) < 0.01
    except:
        pass
    
    return False


class LocalInference:
    """Local inference using transformers."""
    
    def __init__(self, model_id: str, max_tokens: int = 1024, temperature: float = 0.0):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Loading {model_id} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded on {self.device}!")
    
    def generate(self, prompts: list[str]) -> list[str]:
        import torch
        
        outputs = []
        for i, prompt in enumerate(prompts):
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            output = self.tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs.append(output)
            
            if (i + 1) % 5 == 0:
                print(f"  Generated {i+1}/{len(prompts)}")
        
        return outputs
    
    def unload(self):
        """Unload model to free memory."""
        import torch
        
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("Model unloaded.")


class ModalInference:
    """Modal cloud inference with two-level caching:

    1. In-memory: @modal.enter() loads the model once per container; scaledown_window
       keeps the container warm so repeated calls reuse the same process.
    2. Disk (Volume): HF_HOME points at a mounted Modal Volume. First run downloads the
       model to the volume; later runs (new containers) find the weights already there
       and skip the download for much faster startup.
    """
    HF_CACHE_VOLUME_NAME = "math-eval-hf-cache"
    HF_CACHE_MOUNT_PATH = "/cache"

    def __init__(self, model_id: str, max_tokens: int = 1024, temperature: float = 0.0):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        print(f"Configured Modal inference for {model_id}")
    
    def generate(self, prompts: list[str]) -> list[str]:
        import modal
        
        # Persistent volume: first run downloads model here; later runs load from here (no re-download)
        volume = modal.Volume.from_name(
            self.HF_CACHE_VOLUME_NAME,
            create_if_missing=True,
        )
        
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install(
                "torch",
                "transformers>=4.40.0",
                "accelerate",
                "sentencepiece",
                "protobuf",
            )
        )
        
        app = modal.App("math-eval-inference")
        model_id = self.model_id
        max_tokens = self.max_tokens
        cache_path = self.HF_CACHE_MOUNT_PATH
        
        @app.cls(
            image=image,
            gpu="A10G",
            timeout=1200,
            scaledown_window=300,
            serialized=True,
            volumes={cache_path: volume},
        )
        class InferenceModel:
            @modal.enter()
            def load_model(self):
                import os
                # Use mounted volume as HF cache so model weights persist between container runs
                os.environ["HF_HOME"] = cache_path
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                print(f"Loading {model_id} on Modal (cache at {cache_path})...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                self.model.eval()
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                print(f"Model {model_id} loaded (weights cached on volume for next run).")
            
            @modal.method()
            def run(self, prompts: list[str], max_new_tokens: int) -> list[str]:
                import torch
                
                outputs = []
                for i, prompt in enumerate(prompts):
                    inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        generated = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    
                    output = self.tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    outputs.append(output)
                    
                    if (i + 1) % 5 == 0:
                        print(f"  Generated {i+1}/{len(prompts)}")
                
                return outputs
        
        with app.run():
            inference = InferenceModel()
            return inference.run.remote(prompts, max_tokens)
    
    def unload(self):
        pass


def get_model_short_name(model_id: str) -> str:
    """Get a short name for the model for file naming."""
    name = model_id.split("/")[-1]
    name = name.lower().replace("-", "_").replace(".", "_")
    return name


def evaluate_model(
    model_id: str,
    examples: list[dict],
    output_dir: Path,
    use_modal: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> dict:
    """Evaluate a model on MATH dataset."""
    model_name = model_id.split("/")[-1]
    short_name = get_model_short_name(model_id)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"Backend: {'Modal' if use_modal else 'Local'}")
    print(f"{'='*60}")
    
    if use_modal:
        engine = ModalInference(model_id, max_tokens, temperature)
    else:
        engine = LocalInference(model_id, max_tokens, temperature)
    
    print(f"Creating prompts...")
    prompts = [create_few_shot_prompt(ex["problem"]) for ex in examples]
    
    print(f"Running inference on {len(prompts)} examples...")
    start_time = time.time()
    outputs = engine.generate(prompts)
    inference_time = time.time() - start_time
    print(f"Completed in {inference_time:.1f}s")
    
    engine.unload()
    
    correct = 0
    results = []
    
    for i, (example, output) in enumerate(zip(examples, outputs)):
        ground_truth = extract_ground_truth(example["solution"])
        predicted = extract_answer(output)
        is_correct = check_answer(predicted, ground_truth)
        
        if is_correct:
            correct += 1
        
        results.append({
            "problem": example["problem"],
            "level": example.get("level", ""),
            "type": example.get("type", ""),
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "model_output": truncate_to_first_solution(output),
        })
    
    accuracy = correct / len(examples) if examples else 0
    print(f"\nAccuracy: {accuracy*100:.1f}% ({correct}/{len(examples)})")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f"{short_name}_results.json"
    results_data = {
        "model": model_name,
        "model_id": model_id,
        "backend": "modal" if use_modal else "local",
        "n_examples": len(examples),
        "accuracy": accuracy,
        "correct": correct,
        "inference_time": inference_time,
        "results": results,
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {results_path}")
    
    samples_path = output_dir / f"{short_name}_sample_solutions.json"
    samples = []
    for i, (example, output) in enumerate(zip(examples[:2], outputs[:2])):
        samples.append({
            "problem_index": i,
            "problem": example["problem"],
            "level": example.get("level", ""),
            "type": example.get("type", ""),
            "ground_truth_solution": example["solution"],
            "ground_truth_answer": extract_ground_truth(example["solution"]),
            "model_output": truncate_to_first_solution(output),
            "extracted_answer": extract_answer(output),
            "correct": results[i]["correct"],
        })
    
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Sample solutions saved to: {samples_path}")
    
    return {
        "model": model_name,
        "model_id": model_id,
        "backend": "modal" if use_modal else "local",
        "n_examples": len(examples),
        "inference_time": inference_time,
        "correct": correct,
        "accuracy": accuracy,
    }


def print_comparison(results1: dict, results2: dict):
    """Print comparison table for two models."""
    print("\n" + "="*70)
    print(f"COMPARISON: {results1['model']} vs {results2['model']}")
    print("="*70)
    print(f"Benchmark: MATH (Competition Mathematics)")
    print(f"Examples: {results1['n_examples']}")
    print()
    
    col1 = results1['model'][:20]
    col2 = results2['model'][:20]
    
    print(f"{'Metric':<25} {col1:>20} {col2:>20}")
    print("-"*70)
    print(f"{'Accuracy':<25} {results1['accuracy']*100:>19.1f}% {results2['accuracy']*100:>19.1f}%")
    print(f"{'Correct':<25} {results1['correct']:>20} {results2['correct']:>20}")
    print(f"{'Inference Time (s)':<25} {results1['inference_time']:>20.1f} {results2['inference_time']:>20.1f}")
    print(f"{'Backend':<25} {results1['backend']:>20} {results2['backend']:>20}")
    print("-"*70)
    
    diff = abs(results1['accuracy'] - results2['accuracy']) * 100
    print("\nAnalysis:")
    if diff < 2:
        print(f"  - Both models perform similarly (within 2%)")
    elif results1['accuracy'] > results2['accuracy']:
        print(f"  - {results1['model']} outperforms {results2['model']} by {diff:.1f}%")
    else:
        print(f"  - {results2['model']} outperforms {results1['model']} by {diff:.1f}%")
    print()


# Modal deployment for persistent inference
try:
    import modal
    
    app = modal.App("math-eval")
    
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch",
            "transformers>=4.40.0",
            "accelerate",
            "sentencepiece",
            "protobuf",
        )
    )
    
    @app.cls(
        image=image,
        gpu="A10G",
        timeout=1200,
        scaledown_window=300,
    )
    class Model:
        model_id: str = modal.parameter(default="Qwen/Qwen3-1.7B")

        @modal.enter()
        def load_model(self):
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading {self.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"Model loaded!")
        
        @modal.method()
        def generate(self, prompts: list[str], max_tokens: int = 1024) -> list[str]:
            import torch
            
            outputs = []
            for i, prompt in enumerate(prompts):
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=2048
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                output = self.tokenizer.decode(
                    generated[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
                outputs.append(output)
                
                if (i + 1) % 5 == 0:
                    print(f"  Generated {i+1}/{len(prompts)}")
            
            return outputs
    
    @app.local_entrypoint()
    def modal_main(
        model: str = "Qwen/Qwen3-1.7B",
        n_examples: int = 20,
        max_tokens: int = 1024,
    ):
        """Run evaluation via Modal deployment."""
        from datasets import load_dataset
        
        print(f"Loading dataset...")
        examples = load_math_dataset(n_examples)
        
        print(f"Creating prompts...")
        prompts = [create_few_shot_prompt(ex["problem"]) for ex in examples]
        
        print(f"Running inference on Modal...")
        model_cls = Model(model_id=model)
        outputs = model_cls.generate.remote(prompts, max_tokens)
        
        correct = 0
        for example, output in zip(examples, outputs):
            ground_truth = extract_ground_truth(example["solution"])
            predicted = extract_answer(output)
            if check_answer(predicted, ground_truth):
                correct += 1
        
        accuracy = correct / len(examples)
        print(f"\nResults for {model}:")
        print(f"  Accuracy: {accuracy*100:.1f}% ({correct}/{len(examples)})")

except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Model-agnostic evaluation on MATH dataset"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., Qwen/Qwen3-1.7B)"
    )
    parser.add_argument(
        "--model2",
        type=str,
        default=None,
        help="Second model to compare against (optional)"
    )
    parser.add_argument(
        "-n", "--n-examples",
        type=int,
        default=20,
        help="Number of examples (default: 20)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Use Modal for cloud inference"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="algebra",
        choices=["algebra", "counting_and_probability", "geometry", 
                 "intermediate_algebra", "number_theory", "prealgebra", "precalculus"],
        help="MATH dataset subset (default: algebra)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    print("\n" + "="*70)
    print("MATH Benchmark Evaluation")
    print("="*70)
    print(f"Model: {args.model}")
    if args.model2:
        print(f"Model 2: {args.model2}")
    print(f"Examples: {args.n_examples}")
    print(f"Subset: {args.subset}")
    print(f"Backend: {'Modal' if args.modal else 'Local'}")
    print(f"Output directory: {output_dir}")
    
    examples = load_math_dataset(args.n_examples, subset=args.subset)
    print(f"Loaded {len(examples)} examples")
    
    results1 = evaluate_model(
        args.model,
        examples,
        output_dir,
        use_modal=args.modal,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    results2 = None
    if args.model2:
        results2 = evaluate_model(
            args.model2,
            examples,
            output_dir,
            use_modal=args.modal,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print_comparison(results1, results2)
        
        comparison_path = output_dir / "comparison.json"
        comparison = {
            "benchmark": "MATH",
            "n_examples": args.n_examples,
            "timestamp": datetime.now().isoformat(),
            "model1": results1,
            "model2": results2,
        }
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to: {comparison_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
