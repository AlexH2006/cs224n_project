"""
Backup generation script using transformers (no vLLM) on Modal H100.

Avoids all FlashInfer / nvcc / GDN-kernel issues that affect vLLM + Qwen3.5.
Use this to quickly validate:
  - The model produces sensible Lean 4 proof attempts
  - Parsing logic works on real model output
  - The overall pipeline (generate → parse → verify) is correct

Usage:
    python3 -m modal run qwen_multiturn/modal_transformers_gen.py \\
        --model "Qwen/Qwen3.5-9B" --problem-idx 0 --pass-k 4

Kimina must be running locally before running this:
    docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
"""

from __future__ import annotations

import modal

from qwen_multiturn.config import EvalConfig
from qwen_multiturn.dataset import load_problems
from qwen_multiturn.local_lean_verifier import verify
from qwen_multiturn.parsing import create_full_lean_code, extract_full_lean_block
from qwen_multiturn.results import build_problem_log, make_run_dir, save_results

# ---------------------------------------------------------------------------
# Image: plain transformers — no vLLM, no FlashInfer JIT issues
# ---------------------------------------------------------------------------

app = modal.App("qwen-transformers-gen")

hf_cache_volume = modal.Volume.from_name("qwen-eval-hf-cache", create_if_missing=True)

gen_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51.0",
        "accelerate",
        "datasets",
        "sentencepiece",
        "protobuf",
    )
    .add_local_python_source("qwen_multiturn")
)


# ---------------------------------------------------------------------------
# Modal function: load model and generate on H100
# ---------------------------------------------------------------------------

@app.function(
    image=gen_image,
    gpu="H100",
    timeout=1800,
    volumes={"/hf_cache": hf_cache_volume},
)
def generate(
    model_name: str,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> list[str]:
    """Load Qwen3.5 via transformers and generate pass_k outputs."""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/hf_cache"

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print("Model loaded.")

    results: list[str] = []
    for i, prompt in enumerate(prompts):
        print(f"  Generating attempt {i + 1}/{len(prompts)}...")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )
        # Decode only the newly generated tokens (exclude the prompt)
        generated_ids = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text)
        print(f"  Attempt {i + 1} done ({len(text)} chars)")

    return results


# ---------------------------------------------------------------------------
# Local entrypoint: orchestrate generate → parse → verify → save
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_eval(
    model: str = "Qwen/Qwen3.5-9B",
    problem_idx: int = 0,
    pass_k: int = 4,
):
    from qwen_multiturn.prompts import build_prompt
    from transformers import AutoTokenizer

    cfg = EvalConfig(
        model_name=model,
        n_problems=1,
        pass_k=pass_k,
        problem_indices=[problem_idx],
    )

    print("=" * 60)
    print(f"Qwen Transformers Gen  |  model={model}")
    print(f"  problem_idx={problem_idx}, pass@{pass_k}")
    print("=" * 60)

    # 1. Load dataset problem
    problems = load_problems(cfg)
    problem = problems[0]
    print(f"\nProblem: {problem['problem_id']}")
    print(f"  {problem['formal_statement'][:120]}...")

    # 2. Build prompts (tokenizer loaded locally for prompt formatting)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    prompt = build_prompt(
        theorem_code=problem["formal_statement"],
        informal=problem["informal_stmt"],
        header=problem["header"],
        tokenizer=tokenizer,
        cfg=cfg,
    )
    prompts = [prompt] * pass_k
    print(f"\nPrompt ({len(prompt)} chars):\n{prompt[:400]}\n...")

    # 3. Generate on Modal H100
    print(f"\nGenerating {pass_k} attempts on Modal H100...")
    raw_outputs = generate.remote(
        model_name=model,
        prompts=prompts,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
    )

    # 4. Parse → Verify → Save
    print("\nParsing and verifying...")
    attempt_logs = []
    for i, raw_output in enumerate(raw_outputs):
        print(f"\n--- Attempt {i} ---")
        print(f"  Raw output ({len(raw_output)} chars, first 400):")
        print("  " + raw_output[:400].replace("\n", "\n  "))

        extracted = extract_full_lean_block(raw_output)
        full_code = create_full_lean_code(
            theorem_code=problem["formal_statement"],
            extracted_block=extracted,
            default_header=cfg.default_header,
        )

        print(f"\n  Extracted block (first 200 chars):")
        print("  " + (extracted or "<none>")[:200].replace("\n", "\n  "))

        result = verify(full_code, kimina_url=cfg.kimina_url, timeout=cfg.verify_timeout_s)
        success = (
            result.get("success")
            and result.get("complete")
            and not result.get("has_sorry")
        )
        print(f"  success={result.get('success')}  complete={result.get('complete')}  has_sorry={result.get('has_sorry')}")
        if result.get("errors"):
            print(f"  errors: {result['errors'][0][:200]}")

        attempt_logs.append({
            "attempt": i,
            "prompt": prompt,
            "raw_output": raw_output,
            "extracted_block": extracted,
            "full_code": full_code,
            "verification": result,
            "success": success,
            "num_tokens": len(raw_output.split()),
        })

    problem_log = build_problem_log(problem, attempt_logs, cfg)
    run_dir = make_run_dir(cfg)
    save_results(run_dir, cfg, [problem_log])

    print("\n" + "=" * 60)
    print(f"Solved: {problem_log['success']}  |  best_attempt: {problem_log['best_attempt']}")
    print(f"Results saved to: {run_dir}")
    print("=" * 60)
