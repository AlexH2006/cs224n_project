"""
Local generation test: run Qwen3.5 on the first MiniF2F problem (pass@4),
parse outputs, verify with Kimina, save results.

No Modal — loads the model locally via transformers/vLLM.
Uses CPU or MPS (Apple Silicon) if no CUDA GPU is available.

Run with:
    python3 qwen_multiturn/tests/test_local_gen.py --model Qwen/Qwen3.5-9B
    python3 qwen_multiturn/tests/test_local_gen.py --model Qwen/Qwen3.5-4B

Kimina Docker must be running:
    docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_multiturn.config import EvalConfig
from qwen_multiturn.dataset import load_problems
from qwen_multiturn.local_lean_verifier import verify
from qwen_multiturn.parsing import create_full_lean_code, extract_full_lean_block
from qwen_multiturn.prompts import build_prompt
from qwen_multiturn.results import build_problem_log, make_run_dir, save_results


def generate_local(
    prompts: list[str],
    model_name: str,
    cfg: EvalConfig,
) -> list[str]:
    """
    Generate outputs locally using transformers pipeline.
    Falls back gracefully if model is too large for available RAM.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Loading model on device: {device} (this may take a while)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device,
    )

    print(f"  Generating {len(prompts)} outputs (max_new_tokens={cfg.max_new_tokens})...")
    outputs = gen_pipeline(
        prompts,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        do_sample=True,
        return_full_text=False,  # only return generated text, not the prompt
    )

    # pipeline returns list of list[dict]; extract generated_text from each
    return [o[0]["generated_text"] for o in outputs]


def run(model_name: str, pass_k: int):
    cfg = EvalConfig(
        model_name=model_name,
        n_problems=1,
        pass_k=pass_k,
    )

    print("=" * 60)
    print(f"Local generation test  |  model={model_name}")
    print(f"  pass@{pass_k}, dataset={cfg.dataset_name}, split={cfg.dataset_split}")
    print(f"  kimina={cfg.kimina_url}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load problem + tokenizer (for prompt building)
    # ------------------------------------------------------------------
    print("\n[1/4] Loading problem...")
    problems = load_problems(cfg)
    problem = problems[0]
    print(f"  problem_id: {problem['problem_id']}")
    print(f"  theorem:    {problem['formal_statement'][:100]}...")

    from transformers import AutoTokenizer
    print(f"\n[2/4] Building {pass_k} prompts...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    prompt = build_prompt(
        theorem_code=problem["formal_statement"],
        informal=problem["informal_stmt"],
        header=problem["header"],
        tokenizer=tokenizer,
        cfg=cfg,
    )
    prompts = [prompt] * pass_k
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Prompt preview:\n    {prompt[:300].replace(chr(10), chr(10) + '    ')}")

    # ------------------------------------------------------------------
    # 3. Generate
    # ------------------------------------------------------------------
    print(f"\n[3/4] Generating {pass_k} outputs...")
    raw_outputs = generate_local(prompts, model_name, cfg)

    # ------------------------------------------------------------------
    # 4. Parse → Verify → Save
    # ------------------------------------------------------------------
    print(f"\n[4/4] Parsing, verifying, saving...")
    attempt_logs = []
    for i, raw_output in enumerate(raw_outputs):
        print(f"\n  --- Attempt {i} ---")
        print(f"  raw_output (first 300 chars):\n    {raw_output[:300].replace(chr(10), chr(10) + '    ')}")

        extracted = extract_full_lean_block(raw_output)
        full_code = create_full_lean_code(
            theorem_code=problem["formal_statement"],
            extracted_block=extracted,
            default_header=cfg.default_header,
        )

        print(f"  extracted_block: {'<sorry>' if extracted == 'sorry' else extracted[:120].replace(chr(10), ' ')!r}")

        result = verify(full_code, kimina_url=cfg.kimina_url, timeout=cfg.verify_timeout_s)
        success = result.get("success") and result.get("complete") and not result.get("has_sorry")
        print(f"  success={result.get('success')}  complete={result.get('complete')}  has_sorry={result.get('has_sorry')}")
        if result.get("errors"):
            print(f"  errors: {result['errors'][0][:150]}")

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
    print(f"Results: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--pass-k", type=int, default=4)
    args = parser.parse_args()
    run(args.model, args.pass_k)
