"""
CLI entrypoint for local-lean-verify pipeline (Qwen3.5-4B): local loop with verification.

TLDR: run_main() loads dataset/config/problem, then runs the SDPO loop locally:
each iteration calls Modal generate_only -> parse + verify (local_lean_verifier, backend
local or kimina) -> Modal run_sdpo_step. At the end calls finalize_run.remote() and saves.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type

from imo_p6_experiment.config import SDPOConfig
from imo_p6_experiment.local_lean_verifier import verify as local_verify
from imo_p6_experiment.prompts import create_base_prompt, create_feedback_prompt
from imo_p6_experiment.parsing import extract_full_lean_block
from imo_p6_experiment.utils import (
    clamp_problem_idx,
    create_full_lean_code,
    get_field,
    load_dataset_with_fallback,
    plot_training_curves,
    print_problem_summary,
    print_run_banner,
    save_local_run,
    theorem_code_is_commented_out,
)


def run_main(
    trainer_cls: Type[Any],
    config: Optional[SDPOConfig] = None,
    **kwargs: Any,
) -> dict:
    """Run SDPO with local Lean verification. Loop runs locally; generate and train step on Modal."""
    cfg = config if config is not None else SDPOConfig()
    model = kwargs.get("model", cfg.model_name)
    dataset = kwargs.get("dataset", cfg.dataset_name)
    dataset_subset = kwargs.get("dataset_subset", cfg.dataset_subset) or ""
    dataset_split = kwargs.get("dataset_split", cfg.dataset_split)
    problem_idx = kwargs.get("problem_idx", cfg.problem_idx)
    max_iterations = kwargs.get("max_iterations", cfg.max_iterations)
    learning_rate = kwargs.get("learning_rate", cfg.learning_rate)
    temperature = kwargs.get("temperature", cfg.temperature)
    feedback_errors_only = kwargs.get("feedback_errors_only", cfg.feedback_errors_only)
    system_prompt = kwargs.get("system_prompt", cfg.system_prompt) or ""
    default_header = kwargs.get("default_header", cfg.default_header) or ""
    theorem_field = kwargs.get("theorem_field", cfg.theorem_field_override) or ""
    informal_field = kwargs.get("informal_field", cfg.informal_field_override) or ""
    header_field = kwargs.get("header_field", cfg.header_field_override) or ""
    gpu = kwargs.get("gpu", cfg.gpu)

    # Output path: local_verify/{model_name}/{dataset_name}/run_{problem_idx}_{timestamp}
    output_dir_name = kwargs.get("output_dir_name")
    if not output_dir_name:
        model_short = model.split("/")[-1] if "/" in model else model
        output_dir_name = f"local_verify/{model_short}"

    verify_backend = kwargs.get("verify_backend") or os.environ.get("LEAN_VERIFY_BACKEND", "kimina")
    kimina_base_url = kwargs.get("kimina_base_url") or os.environ.get("LEAN_VERIFY_KIMINA_URL", "http://localhost:8000")
    kimina_api_key = kwargs.get("kimina_api_key") or os.environ.get("LEAN_VERIFY_KIMINA_API_KEY")

    print_run_banner(
        model, gpu, dataset, dataset_subset, dataset_split,
        problem_idx, max_iterations, feedback_errors_only,
    )

    ds = load_dataset_with_fallback(dataset, dataset_subset, dataset_split)

    # Print test set (index and name/id for each problem)
    print(f"Test set ({len(ds)} problems):")
    for i in range(len(ds)):
        p = dict(ds[i])
        name = p.get("name") or get_field(p, cfg.id_fields, "")
        print(f"  {i}: {name}")

    problem_idx = clamp_problem_idx(problem_idx, len(ds))
    problem = dict(ds[problem_idx])

    theorem_fields = ([theorem_field] if theorem_field else []) + cfg.theorem_fields
    informal_fields = ([informal_field] if informal_field else []) + cfg.informal_fields
    header_fields = ([header_field] if header_field else []) + cfg.header_fields
    id_fields = cfg.id_fields

    print_problem_summary(
        problem, problem_idx, id_fields,
        theorem_fields, informal_fields, header_fields,
    )

    config_dict = {
        "model_name": model,
        "dataset_name": dataset,
        "dataset_subset": dataset_subset or None,
        "dataset_split": dataset_split,
        "problem_idx": problem_idx,
        "max_iterations": max_iterations,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "feedback_include_failed_proof": not feedback_errors_only,
        "theorem_fields": theorem_fields,
        "informal_fields": informal_fields,
        "header_fields": header_fields,
        "id_fields": id_fields,
        "output_dir": output_dir_name,
    }

    if system_prompt:
        config_dict["system_prompt"] = system_prompt
    if default_header:
        config_dict["default_header"] = default_header

    config_dict.setdefault("default_header", cfg.default_header)
    config_dict.setdefault("distillation_topk", cfg.distillation_topk)
    config_dict.setdefault("max_new_tokens", cfg.max_new_tokens)
    config_dict.setdefault("top_p", cfg.top_p)
    config_dict.setdefault("stop_tokens", cfg.stop_tokens)

    print(f"Using {gpu} GPU")
    print(
        f"Verification: {verify_backend} (Kimina Docker)"
        if verify_backend == "kimina"
        else "Verification: local (Goedel-Prover lake exe repl)"
    )

    trainer = trainer_cls(model_name=model, gpu=gpu)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    run_config = SDPOConfig(**config_dict)
    base_prompt = create_base_prompt(run_config, problem, get_field, tokenizer)

    metrics = {
        "iterations": [],
        "losses": [],
        "rewards": [],
        "kl_divs": [],
        "entropies": [],
        "grad_norms": [],
        "timestamps": [],
    }
    logs = {
        "problem": problem,
        "config": config_dict,
        "iteration_logs": [],
        "start_time": datetime.now().isoformat(),
    }
    best_proof = None

    for iteration in range(max_iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        raw_output, generated_ids = trainer.generate_only.remote(config_dict, base_prompt)
        num_tokens = len(generated_ids)
        print(f"  Generated {len(raw_output)} chars, {num_tokens} tokens")

        is_truncated = "<think>" in raw_output and "</think>" not in raw_output
        extracted_block = extract_full_lean_block(raw_output)
        lean4_code = get_field(problem, run_config.theorem_fields)
        header = get_field(problem, run_config.header_fields)

        if theorem_code_is_commented_out(lean4_code):
            print("  Skipping: formal statement is entirely commented out (unsupported problem)")
            verification = {
                "success": False,
                "complete": False,
                "has_sorry": True,
                "feedback": "Formal statement is entirely commented out.",
                "errors": ["Commented-out formal statement"],
                "messages": [],
                "sorries": [],
                "source": "skipped",
                "is_server_error": False,
            }
            full_code = lean4_code
            iter_log = {
                "iteration": iteration + 1,
                "student_prompt": base_prompt,
                "teacher_prompt": None,
                "raw_output": raw_output,
                "extracted_block": extracted_block,
                "full_code": full_code,
                "verification": verification,
                "success": False,
                "loss": None,
                "reward": None,
                "kl_div": None,
                "entropy": None,
                "grad_norm": None,
                "num_tokens": num_tokens,
            }
            logs["iteration_logs"].append(iter_log)
            break

        full_code = create_full_lean_code(
            theorem_code=lean4_code,
            extracted_block=extracted_block,
            header=header,
            default_header=run_config.default_header,
        )

        max_verify_retries = 3
        verification = None
        for verify_attempt in range(max_verify_retries):
            print(f"  Verifying ({verify_backend}) (attempt {verify_attempt + 1}/{max_verify_retries})...")
            verification = local_verify(
                full_code,
                timeout=300,
                backend=verify_backend,
                kimina_base_url=kimina_base_url,
                kimina_api_key=kimina_api_key,
            )
            if not verification.get("is_server_error", False):
                break
            if verify_attempt < max_verify_retries - 1:
                print("  Local verification error, retrying...")
                time.sleep(2)

        is_success = verification["success"] and verification["complete"]
        is_server_error = verification.get("is_server_error", False)
        is_sorry_like = extracted_block.strip().lower() == "sorry"

        if is_sorry_like:
            is_success = False
            verification["complete"] = False
            verification["has_sorry"] = True
            if is_truncated:
                verification["feedback"] = "Reasoning was cut off before completion."
            else:
                verification["feedback"] = "No valid Lean 4 code block found. Output a complete proof in a ```lean4 code block."

        if is_server_error:
            print("  Verification: SERVER ERROR (will not count as failed proof)")
        elif is_truncated:
            print("  Verification: FAILED (output truncated)")
        elif is_sorry_like:
            print("  Verification: FAILED (no valid code block extracted)")
        else:
            print(f"  Verification: {'SUCCESS' if is_success else 'FAILED'}")

        if not is_success and not is_server_error:
            feedback = verification.get("feedback", "")
            errors = verification.get("errors", [])
            if feedback:
                lines = feedback.strip().split("\n")
                block = "  Lean feedback:\n" + "\n".join(f"    {line}" for line in lines)
                print(block, flush=True)
            elif errors:
                err_block = "  Lean errors:\n" + "\n".join(f"    {e}" for e in errors)
                print(err_block, flush=True)
            else:
                print("  (no Lean feedback captured)", flush=True)

        feedback = verification.get("feedback") or "Proof verification failed."
        teacher_prompt = create_feedback_prompt(
            run_config,
            problem,
            latest_feedback=(feedback, ""),
            feedback_history=[],
            get_field=get_field,
            tokenizer=tokenizer,
        )

        print("\n" + "=" * 60)
        print(f"PROMPTS & ATTEMPTED PROOF (iteration {iteration + 1}/{max_iterations})")
        print("=" * 60)
        if iteration == 0:
            print("\n--- Base (student) prompt ---")
            print(base_prompt)
            print("--- End base prompt ---\n")
        print("--- Teacher prompt ---")
        print(teacher_prompt)
        print("--- End teacher prompt ---\n")
        print("--- Attempted proof ---")
        print(extracted_block if extracted_block.strip() else "(empty or no block extracted)")
        print("--- End attempted proof ---\n")

        payload = {
            "iteration": iteration + 1,
            "base_prompt": base_prompt,
            "teacher_prompt": teacher_prompt,
            "raw_output": raw_output,
            "generated_ids": generated_ids,
            "extracted_block": extracted_block,
            "full_code": full_code,
            "verification": verification,
            "num_tokens": num_tokens,
            "feedback": feedback,
            "is_success": is_success,
            "is_server_error": is_server_error,
        }

        if is_success:
            best_proof = extracted_block
            iter_log = trainer.run_sdpo_step.remote(config_dict, problem, payload)
            iter_log["loss"] = None
            iter_log["reward"] = None
            iter_log["kl_div"] = None
            iter_log["entropy"] = None
            iter_log["grad_norm"] = None
            logs["iteration_logs"].append(iter_log)
            metrics["iterations"].append(iteration + 1)
            metrics["losses"].append(0.0)
            metrics["rewards"].append(1.0)
            metrics["kl_divs"].append(0.0)
            metrics["entropies"].append(0.0)
            metrics["grad_norms"].append(0.0)
            metrics["timestamps"].append(time.time() - iter_start)
            print("  ✓ Proof found!")
            break

        if is_server_error:
            iter_log = trainer.run_sdpo_step.remote(config_dict, problem, payload)
            logs["iteration_logs"].append(iter_log)
            print("  Skipping SDPO update due to server error")
            continue

        iter_log = trainer.run_sdpo_step.remote(config_dict, problem, payload)
        logs["iteration_logs"].append(iter_log)

        metrics["iterations"].append(iteration + 1)
        metrics["losses"].append(iter_log.get("loss", 0.0) or 0.0)
        metrics["rewards"].append(iter_log.get("reward", 0.0) or 0.0)
        metrics["kl_divs"].append(iter_log.get("kl_div", 0.0) or 0.0)
        metrics["entropies"].append(iter_log.get("entropy", 0.0) or 0.0)
        metrics["grad_norms"].append(iter_log.get("grad_norm", 0.0) or 0.0)
        metrics["timestamps"].append(time.time() - iter_start)

    final_model_path = trainer.finalize_run.remote(config_dict)
    logs["end_time"] = datetime.now().isoformat()
    logs["best_proof"] = best_proof
    logs["final_model_path"] = final_model_path

    save_path = save_local_run(Path("."), run_config, logs, metrics)
    print(f"\nSaved run to: {save_path}")

    if metrics["iterations"]:
        plot_training_curves(
            metrics,
            Path(save_path) / "training_curves.png",
            f"SDPO Test-Time RL - {run_config.model_name}",
        )

    return {
        "best_proof": best_proof,
        "metrics": metrics,
        "logs": logs,
        "save_path": save_path,
        "final_model_path": final_model_path,
    }