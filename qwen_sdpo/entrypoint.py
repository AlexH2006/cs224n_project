"""
TLDR: Local SDPO loop — orchestrates generate → parse → verify → update for one problem.

The local driver runs on your machine. GPU-heavy work (generation + gradient step)
executes on Modal via trainer.generate_only.remote() and trainer.run_sdpo_step.remote().
Parsing and verification use self-contained qwen_sdpo modules (parsing, _verifier).

Loop for each SDPO iteration:
  1. Modal: generate one proof attempt (vLLM, student prompt)
  2. Local: parse the lean4 code block  (qwen_sdpo.parsing)
  3. Local: verify with Kimina Docker   (qwen_sdpo._verifier)
  4. Build teacher prompt from THIS iteration's compiler feedback
  5. Modal: SDPO gradient step          (student + teacher prompts)
  6. Log metrics, save locally

The teacher prompt is always built from the current iteration's verification
result, never from a prior step. Every failed iteration produces a gradient
update — there is no warm-up lag.

Exit conditions:
  - Proof verified successfully (is_success=True): loop ends, best proof saved.
  - max_iterations reached: training stops.

Usage:
  Typically called by modal_app.run_sdpo() (CLI entry point).
  Can also be called directly for testing:

      from qwen_sdpo.config import SDPOConfig
      from qwen_sdpo.entrypoint import run_main
      from qwen_sdpo.modal_trainer import QwenSDPOTrainer

      cfg = SDPOConfig(model_name="Qwen/Qwen3.5-4B", problem_idx=0)
      trainer = QwenSDPOTrainer(model_name=cfg.model_name, gpu=cfg.gpu)
      run_main(trainer=trainer, cfg=cfg)

Prerequisite: Kimina Docker must be running locally:
  docker run -d -p 8000:8000 projectnumina/kimina-lean-server:2.0.0

Used by: modal_app.py (@app.local_entrypoint run_sdpo).
"""

import time
from datetime import datetime
from typing import Any, Optional

from qwen_sdpo._dataset import get_field
from qwen_sdpo._verifier import verify as kimina_verify
from qwen_sdpo.parsing import (
    create_full_lean_code,
    extract_full_lean_block_parsed,
    get_answer_token_slice,
    get_code_token_slice,
)
from qwen_sdpo.config import SDPOConfig
from qwen_sdpo.prompts import build_student_prompt, build_teacher_prompt
from qwen_sdpo.results import plot_training_curves, save_local_run


def _load_problem(cfg: SDPOConfig) -> dict:
    """Load one problem from the HuggingFace dataset specified in cfg."""
    from datasets import load_dataset

    print(f"Loading dataset: {cfg.dataset_name} (split={cfg.dataset_split})")
    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    print(f"  Dataset size: {len(ds)} problems")

    idx = cfg.problem_idx
    if idx >= len(ds):
        print(f"  WARNING: problem_idx {idx} out of range, using 0")
        idx = 0

    row = ds[idx]
    problem = {
        "problem_idx": idx,
        "problem_id": get_field(row, cfg.id_fields, default=str(idx)),
        "formal_statement": get_field(row, cfg.theorem_fields),
        "informal_stmt": get_field(row, cfg.informal_fields),
        "header": get_field(row, cfg.header_fields),
    }
    print(f"  Loaded problem: {problem['problem_id']}")
    return problem


def _print_banner(cfg: SDPOConfig) -> None:
    print("=" * 60)
    print("SDPO Qwen3.5 — Test-Time RL on Modal")
    print("=" * 60)
    print(f"Model:           {cfg.model_name}")
    print(f"GPU:             {cfg.gpu}")
    print(f"Dataset:         {cfg.dataset_name} [{cfg.dataset_split}]")
    print(f"Problem index:   {cfg.problem_idx}")
    print(f"Max iterations:  {cfg.max_iterations}")
    print(f"Feedback mode:   {'errors only' if cfg.feedback_errors_only else 'errors + failed proof'}")
    print(f"Teacher COT:     {cfg.teacher_response_mode}")
    print(f"Kimina URL:      {cfg.kimina_url}")
    print("=" * 60)


def _verify_with_retries(
    full_code: str, cfg: SDPOConfig
) -> dict:
    """Call Kimina verifier with retry logic on transient server errors."""
    result = None
    for attempt in range(cfg.verify_retries):
        print(f"  Verifying (attempt {attempt + 1}/{cfg.verify_retries})...")
        result = kimina_verify(
            full_code,
            kimina_url=cfg.kimina_url,
            timeout=cfg.verify_timeout_s,
        )
        if not result.get("is_server_error", False):
            break
        if attempt < cfg.verify_retries - 1:
            print(f"  Server error, waiting {cfg.verify_retry_wait_s}s before retry...")
            time.sleep(cfg.verify_retry_wait_s)
    return result


def run_main(
    trainer: Any,
    cfg: Optional[SDPOConfig] = None,
) -> dict:
    """Run the SDPO loop locally for one problem. Returns the final results dict.

    Args:
        trainer: Instantiated Modal QwenSDPOTrainer object (already bound to a GPU).
                 Created in modal_app.py as QwenSDPOTrainer(model_name=..., gpu=...).
        cfg:     SDPOConfig. Uses defaults if None.

    Returns:
        Results dict mirroring the Modal logs structure, also saved locally.
    """
    cfg = cfg or SDPOConfig()
    _print_banner(cfg)

    problem = _load_problem(cfg)
    problem_id = problem["problem_id"]
    theorem_code = problem["formal_statement"]
    informal = problem["informal_stmt"]
    header = problem["header"]

    # Load tokenizer locally (lightweight — no model weights) just for prompt formatting.
    print(f"\nLoading tokenizer for {cfg.model_name}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-build the fixed student prompt (same for every iteration).
    student_prompt = build_student_prompt(theorem_code, informal, header, tokenizer, cfg)

    # Serialisable config dict sent to Modal methods.
    cfg_dict = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "dataset_split": cfg.dataset_split,
        "problem_idx": cfg.problem_idx,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "min_p": cfg.min_p,
        "repetition_penalty": cfg.repetition_penalty,
        "max_iterations": cfg.max_iterations,
        "learning_rate": cfg.learning_rate,
        "distillation_topk": cfg.distillation_topk,
        "feedback_errors_only": cfg.feedback_errors_only,
        "teacher_response_mode": cfg.teacher_response_mode,
        "default_header": cfg.default_header,
        "use_lora": cfg.use_lora,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "kimina_url": cfg.kimina_url,
        "verify_timeout_s": cfg.verify_timeout_s,
        "verify_retries": cfg.verify_retries,
        "verify_retry_wait_s": cfg.verify_retry_wait_s,
        "results_base_dir": cfg.results_base_dir,
        "gpu": cfg.gpu,
    }

    metrics = {
        "iterations": [], "losses": [], "rewards": [],
        "kl_divs": [], "entropies": [], "grad_norms": [], "timestamps": [],
    }
    logs = {
        "problem_id": problem_id,
        "problem": problem,
        "config": cfg_dict,
        "iteration_logs": [],
        "start_time": datetime.now().isoformat(),
    }

    best_proof: Optional[str] = None

    for iteration in range(cfg.max_iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration + 1}/{cfg.max_iterations} ---")

        # --- Generate ---
        raw_output, generated_ids_list, finish_reason = trainer.generate_only.remote(
            cfg_dict, student_prompt
        )
        num_tokens = len(generated_ids_list)
        print(f"  Generated {len(raw_output)} chars, {num_tokens} tokens, finish={finish_reason}")

        # --- Parse ---
        # finish_reason="length" means vLLM hit max_tokens and stopped forcibly —
        # the definitive truncation signal, regardless of think-tag presence.
        parse_result = extract_full_lean_block_parsed(
            raw_output,
            finish_reason=finish_reason,
        )
        extracted_block = parse_result.block
        is_truncated = parse_result.truncated
        is_no_block = parse_result.no_block
        full_code = create_full_lean_code(theorem_code, extracted_block, cfg.default_header)

        # --- Verify ---
        # Skip the network round-trip if parsing already failed — the verifier
        # would just see a sorry-bearing file and produce an unhelpful error.
        if parse_result.failed:
            verification = {"success": False, "complete": False, "errors": []}
            if is_truncated:
                verification["truncated"] = True
                verification["feedback"] = (
                    "FATAL ERROR: Generation truncated due to excessively lengthy reasoning. "
                    "The attempt is severely flawed because it repeatedly second-guesses itself."
                )
            else:
                verification["no_block"] = True
                verification["feedback"] = (
                    "No lean4 code block was found in your response. "
                    "At the very end of your response, you MUST output your proof as "
                    "exactly one ```lean4 code block that is complete and self-contained."
                )
        else:
            verification = _verify_with_retries(full_code, cfg)

        is_server_error = verification.get("is_server_error", False)
        is_success = (
            not parse_result.failed
            and verification.get("success", False)
            and verification.get("complete", False)
        )

        # Log parse + verification result.
        if is_truncated:
            print("  Parse: TRUNCATED (reasoning exceeded token limit — no code block reached)")
        elif is_no_block:
            print("  Parse: FAILED (no lean4 code block found in output)")
        elif is_server_error:
            print("  Verification: SERVER ERROR (skipping SDPO update)")
        else:
            print(f"  Verification: {'SUCCESS' if is_success else 'FAILED'}")

        # Print feedback when the proof failed.
        if not is_success and not is_server_error:
            feedback_text = verification.get("feedback", "")
            errors = verification.get("errors", [])
            if feedback_text:
                lines = feedback_text.strip().split("\n")
                label = "Truncation feedback" if is_truncated else "Parse feedback" if is_no_block else "Lean feedback"
                print(f"  {label}:\n" + "\n".join(f"    {l}" for l in lines), flush=True)
            elif errors:
                print("  Lean errors:\n" + "\n".join(f"    {e}" for e in errors), flush=True)
            else:
                print("  (no feedback captured)", flush=True)

        # --- Build teacher prompt from THIS iteration's verification result. ---
        # The teacher is the feedback-conditioned policy: same model, same problem,
        # but with the compiler error from the current attempt appended to the prompt.
        # This is the core of SDPO self-distillation — the teacher signal is always
        # derived from the immediately preceding generation, never from a prior step.
        feedback = verification.get("feedback") or "Proof verification failed."
        # Disable Qwen3.5 thinking for answer_only/code_only unless truncated.
        # When truncated, re-enable thinking so model learns shorter reasoning.
        teacher_enable_thinking = is_truncated or cfg.teacher_response_mode == "full_output"
        teacher_prompt: Optional[str] = build_teacher_prompt(
            theorem_code, informal, header, feedback, tokenizer, cfg,
            enable_thinking=teacher_enable_thinking,
        )

        # --- Build payload: teacher_response_ids and teacher_response_mode ---
        # Apply mode-specific slice logic. Truncation: answer_only still uses get_answer_token_slice;
        # code_only has no block when truncated → use full. Payload uses "full" | "answer_only" | "code_only".
        mode = cfg.teacher_response_mode
        if mode == "full_output":
            teacher_response_ids = generated_ids_list
            teacher_response_mode = "full"
            cot_len = 0
        elif mode == "answer_only":
            answer_ids, cot_len = get_answer_token_slice(
                raw_output, generated_ids_list, tokenizer
            )
            teacher_response_ids = answer_ids
            teacher_response_mode = "answer_only"
        elif mode == "code_only":
            if (
                is_truncated
                or parse_result.failed
                or parse_result.code_block_start_char is None
                or parse_result.code_block_end_char is None
            ):
                teacher_response_ids = generated_ids_list
                teacher_response_mode = "full"
                cot_len = 0
            else:
                code_ids, code_start = get_code_token_slice(
                    raw_output,
                    generated_ids_list,
                    tokenizer,
                    parse_result.code_block_start_char,
                    parse_result.code_block_end_char,
                )
                if code_start == 0 and len(code_ids) == len(generated_ids_list):
                    teacher_response_ids = generated_ids_list
                    teacher_response_mode = "full"
                    cot_len = 0
                else:
                    teacher_response_ids = code_ids
                    teacher_response_mode = "code_only"
                    cot_len = code_start
        else:
            teacher_response_ids = generated_ids_list
            teacher_response_mode = "full"
            cot_len = 0

        payload = {
            "iteration": iteration + 1,
            "base_prompt": student_prompt,
            "teacher_prompt": teacher_prompt,
            "raw_output": raw_output,
            "generated_ids": generated_ids_list,
            "teacher_response_ids": teacher_response_ids,
            "teacher_response_mode": teacher_response_mode,
            "cot_len": cot_len,
            "extracted_block": extracted_block,
            "full_code": full_code,
            "verification": verification,
            "num_tokens": num_tokens,
            "feedback": feedback,
            "is_success": is_success,
            "is_server_error": is_server_error,
        }

        if is_success:
            # Proof found — log this iteration (no gradient update needed) then stop.
            best_proof = extracted_block
            iter_log = trainer.run_sdpo_step.remote(cfg_dict, payload)
            logs["iteration_logs"].append(iter_log)
            metrics["iterations"].append(iteration + 1)
            metrics["losses"].append(0.0)
            metrics["rewards"].append(1.0)
            metrics["kl_divs"].append(0.0)
            metrics["entropies"].append(0.0)
            metrics["grad_norms"].append(0.0)
            metrics["timestamps"].append(time.time() - iter_start)
            print("  Proof verified!")
            break

        if is_server_error:
            # Don't update on Kimina outages — the feedback signal is unreliable.
            iter_log = trainer.run_sdpo_step.remote(cfg_dict, payload)
            logs["iteration_logs"].append(iter_log)
            print("  Skipping SDPO update (server error).")
            continue

        iter_log = trainer.run_sdpo_step.remote(cfg_dict, payload)
        logs["iteration_logs"].append(iter_log)
        loss = iter_log.get("loss") or 0.0
        reward = iter_log.get("reward") or 0.0
        grad_norm = iter_log.get("grad_norm") or 0.0
        metrics["iterations"].append(iteration + 1)
        metrics["losses"].append(loss)
        metrics["rewards"].append(reward)
        metrics["kl_divs"].append(iter_log.get("kl_div") or 0.0)
        metrics["entropies"].append(iter_log.get("entropy") or 0.0)
        metrics["grad_norms"].append(grad_norm)
        metrics["timestamps"].append(time.time() - iter_start)
        print(
            f"  Loss: {loss:.4f}  "
            f"Reward: {reward:.4f}  "
            f"Grad norm: {grad_norm:.4f}"
        )

    # --- Finalise ---
    logs["end_time"] = datetime.now().isoformat()
    logs["success"] = best_proof is not None
    logs["best_proof"] = best_proof
    logs["metrics"] = metrics
    logs["total_generation_tokens"] = sum(
        e.get("num_tokens", 0) for e in logs["iteration_logs"]
    )

    results = trainer.finalize_run.remote(cfg_dict, logs)

    # Print summary.
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success:          {results['success']}")
    print(f"Iterations used:  {len(results['iteration_logs'])}")
    if results["success"] and results.get("best_proof"):
        preview = results["best_proof"][:200]
        print(f"Best proof (preview):\n  {preview}...")
    if results["metrics"].get("losses"):
        print(f"Final loss:       {results['metrics']['losses'][-1]:.4f}")
    if results.get("run_dir"):
        print(f"Modal volume:     {results['run_dir']}")

    # Mirror results to local disk.
    # Use the local `logs` dict (not `results`) so that per_token_kl records
    # are still present and _save_kl_artifacts can generate the KL heatmaps.
    # `results` has already had per_token_kl stripped on the Modal volume side.
    local_run_dir = save_local_run(logs, cfg, cfg.problem_idx)
    if metrics.get("iterations"):
        model_tag = cfg.model_name.split("/")[-1]
        plot_training_curves(
            metrics,
            local_run_dir / "training_curves.png",
            title=f"SDPO — {model_tag} — idx={cfg.problem_idx}",
        )
        print(f"Training curves:  {local_run_dir / 'training_curves.png'}")

    return results
