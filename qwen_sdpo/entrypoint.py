"""
TLDR: Local SDPO loop — orchestrates generate → parse → verify → update for one problem.

The local driver runs on your machine. GPU-heavy work (generation + gradient step)
executes on Modal via trainer.generate_batch.remote() and trainer.run_sdpo_step_minibatch.remote().
Parsing and verification use self-contained qwen_sdpo modules (parsing, _verifier).

Minibatch flow (config.minibatch_size, default 1): each iteration generates minibatch_size
samples in one vLLM batch; each sample is parsed/verified and a payload built; one
run_sdpo_step_minibatch aggregates valid (non-skip) losses and performs a single
gradient update. When minibatch_size=1, behavior matches the original single-sample flow.

Loop for each SDPO iteration:
  1. Modal: generate_batch(minibatch_size prompts) → list of (raw_output, ids, finish_reason)
  2. Local: for each result — parse, verify, build teacher prompt, build payload
  3. If any sample succeeded: set best_proof, call run_sdpo_step_minibatch (logging only), break
  4. Modal: run_sdpo_step_minibatch(payloads) → one aggregated iter_log, one optimizer step
  5. Log metrics, save locally

The teacher prompt is always built from the current iteration's verification
result, never from a prior step. Every failed, non-truncated iteration produces
a gradient update — there is no warm-up lag. Truncated generations skip the
update and the loop continues.

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
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from qwen_sdpo._dataset import get_field
from qwen_sdpo.checkpoint_manifest import (
    append_checkpoint_entry,
    init_manifest,
    ENTRY_BASE_MODEL,
    ENTRY_LOCAL_RUN_DIR,
    ENTRY_MAX_ITERATIONS,
    ENTRY_MODAL_RUN_DIR,
    ENTRY_PROBLEM_ID,
    ENTRY_PROBLEM_IDX,
    ENTRY_SUCCESS,
)
from qwen_sdpo._verifier import verify as kimina_verify
from qwen_sdpo.parsing import (
    create_full_lean_code,
    extract_full_lean_block_parsed,
    get_answer_token_slice,
    get_code_token_slice,
)
from qwen_sdpo.config import SDPOConfig
from qwen_sdpo.modal_trainer import _should_skip_gradient_step
from qwen_sdpo.prompts import (
    build_student_prompt,
    build_teacher_prompt,
    build_teacher_prompt_no_thinking,
)
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


def _build_cfg_dict(cfg: SDPOConfig) -> dict:
    """Build the serialisable config dict sent to Modal methods. Used by run_main and tests."""
    return {
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
        "use_think_mode": cfg.use_think_mode,
        "teacher_response_mode": cfg.teacher_response_mode,
        "max_feedback_errors": cfg.max_feedback_errors,
        "minibatch_size": cfg.minibatch_size,
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


def _feedback_for_teacher_prompt(verification: dict, max_errors: int) -> str:
    """Return the feedback string for the teacher prompt, truncated to the first max_errors when verification has an errors list."""
    errors = verification.get("errors") or []
    if isinstance(errors, list) and len(errors) > 0:
        first_n = errors[:max_errors]
        return "\n".join(str(e) for e in first_n)
    return verification.get("feedback") or "Proof verification failed."


def _print_banner(cfg: SDPOConfig) -> None:
    print("=" * 60)
    print("SDPO Qwen3.5 — Test-Time RL on Modal")
    print("=" * 60)
    print(f"Model:           {cfg.model_name}")
    print(f"GPU:             {cfg.gpu}")
    print(f"Dataset:         {cfg.dataset_name} [{cfg.dataset_split}]")
    print(f"Problem index:   {cfg.problem_idx}")
    print(f"Max iterations:  {cfg.max_iterations}")
    print(f"Minibatch size:  {cfg.minibatch_size}")
    print(f"Feedback mode:   {'errors only' if cfg.feedback_errors_only else 'errors + failed proof'}")
    print(f"Think mode:      {'on' if cfg.use_think_mode else 'off (non-thinking)'}")
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
    cfg_dict = _build_cfg_dict(cfg)

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

        # --- Minibatch generate: one vLLM batch call ---
        prompts = [student_prompt] * cfg.minibatch_size
        results = trainer.generate_batch.remote(cfg_dict, prompts)
        print(f"  Generated minibatch of {len(results)} samples")

        payloads = []
        any_success = False
        for b, (raw_output, generated_ids_list, finish_reason) in enumerate(results):
            num_tokens = len(generated_ids_list)

            # --- Parse ---
            parse_result = extract_full_lean_block_parsed(
                raw_output,
                finish_reason=finish_reason,
            )
            extracted_block = parse_result.block
            is_truncated = parse_result.truncated
            is_no_block = parse_result.no_block
            full_code = create_full_lean_code(theorem_code, extracted_block, cfg.default_header)

            # --- Verify ---
            if parse_result.failed:
                verification = {"success": False, "complete": False, "errors": []}
                if is_truncated:
                    verification["truncated"] = True
                    verification["feedback"] = (
                        "Generation was truncated; no complete proof was produced."
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

            if is_success:
                any_success = True
                best_proof = extracted_block

            # --- Build teacher prompt from this sample's verification result ---
            feedback = _feedback_for_teacher_prompt(verification, cfg.max_feedback_errors)
            if cfg.use_think_mode:
                teacher_enable_thinking = is_truncated or cfg.teacher_response_mode == "full_output"
                teacher_prompt = build_teacher_prompt(
                    theorem_code, informal, header, feedback, tokenizer, cfg,
                    enable_thinking=teacher_enable_thinking,
                )
            else:
                teacher_prompt = build_teacher_prompt_no_thinking(
                    theorem_code, informal, header, feedback, tokenizer, cfg,
                )

            # --- Build payload: teacher_response_ids and teacher_response_mode ---
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

            # KL loss mask: when kl_mask_final_code_only, pass final code block token span
            # for loss only; context remains full. Trainer sums KL only over this span.
            kl_final_code_start = None
            kl_final_code_end = None
            if (
                cfg.kl_mask_final_code_only
                and not is_truncated
                and not parse_result.failed
                and parse_result.code_block_start_char is not None
                and parse_result.code_block_end_char is not None
            ):
                code_ids, code_start = get_code_token_slice(
                    raw_output,
                    generated_ids_list,
                    tokenizer,
                    parse_result.code_block_start_char,
                    parse_result.code_block_end_char,
                )
                kl_final_code_start = code_start
                kl_final_code_end = code_start + len(code_ids)

            payload = {
                "iteration": iteration + 1,
                "base_prompt": student_prompt,
                "teacher_prompt": teacher_prompt,
                "raw_output": raw_output,
                "generated_ids": generated_ids_list,
                "teacher_response_ids": teacher_response_ids,
                "teacher_response_mode": teacher_response_mode,
                "cot_len": cot_len,
                "kl_final_code_start": kl_final_code_start,
                "kl_final_code_end": kl_final_code_end,
                "extracted_block": extracted_block,
                "full_code": full_code,
                "verification": verification,
                "num_tokens": num_tokens,
                "feedback": feedback,
                "is_success": is_success,
                "is_server_error": is_server_error,
                "is_truncated": is_truncated,
            }
            payloads.append(payload)
            if any_success:
                break

        # --- Log and step: one call per iteration (minibatch or single) ---
        iter_log = trainer.run_sdpo_step_minibatch.remote(cfg_dict, payloads)
        logs["iteration_logs"].append(iter_log)

        n_valid = iter_log.get("n_valid", 0)
        minibatch_size = iter_log.get("minibatch_size", len(payloads))
        if any_success:
            metrics["iterations"].append(iteration + 1)
            metrics["losses"].append(0.0)
            metrics["rewards"].append(1.0)
            metrics["kl_divs"].append(0.0)
            metrics["entropies"].append(0.0)
            metrics["grad_norms"].append(0.0)
            metrics["timestamps"].append(time.time() - iter_start)
            print(f"  Proof verified! ({n_valid}/{minibatch_size} valid in minibatch)")
            break

        if n_valid == 0:
            print(f"  Skipping SDPO update (all {minibatch_size} skipped).")
            metrics["iterations"].append(iteration + 1)
            metrics["losses"].append(0.0)
            metrics["rewards"].append(0.0)
            metrics["kl_divs"].append(0.0)
            metrics["entropies"].append(0.0)
            metrics["grad_norms"].append(0.0)
            metrics["timestamps"].append(time.time() - iter_start)
            continue

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
            f"  Loss: {loss:.4f}  Reward: {reward:.4f}  Grad norm: {grad_norm:.4f}  "
            f"({n_valid}/{minibatch_size} valid)"
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

    # Include local_run_dir so batch driver can record it in the manifest without
    # duplicating make_run_dir / save_local_run logic.
    return {**results, "local_run_dir": str(local_run_dir)}


def run_main_batch(
    trainer: Any,
    cfg: SDPOConfig,
    problem_indices: list[int],
    problem_id_by_idx: dict[int, str],
    manifest_path: Path,
) -> dict:
    """Run SDPO for multiple problems: each from base HF model, persist after each, reset between.

    Iterates over problem_indices. For each problem: runs run_main (same as single-problem
    pipeline), appends one entry to the manifest, then calls reset_to_base.remote() so the
    next problem trains from base on the same GPU (no cold start). Results are saved
    after each problem (incremental persistence).

    Args:
        trainer: Modal QwenSDPOTrainer instance.
        cfg: Base SDPOConfig; problem_idx is overridden per problem.
        problem_indices: List of dataset indices to train on (order preserved).
        problem_id_by_idx: Map problem_idx -> problem_id (from sampled_problems JSON).
        manifest_path: Path to the local manifest JSON file.

    Returns:
        Summary dict with "results" (list of run_main return dicts), "manifest_path", "success_count".
    """
    manifest_path = Path(manifest_path)
    init_manifest(manifest_path, base_model=cfg.model_name, max_iterations=cfg.max_iterations)
    cfg_dict = _build_cfg_dict(cfg)
    results_list: list[dict] = []

    for i, problem_idx in enumerate(problem_indices):
        problem_cfg = replace(cfg, problem_idx=problem_idx)
        print(f"\n{'='*60}\nBatch problem {i+1}/{len(problem_indices)}: index {problem_idx}\n{'='*60}")
        run_result = run_main(trainer=trainer, cfg=problem_cfg)
        results_list.append(run_result)

        problem_id = (
            problem_id_by_idx.get(problem_idx)
            or run_result.get("problem_id")
            or str(problem_idx)
        )
        modal_run_dir = run_result.get("run_dir", "")
        local_run_dir = run_result.get("local_run_dir", "")
        success = run_result.get("success", False)

        entry = {
            ENTRY_PROBLEM_IDX: problem_idx,
            ENTRY_PROBLEM_ID: problem_id,
            ENTRY_BASE_MODEL: cfg.model_name,
            ENTRY_MAX_ITERATIONS: cfg.max_iterations,
            ENTRY_MODAL_RUN_DIR: modal_run_dir,
            ENTRY_LOCAL_RUN_DIR: local_run_dir,
            ENTRY_SUCCESS: success,
        }
        append_checkpoint_entry(manifest_path, entry)

        if i < len(problem_indices) - 1:
            print("Resetting to base model for next problem...")
            trainer.reset_to_base.remote(cfg_dict)

    success_count = sum(1 for r in results_list if r.get("success"))
    return {
        "results": results_list,
        "manifest_path": str(manifest_path),
        "success_count": success_count,
        "total": len(problem_indices),
    }
