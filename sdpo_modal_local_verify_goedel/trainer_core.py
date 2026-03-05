"""
Core SDPO loop: generate → verify → feedback → loss → step. No Modal dependency.

TLDR: run_sdpo(config, problem, verify_fn, generate_fn, model, tokenizer, get_field, ...)
implements the single-problem iteration loop and returns logs. Used by: modal_trainer.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from sdpo_modal_local_verify_goedel.config import SDPOConfig
from sdpo_modal_local_verify_goedel.prompts import create_base_prompt, create_feedback_prompt
from sdpo_modal_local_verify_goedel.parsing import extract_full_lean_block, is_truncated_output
from sdpo_modal_local_verify_goedel.utils import (
    collect_per_token_kl,
    create_full_lean_code,
    get_field as default_get_field,
    save_run,
    theorem_code_is_commented_out,
)
from sdpo_modal_local_verify_goedel.sdpo_loss import compute_sdpo_loss


def run_sdpo(
    config: SDPOConfig,
    problem: dict,
    *,
    verify_fn: Callable[[str], dict],
    generate_fn: Callable[[str], tuple[str, "torch.Tensor"]],
    model,
    tokenizer,
    get_field: Optional[Callable[[dict, list, str], str]] = None,
    debug_lean_path: Optional[Path] = None,
    output_root: Path,
    config_dict_for_logs: Optional[dict] = None,
) -> dict:
    """Run SDPO test-time RL on a single problem. Returns logs dict (iteration_logs, metrics, success, best_proof, ...)."""
    import torch

    get_field = get_field or default_get_field

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
        "config": config.__dict__ if hasattr(config, "__dict__") else {},
        "iteration_logs": [],
        "start_time": datetime.now().isoformat(),
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    latest_feedback: tuple[str, str] | None = None
    best_proof = None

    base_prompt = create_base_prompt(config, problem, get_field, tokenizer)

    for iteration in range(config.max_iterations):
        iter_start = time.time()
        print(f"\n--- Iteration {iteration + 1}/{config.max_iterations} ---")

        # 1. Generate
        raw_output, generated_ids = generate_fn(base_prompt)
        num_tokens = int(generated_ids.numel())
        print(f"  Generated {len(raw_output)} chars, {num_tokens} tokens")

        is_truncated = is_truncated_output(raw_output)
        extracted_block = extract_full_lean_block(raw_output)

        lean4_code = get_field(problem, config.theorem_fields)
        header = get_field(problem, config.header_fields)

        # 2. Guard: commented-out theorem
        if theorem_code_is_commented_out(lean4_code):
            print("  Skipping: formal statement is entirely commented out (unsupported problem)")
            verification = {
                "success": False,
                "complete": False,
                "has_sorry": True,
                "feedback": "Formal statement is entirely commented out — this problem is not supported (e.g. '-- Error: Real.log').",
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

        # 3. Build full Lean code and verify
        full_code = create_full_lean_code(
            theorem_code=lean4_code,
            extracted_block=extracted_block,
            header=header,
            default_header=config.default_header,
        )

        max_verify_retries = 3
        verification = None
        for verify_attempt in range(max_verify_retries):
            print(f"  Verifying with Kimina Lean Server... (attempt {verify_attempt + 1}/{max_verify_retries})")
            if debug_lean_path:
                try:
                    with open(debug_lean_path, "a") as dbg:
                        dbg.write(
                            json.dumps(
                                {
                                    "event": "verify_start",
                                    "ts": datetime.now().isoformat(),
                                    "iteration": iteration + 1,
                                    "attempt": verify_attempt + 1,
                                    "code_len": len(full_code),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
            verification = verify_fn(full_code)
            if debug_lean_path:
                debug_info = verification.pop("debug", None) or {}
                try:
                    with open(debug_lean_path, "a") as dbg:
                        dbg.write(
                            json.dumps(
                                {
                                    "event": "verify_end",
                                    "ts": datetime.now().isoformat(),
                                    "iteration": iteration + 1,
                                    "attempt": verify_attempt + 1,
                                    "duration_s": round(debug_info.get("verifier_wall_s", 0), 3),
                                    "success": verification.get("success", False),
                                    "complete": verification.get("complete", False),
                                    "is_server_error": verification.get("is_server_error", False),
                                    "error_snippet": (verification.get("feedback") or "")[:500] or None,
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
            if not verification.get("is_server_error", False):
                break
            if verify_attempt < max_verify_retries - 1:
                print(f"  Server error on verification, retrying ({verify_attempt + 1}/{max_verify_retries})...")
                time.sleep(5)

        is_success = verification["success"] and verification["complete"]
        is_server_error = verification.get("is_server_error", False)

        is_sorry_like = extracted_block.strip().lower() == "sorry"
        if is_sorry_like:
            is_success = False
            verification["complete"] = False
            verification["has_sorry"] = True
            if is_truncated:
                verification["feedback"] = "Reasoning sequence trapped in a circular logic loop."
                verification["is_truncated"] = True
            else:
                verification["feedback"] = "No valid Lean 4 code block found. Output a complete proof in a ```lean4 code block."

        if is_server_error:
            print("  Verification: SERVER ERROR (will not count as failed proof)")
        elif is_truncated:
            print("  Verification: FAILED (output truncated, no code block produced)")
        elif is_sorry_like:
            print("  Verification: FAILED (no valid code block extracted)")
        else:
            print(f"  Verification: {'SUCCESS' if is_success else 'FAILED'}")

        current_teacher_prompt = (
            create_feedback_prompt(config, problem, latest_feedback, get_field, tokenizer)
            if latest_feedback is not None
            else None
        )

        iter_log = {
            "iteration": iteration + 1,
            "student_prompt": base_prompt,
            "teacher_prompt": current_teacher_prompt,
            "raw_output": raw_output,
            "extracted_block": extracted_block,
            "full_code": full_code,
            "verification": verification,
            "success": is_success,
            "num_tokens": num_tokens,
        }

        if is_success:
            best_proof = extracted_block
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
            iter_log["loss"] = None
            iter_log["reward"] = None
            iter_log["kl_div"] = None
            iter_log["entropy"] = None
            iter_log["grad_norm"] = None
            iter_log["server_error"] = True
            logs["iteration_logs"].append(iter_log)
            print("  Skipping SDPO update due to server error")
            continue

        if verification.get("has_sorry"):
            feedback = verification["feedback"] or "Forbidden to output `sorry` tactic. You must provide actual proof tactics that complete the proof."
        else:
            feedback = verification["feedback"] or "Proof verification failed"
        latest_feedback = (feedback, extracted_block)

        teacher_prompt = create_feedback_prompt(config, problem, latest_feedback, get_field, tokenizer)
        print("  SDPO: Student sees problem only, Teacher sees most recent compiler feedback")

        per_token_kl, reward, avg_kl, entropy = compute_sdpo_loss(
            model, tokenizer, config, base_prompt, teacher_prompt, generated_ids
        )
        loss = per_token_kl.mean()
        optimizer.zero_grad()
        loss.backward()

        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_val = loss.item()

        print(f"  Loss: {loss_val:.4f}, Reward: {reward:.4f}, KL: {avg_kl:.4f}")
        print(f"  Entropy: {entropy:.4f}, Grad norm: {grad_norm:.4f}")

        iter_log["teacher_prompt"] = teacher_prompt
        iter_log["loss"] = loss_val
        iter_log["reward"] = reward
        iter_log["kl_div"] = avg_kl
        iter_log["entropy"] = entropy
        iter_log["grad_norm"] = grad_norm
        iter_log["feedback"] = feedback
        # Stored here; save_run strips this from logs.json and writes it to kl/ separately
        iter_log["per_token_kl"] = collect_per_token_kl(per_token_kl, generated_ids, tokenizer)
        logs["iteration_logs"].append(iter_log)
        metrics["iterations"].append(iteration + 1)
        metrics["losses"].append(loss_val)
        metrics["rewards"].append(reward)
        metrics["kl_divs"].append(avg_kl)
        metrics["entropies"].append(entropy)
        metrics["grad_norms"].append(grad_norm)
        metrics["timestamps"].append(time.time() - iter_start)

    logs["end_time"] = datetime.now().isoformat()
    logs["success"] = best_proof is not None
    logs["best_proof"] = best_proof
    logs["metrics"] = metrics
    logs["total_generation_tokens"] = sum(e.get("num_tokens", 0) for e in logs["iteration_logs"])
    if config_dict_for_logs is not None:
        logs["config"] = config_dict_for_logs

    model_save_path = save_run(
        output_root=output_root,
        config=config,
        logs=logs,
        metrics=metrics,
        model=model,
        tokenizer=tokenizer,
    )
    logs["model_save_path"] = model_save_path

    return logs
