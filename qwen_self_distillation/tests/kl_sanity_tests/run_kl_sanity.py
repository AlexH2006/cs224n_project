"""
TLDR: Gradient-free KL divergence sanity check — local HuggingFace only, no Modal/vLLM.

Reads prompts from JSON, runs forward passes with torch.no_grad(), computes per-token
KL(student || teacher), and saves context, per-token KL, and heatmap visualization.

Supported models (shorthand): 4B, 9B (Qwen3.5), goedel8b (Goedel-Prover-V2-8B).

Design: Full mode only. All computation under no_grad() to minimize memory on laptop.
Uses same logic as qwen_sdpo.sdpo_loss but never enables gradients.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Tail bucket (local copy — no grad dependencies)
# -----------------------------------------------------------------------------


def _add_tail(log_probs: "torch.Tensor") -> "torch.Tensor":
    """Append tail bucket for KL: log of probability mass outside top-K.

    Mirrors qwen_sdpo.sdpo_loss.add_tail. Ensures KL covers full distribution.
    """
    import torch

    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


# -----------------------------------------------------------------------------
# Gradient-free KL computation
# -----------------------------------------------------------------------------


def _compute_kl_no_grad(
    model: Any,
    tokenizer: Any,
    distillation_topk: int,
    student_prompt: str,
    teacher_prompt: str,
    generated_ids: "torch.Tensor",
) -> "torch.Tensor":
    """Compute per-token KL(student || teacher) under torch.no_grad().

    Full mode only: both student and teacher see prompt + full solution.
    Mirrors qwen_sdpo.sdpo_loss.compute_sdpo_loss logic for full mode,
    but never enables gradients (suitable for laptop memory).

    Returns:
        per_token_kl: 1-D tensor [seq_len] of per-token KL values.
    """
    import torch
    import torch.nn.functional as F

    model_device = next(model.parameters()).device
    K = distillation_topk

    # Tokenize prompts (truncation matches sdpo_loss)
    student_prompt_ids = tokenizer(
        student_prompt, return_tensors="pt", truncation=True, max_length=2048
    ).input_ids.to(model_device)
    teacher_prompt_ids = tokenizer(
        teacher_prompt, return_tensors="pt", truncation=True, max_length=2048
    ).input_ids.to(model_device)

    response_ids = generated_ids.to(model_device)
    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)

    # Full mode: concatenate prompt + response for both
    student_input_ids = torch.cat([student_prompt_ids, response_ids], dim=1)
    teacher_input_ids = torch.cat([teacher_prompt_ids, response_ids], dim=1)

    student_prompt_len = student_prompt_ids.shape[1]
    teacher_prompt_len = teacher_prompt_ids.shape[1]
    seq_len = response_ids.shape[1]

    # Causal LM: logits[i] predicts input_ids[i+1]
    student_start = student_prompt_len - 1
    student_end = student_prompt_len - 1 + seq_len
    teacher_start = teacher_prompt_len - 1
    teacher_end = teacher_prompt_len - 1 + seq_len

    with torch.no_grad():
        student_logits = model(input_ids=student_input_ids).logits[
            0, student_start:student_end
        ]
        teacher_logits = model(input_ids=teacher_input_ids).logits[
            0, teacher_start:teacher_end
        ]

    n_student = student_logits.shape[0]
    n_teacher = teacher_logits.shape[0]
    if n_student != n_teacher:
        raise ValueError(
            f"Student/teacher logit length mismatch: {n_student} vs {n_teacher}"
        )

    K_actual = min(K, student_logits.size(-1))
    student_topk_logits, topk_indices = torch.topk(student_logits, K_actual, dim=-1)
    student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    student_topk_logps = student_topk_logits - student_logsumexp

    with torch.no_grad():
        teacher_topk_logits = torch.gather(
            teacher_logits, dim=-1, index=topk_indices
        )
        teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
        teacher_topk_logps = teacher_topk_logits - teacher_logsumexp

    student_with_tail = _add_tail(student_topk_logps)
    teacher_with_tail = _add_tail(teacher_topk_logps)

    kl_per_bucket = F.kl_div(
        teacher_with_tail,
        student_with_tail,
        reduction="none",
        log_target=True,
    )
    per_token_kl = kl_per_bucket.sum(dim=-1)

    return per_token_kl


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def _load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen3.5-4B",
) -> tuple[Any, Any]:
    """Load 4-bit quantized model and tokenizer (no LoRA)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer


# -----------------------------------------------------------------------------
# JSON I/O
# -----------------------------------------------------------------------------

_REQUIRED_KEYS = frozenset({"student_prompt", "teacher_prompt", "generated_solution"})

_MODEL_ALIASES = {
    "4B": "Qwen/Qwen3.5-4B",
    "9B": "Qwen/Qwen3.5-9B",
    "goedel8b": "Goedel-LM/Goedel-Prover-V2-8B",
}


def _resolve_model(name: str) -> str:
    """Resolve shorthand (4B, 9B, goedel8b) to full HuggingFace model path."""
    return _MODEL_ALIASES.get(name, name)


def _load_prompts(path: Path) -> dict[str, str]:
    """Load and validate prompts JSON. Raises on missing keys or invalid file."""
    with open(path, "r") as f:
        data = json.load(f)
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys in {path}: {sorted(missing)}")
    return {
        "student_prompt": str(data["student_prompt"]),
        "teacher_prompt": str(data["teacher_prompt"]),
        "generated_solution": str(data["generated_solution"]),
    }


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def run_kl_sanity_check(
    input_path: Path,
    output_base_dir: Path | None = None,
    model_name: str | None = None,
    distillation_topk: int = 20,
) -> Path:
    """Run the full KL sanity pipeline: load prompts, compute KL, save artifacts.

    Args:
        input_path: Path to JSON with student_prompt, teacher_prompt, generated_solution.
        output_base_dir: Directory for run_{timestamp}/. Default: same dir as this script.
        model_name: HuggingFace model name.
        distillation_topk: Top-K for KL computation.

    Returns:
        Path to the run directory (run_{timestamp}/).
    """
    import torch

    from qwen_sdpo.results import collect_per_token_kl, plot_token_kl_heatmap

    if output_base_dir is None:
        output_base_dir = Path(__file__).resolve().parent

    prompts = _load_prompts(input_path)
    # Model: model_name param overrides; else "model" from JSON; else 4B
    with open(input_path) as f:
        raw = json.load(f)
    resolved_model = _resolve_model(
        model_name if model_name is not None else raw.get("model", "Qwen/Qwen3.5-4B")
    )
    student_prompt = prompts["student_prompt"]
    teacher_prompt = prompts["teacher_prompt"]
    generated_solution = prompts["generated_solution"]

    model, tokenizer = _load_model_and_tokenizer(resolved_model)

    # Tokenize solution (add_special_tokens=False: continuation, not new sequence)
    gen_ids = tokenizer(
        generated_solution,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    if gen_ids.numel() == 0:
        raise ValueError(
            "generated_solution tokenizes to empty sequence; cannot compute KL"
        )

    per_token_kl = _compute_kl_no_grad(
        model=model,
        tokenizer=tokenizer,
        distillation_topk=distillation_topk,
        student_prompt=student_prompt,
        teacher_prompt=teacher_prompt,
        generated_ids=gen_ids,
    )

    records = collect_per_token_kl(per_token_kl, gen_ids, tokenizer)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Output 1: context (prefilled prompts + solution + model used)
    context_path = run_dir / "context.json"
    with open(context_path, "w") as f:
        json.dump(
            {
                "model": resolved_model,
                "student_prompt": student_prompt,
                "teacher_prompt": teacher_prompt,
                "generated_solution": generated_solution,
            },
            f,
            indent=2,
        )

    # Output 2: per-token KL
    kl_path = run_dir / "per_token_kl.json"
    with open(kl_path, "w") as f:
        json.dump(records, f, indent=2)

    # Output 3: heatmap
    heatmap_path = run_dir / "kl_heatmap.png"
    plot_token_kl_heatmap(records, heatmap_path)

    print(f"Run saved to: {run_dir}")
    return run_dir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KL divergence sanity check for Qwen SDPO (gradient-free)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to JSON with student_prompt, teacher_prompt, generated_solution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base directory for run_{timestamp}/. Default: kl_sanity_tests/",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name or shorthand (4B, 9B, goedel8b). Omit to use 'model' from JSON.",
    )
    parser.add_argument(
        "--distillation-topk",
        type=int,
        default=20,
        help="Top-K for KL computation",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent

    run_kl_sanity_check(
        input_path=args.input,
        output_base_dir=output_dir,
        model_name=args.model,
        distillation_topk=args.distillation_topk,
    )


if __name__ == "__main__":
    main()
