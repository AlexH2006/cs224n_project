"""
TLDR: SDPO loss — KL divergence between student and teacher distributions with tail bucket.

The loss is computed on the top-K token logits plus a tail bucket that captures the
remaining probability mass, ensuring the KL covers the full distribution without the
O(vocab_size) cost of full KL.

Logic is model-agnostic: same function works for Qwen3.5-4B, Qwen3.5-9B, and any
other causal LM. No architecture-specific code here.

Ported from sdpo_modal_local_verify_kimina/sdpo_loss.py — no logic changes.

Used by: modal_trainer.py (QwenSDPOTrainer.run_sdpo_step).
"""

from typing import TYPE_CHECKING, Optional

from qwen_sdpo.config import SDPOConfig

if TYPE_CHECKING:
    import torch


def add_tail(log_probs: "torch.Tensor") -> "torch.Tensor":
    """Append a tail bucket for KL computation (log of probability mass outside top-K).

    The tail bucket ensures that the KL divergence sums to the full KL, not just the
    top-K approximation. Without this, probability mass outside top-K is silently ignored,
    which underestimates divergence when student and teacher disagree on rare tokens.
    """
    import torch

    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def compute_sdpo_loss(
    model,
    tokenizer,
    config: SDPOConfig,
    base_prompt: str,
    teacher_prompt: str,
    generated_ids: "torch.Tensor",
    *,
    teacher_response_ids: Optional["torch.Tensor"] = None,
    cot_len: Optional[int] = None,
) -> tuple["torch.Tensor", float, float, float]:
    """Compute SDPO loss: KL(student || teacher) on top-K tokens.

    Both student and teacher use the same model weights; the difference is prompt
    conditioning. When teacher_response_ids and cot_len are provided (answer-only
    mode), teacher sees only answer tokens and KL is computed over those; otherwise
    full generation is used for both.

    Args:
        model:                 HuggingFace CausalLM with LoRA adapters (on GPU).
        tokenizer:             HuggingFace tokenizer matching the model.
        config:                SDPOConfig (uses distillation_topk).
        base_prompt:           Student prompt string (problem only).
        teacher_prompt:        Teacher prompt string (problem + error feedback).
        generated_ids:         1-D or 2-D (1, seq_len) tensor of full generated IDs.
        teacher_response_ids:  Optional. When set, teacher receives only these
                              (answer tokens). Must be used with cot_len.
        cot_len:               Optional. Number of COT tokens in generated_ids.
                              When set with teacher_response_ids, KL over answer only.

    Returns:
        (per_token_kl, total_reward, avg_kl, entropy)
          per_token_kl: [seq_len] tensor of per-token KL values (differentiable).
          total_reward: scalar float — log-likelihood gain over teacher baseline.
          avg_kl:       scalar float — mean KL across generated tokens.
          entropy:      scalar float — student policy entropy (top-K + tail approx).
    """
    import torch
    import torch.nn.functional as F

    K = config.distillation_topk
    model_device = next(model.parameters()).device

    student_prompt_ids = tokenizer(
        base_prompt, return_tensors="pt", truncation=True, max_length=2048
    ).input_ids.to(model_device)

    teacher_prompt_ids = tokenizer(
        teacher_prompt, return_tensors="pt", truncation=True, max_length=2048
    ).input_ids.to(model_device)

    response_ids = generated_ids.to(model_device)
    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)

    # When teacher_response_ids and cot_len are provided, use the slice path so student
    # and teacher logits align over the same segment (answer_only or code_only).
    # Allow cot_len >= 0 so code_only with code at start still uses slice path.
    use_answer_only = (
        teacher_response_ids is not None
        and cot_len is not None
        and teacher_response_ids.numel() > 0
    )

    if use_answer_only:
        teacher_resp = teacher_response_ids.to(model_device)
        if teacher_resp.dim() == 1:
            teacher_resp = teacher_resp.unsqueeze(0)
        teacher_input_ids = torch.cat([teacher_prompt_ids, teacher_resp], dim=1)
        answer_len = teacher_resp.shape[1]
    else:
        teacher_input_ids = torch.cat([teacher_prompt_ids, response_ids], dim=1)
        answer_len = response_ids.shape[1]

    student_input_ids = torch.cat([student_prompt_ids, response_ids], dim=1)
    student_prompt_len = student_prompt_ids.shape[1]
    teacher_prompt_len = teacher_prompt_ids.shape[1]
    seq_len = response_ids.shape[1]

    if use_answer_only:
        # Causal LM: logits[i] predicts input_ids[i+1]. We want logits for the same
        # segment length on both sides. Use answer_len = len(teacher_response_ids) so
        # that code_only (where the slice may not extend to end of generation) matches.
        student_start = student_prompt_len + cot_len - 1
        student_end = student_start + answer_len  # exclusive; yields answer_len logits
        teacher_start = teacher_prompt_len - 1
        teacher_end = teacher_prompt_len + answer_len - 1  # exclusive; yields answer_len logits
    else:
        student_start = student_prompt_len - 1
        student_end = student_prompt_len - 1 + seq_len
        teacher_start = teacher_prompt_len - 1
        teacher_end = teacher_prompt_len - 1 + seq_len

    # Student forward pass (gradients flow through this).
    student_logits = model(input_ids=student_input_ids).logits[
        0, student_start:student_end
    ]
    with torch.no_grad():
        teacher_logits = model(input_ids=teacher_input_ids).logits[
            0, teacher_start:teacher_end
        ]
    # Both must have the same sequence length for KL (and for gather with topk_indices).
    n_student = student_logits.shape[0]
    n_teacher = teacher_logits.shape[0]
    if n_student != n_teacher:
        raise ValueError(
            f"Student and teacher logit sequence length mismatch: student={n_student}, teacher={n_teacher}. "
            f"cot_len={cot_len}, answer_len={answer_len if use_answer_only else 'N/A'}, seq_len={seq_len}."
        )

    K_actual = min(K, student_logits.size(-1))
    student_topk_logits, topk_indices = torch.topk(student_logits, K_actual, dim=-1)
    student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    student_topk_logps = student_topk_logits - student_logsumexp

    # Teacher forward pass (no gradients).
    with torch.no_grad():
        teacher_topk_logits = torch.gather(teacher_logits, dim=-1, index=topk_indices)
        teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
        teacher_topk_logps = teacher_topk_logits - teacher_logsumexp

    student_with_tail = add_tail(student_topk_logps)
    teacher_with_tail = add_tail(teacher_topk_logps)

    entropy = -(student_with_tail.exp() * student_with_tail).sum(dim=-1).mean().item()

    kl_per_bucket = F.kl_div(
        teacher_with_tail.detach(),
        student_with_tail,
        reduction="none",
        log_target=True,
    )
    per_token_kl = kl_per_bucket.sum(dim=-1)

    with torch.no_grad():
        if use_answer_only:
            target_ids = teacher_resp[0]
        else:
            target_ids = response_ids[0]
        student_lp = -F.cross_entropy(student_logits.detach(), target_ids, reduction="none")
        teacher_lp = -F.cross_entropy(teacher_logits, target_ids, reduction="none")
        total_reward = (student_lp - teacher_lp).sum().item()
        avg_kl = per_token_kl.mean().item()

    return per_token_kl, total_reward, avg_kl, entropy
