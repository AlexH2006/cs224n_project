"""
SDPO loss: KL divergence between student and teacher with tail bucket.

TLDR: add_tail appends a tail bucket for top-K KL; compute_sdpo_loss returns
per_token_kl, reward, avg_kl, entropy. Used by: trainer_core.
"""

from typing import TYPE_CHECKING

from sdpo_modal_local_verify_kimina.config import SDPOConfig

if TYPE_CHECKING:
    import torch


def add_tail(log_probs: "torch.Tensor") -> "torch.Tensor":
    """Append tail bucket for KL computation (log of mass outside top-K)."""
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
) -> tuple["torch.Tensor", float, float, float]:
    """Compute SDPO loss: KL(student || teacher) on top-K tokens; return per_token_kl, reward, avg_kl, entropy."""
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

    student_input_ids = torch.cat([student_prompt_ids, response_ids], dim=1)
    teacher_input_ids = torch.cat([teacher_prompt_ids, response_ids], dim=1)

    student_prompt_len = student_prompt_ids.shape[1]
    teacher_prompt_len = teacher_prompt_ids.shape[1]
    seq_len = response_ids.shape[1]

    student_logits = model(input_ids=student_input_ids).logits[
        0, student_prompt_len - 1 : student_prompt_len - 1 + seq_len
    ]

    logsumexp_vals = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    log_probs = student_logits - logsumexp_vals
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean().item()

    K_actual = min(K, student_logits.size(-1))
    student_topk_logits, topk_indices = torch.topk(student_logits, K_actual, dim=-1)
    student_logsumexp = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    student_topk_logps = student_topk_logits - student_logsumexp

    with torch.no_grad():
        teacher_logits = model(input_ids=teacher_input_ids).logits[
            0, teacher_prompt_len - 1 : teacher_prompt_len - 1 + seq_len
        ]
        teacher_topk_logits = torch.gather(teacher_logits, dim=-1, index=topk_indices)
        teacher_logsumexp = torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
        teacher_topk_logps = teacher_topk_logits - teacher_logsumexp

    student_with_tail = add_tail(student_topk_logps)
    teacher_with_tail = add_tail(teacher_topk_logps)

    kl_per_bucket = F.kl_div(
        teacher_with_tail.detach(),
        student_with_tail,
        reduction="none",
        log_target=True,
    )
    per_token_kl = kl_per_bucket.sum(dim=-1)

    with torch.no_grad():
        target_ids = response_ids[0]
        student_lp = -F.cross_entropy(student_logits.detach(), target_ids, reduction="none")
        teacher_lp = -F.cross_entropy(teacher_logits, target_ids, reduction="none")
        total_reward = (student_lp - teacher_lp).sum().item()
        avg_kl = per_token_kl.mean().item()

    return per_token_kl, total_reward, avg_kl, entropy
