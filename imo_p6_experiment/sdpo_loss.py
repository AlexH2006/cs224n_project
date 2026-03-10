"""
SDPO loss: KL divergence between student and teacher with tail bucket.

TLDR: add_tail appends a tail bucket for top-K KL; compute_sdpo_loss returns
per_token_kl, reward, avg_kl, entropy. Used by: trainer_core.

When proof_token_range is provided, KL / reward / entropy are computed only
over the Lean proof tokens (not the reasoning prefix).
"""

from typing import TYPE_CHECKING, Optional

from sdpo_modal_local_verify_qwen.config import SDPOConfig

if TYPE_CHECKING:
    import torch


def get_proof_token_range(
    raw_output: str,
    extracted_block: str,
    tokenizer,
) -> Optional[tuple[int, int]]:
    """Map the extracted Lean block back to (start, end) token indices in the full response.

    Tokenizes the text *before* the block and the block itself (without special tokens)
    to approximate where the proof lives inside generated_ids.
    Returns None if the block is empty or not found in raw_output.
    """
    if not extracted_block or not extracted_block.strip():
        return None
    block_start_char = raw_output.find(extracted_block)
    if block_start_char < 0:
        return None
    prefix_ids = tokenizer(
        raw_output[:block_start_char], add_special_tokens=False
    ).input_ids
    block_ids = tokenizer(extracted_block, add_special_tokens=False).input_ids
    if not block_ids:
        return None
    return (len(prefix_ids), len(prefix_ids) + len(block_ids))


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
    proof_token_range: Optional[tuple[int, int]] = None,
) -> tuple["torch.Tensor", float, float, float]:
    """Compute SDPO loss: KL(student || teacher) on top-K tokens.

    If proof_token_range is given, only the tokens in [start, end) (the Lean
    proof) contribute to the returned per_token_kl, reward, avg_kl, and entropy.
    The full response is still fed through the model so that autoregressive
    conditioning is correct; only the *aggregation* is restricted.
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

    student_input_ids = torch.cat([student_prompt_ids, response_ids], dim=1)
    teacher_input_ids = torch.cat([teacher_prompt_ids, response_ids], dim=1)

    student_prompt_len = student_prompt_ids.shape[1]
    teacher_prompt_len = teacher_prompt_ids.shape[1]
    seq_len = response_ids.shape[1]

    student_logits = model(input_ids=student_input_ids).logits[
        0, student_prompt_len - 1 : student_prompt_len - 1 + seq_len
    ]

    # Validate / clamp proof_token_range to actual sequence length.
    if proof_token_range is not None:
        ps, pe = proof_token_range
        pe = min(pe, seq_len)
        ps = max(0, min(ps, pe))
        if ps >= pe:
            proof_token_range = None
        else:
            proof_token_range = (ps, pe)

    logsumexp_vals = torch.logsumexp(student_logits, dim=-1, keepdim=True)
    log_probs = student_logits - logsumexp_vals
    if proof_token_range is not None:
        ps, pe = proof_token_range
        entropy = -(log_probs[ps:pe].exp() * log_probs[ps:pe]).sum(dim=-1).mean().item()
    else:
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
        if proof_token_range is not None:
            ps, pe = proof_token_range
            total_reward = (student_lp[ps:pe] - teacher_lp[ps:pe]).sum().item()
            avg_kl = per_token_kl[ps:pe].mean().item()
        else:
            total_reward = (student_lp - teacher_lp).sum().item()
            avg_kl = per_token_kl.mean().item()

    # Slice per_token_kl to proof tokens so callers' .mean() computes loss
    # only over the Lean proof, not the reasoning prefix.
    if proof_token_range is not None:
        ps, pe = proof_token_range
        per_token_kl = per_token_kl[ps:pe]

    return per_token_kl, total_reward, avg_kl, entropy
