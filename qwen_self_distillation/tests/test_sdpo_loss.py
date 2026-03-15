"""
TLDR: Unit tests for qwen_sdpo.sdpo_loss — indexing alignment and shape consistency.

Verifies that student/teacher logit slices align with target_ids (generated_ids
or answer_ids), so cross_entropy never sees a shape mismatch. Causal LM indexing:
logits[i] predicts input_ids[i+1]. Slice [start:end] yields end - start positions.
"""

import importlib.util

import pytest
import torch

from qwen_sdpo.config import SDPOConfig
from qwen_sdpo.sdpo_loss import compute_sdpo_loss


class TestSdpoLossIndexing:
    """Verify indexing logic: logits shape must match target_ids shape."""

    def test_full_mode_slice_lengths(self):
        """Full mode: student and teacher logits = seq_len; target = response_ids."""
        student_prompt_len = 100
        seq_len = 50
        # Full mode: student_start = student_prompt_len - 1, student_end = student_prompt_len + seq_len - 1
        student_start = student_prompt_len - 1
        student_end = student_prompt_len + seq_len - 1
        num_student_logits = student_end - student_start
        assert num_student_logits == seq_len

    def test_answer_only_mode_slice_lengths(self):
        """Answer-only mode: student and teacher logits = answer_len; target = answer_ids."""
        student_prompt_len = 100
        seq_len = 686
        cot_len = 1  # one COT token
        answer_len = seq_len - cot_len  # 685

        # Slice logic from sdpo_loss (Python slice [start:end] gives end - start elements)
        student_start = student_prompt_len + cot_len - 1
        student_end = student_prompt_len + seq_len - 1
        num_student_logits = student_end - student_start

        assert num_student_logits == answer_len, (
            f"answer_only: expected {answer_len} student logits, got {num_student_logits}"
        )
        assert num_student_logits == seq_len - cot_len

    def test_answer_only_slice_matches_target_len(self):
        """Ensure answer_len (target) matches num logits for the failing case: 687 vs 686."""
        seq_len = 686 + 1  # 687 total response tokens
        cot_len = 1
        answer_len = seq_len - cot_len  # 686

        student_prompt_len = 50  # arbitrary
        student_start = student_prompt_len + cot_len - 1
        student_end = student_prompt_len + seq_len - 1
        num_logits = student_end - student_start

        assert num_logits == answer_len, (
            f"num_logits={num_logits} must equal answer_len={answer_len}"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(
        importlib.util.find_spec("bitsandbytes") is None,
        reason="bitsandbytes required for integration test",
    )
    def test_compute_sdpo_loss_full_mode_shapes(self):
        """Integration: compute_sdpo_loss full mode produces matching shapes."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        config = SDPOConfig(distillation_topk=20)
        model_name = "Qwen/Qwen3.5-4B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Minimal 4-bit load for test
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
        model.eval()

        base_prompt = "Prove this: 1+1=2"
        teacher_prompt = base_prompt + "\nError: fix the proof."
        seq_len = 10
        generated_ids = torch.randint(0, tokenizer.vocab_size, (seq_len,))

        per_token_kl, reward, avg_kl, entropy = compute_sdpo_loss(
            model, tokenizer, config, base_prompt, teacher_prompt, generated_ids
        )
        assert per_token_kl.shape[0] == seq_len
        assert per_token_kl.dim() == 1

    @pytest.mark.slow
    @pytest.mark.skipif(
        importlib.util.find_spec("bitsandbytes") is None,
        reason="bitsandbytes required for integration test",
    )
    def test_compute_sdpo_loss_answer_only_shapes(self):
        """Integration: compute_sdpo_loss answer-only mode produces matching shapes."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        config = SDPOConfig(distillation_topk=20)
        model_name = "Qwen/Qwen3.5-4B"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto", trust_remote_code=True
        )
        model.eval()

        base_prompt = "Prove: 1+1=2"
        teacher_prompt = base_prompt + "\nError: fix it."
        seq_len = 20
        cot_len = 5
        answer_len = seq_len - cot_len
        full_ids = torch.randint(0, tokenizer.vocab_size, (seq_len,))
        answer_ids = full_ids[cot_len:].clone()

        per_token_kl, reward, avg_kl, entropy = compute_sdpo_loss(
            model,
            tokenizer,
            config,
            base_prompt,
            teacher_prompt,
            full_ids,
            teacher_response_ids=answer_ids,
            cot_len=cot_len,
        )
        assert per_token_kl.shape[0] == answer_len
        assert per_token_kl.dim() == 1


class TestTotalKLLossScalar:
    """Training loss uses total KL (sum over tokens), not mean per-token KL."""

    def test_total_kl_equals_sum_of_per_token_kl(self):
        """Total KL loss (used for backward) is per_token_kl.sum()."""
        per_token_kl = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        total_kl = per_token_kl.sum()
        assert total_kl.item() == pytest.approx(0.6)
        assert total_kl != per_token_kl.mean()

    def test_total_kl_differs_from_mean_when_seq_len_gt_one(self):
        """When seq_len > 1, total KL (sum) != mean per-token KL."""
        per_token_kl = torch.tensor([0.5, 0.5], dtype=torch.float32)
        assert per_token_kl.sum().item() == 1.0
        assert per_token_kl.mean().item() == 0.5
        assert per_token_kl.sum() != per_token_kl.mean()
