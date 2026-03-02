# GPU Configuration Notes for SDPO Training

**Date:** 2026-02-20  
**Topics:** gpu, config, sdpo

---

This document tracks known issues and configurations for running SDPO test-time training on different GPU configurations.

## A100-40GB (Default) - For 1-2B Models

**Status**: ✅ Working (Verified 2026-02-20)

**Recommended Models**:
- `AI-MO/Kimina-Prover-RL-1.7B` (default)

**Configuration**:
- `gpu_memory_utilization`: 0.25
- `max_model_len`: 4096
- Full precision (bfloat16) for both vLLM and HuggingFace model

**Verified Behavior**:
- vLLM engine initializes successfully (~3.2 GiB model, ~5.2 GiB KV cache)
- HuggingFace model loads with gradient checkpointing
- SDPO training loop runs with proper gradient flow (grad norms ~5-8)
- Model weights saved successfully after training

**Note**: There's a warning `None of the inputs have requires_grad=True` from gradient checkpointing, but gradients are still computed correctly (verified by non-zero grad norms).

**Command**:
```bash
modal run training/lean_sdpo_modal.py \
  --dataset "amitayusht/PutnamBench" \
  --model "AI-MO/Kimina-Prover-RL-1.7B" \
  --dataset-split "train" \
  --problem-idx 0 \
  --max-iterations 5
```

---

## A100-80GB - For 7-8B Models

**Status**: ⚠️ Partial - Memory constraints with full SDPO training

**Target Models**:
- `Goedel-LM/Goedel-Prover-V2-8B`

**Configuration**:
- `gpu_memory_utilization`: 0.30
- `max_model_len`: 4096

### Known Issues

#### 1. OOM During Optimizer Step (Full Precision)
**Error**: `OutOfMemoryError: CUDA out of memory` during `optimizer.step()`

**Cause**: Loading both vLLM engine (~24GB) and HuggingFace model (~16GB) + optimizer states (~32GB for Adam) exceeds 80GB.

**Memory breakdown for 8B model**:
- vLLM engine: ~24GB (at 30% utilization)
- HF model weights: ~16GB (bfloat16)
- Adam optimizer states: ~32GB (2x model size for momentum + variance)
- Activations/gradients: ~8GB+
- **Total**: >80GB

#### 2. NaN Loss with 8-bit Quantization
**Error**: Loss becomes `nan` after first iteration

**Cause**: 8-bit quantized models via bitsandbytes don't properly support gradient computation for all layers. The quantized weights don't have `requires_grad=True`.

#### 3. LoRA + 4-bit Quantization (Current Approach)
**Status**: Testing in progress

**Approach**: Use 4-bit quantization with LoRA adapters for trainable parameters.

**Potential Issues**:
- LoRA only trains a small subset of parameters
- May not be sufficient for SDPO's KL distillation objective
- Need to verify gradient flow through quantized base model

### Recommendations for 8B Models

1. **Use H100 GPU** (if available): 80GB HBM3 with better memory bandwidth
2. **Use inference-only mode**: Skip SDPO training, just evaluate model
3. **Use smaller models**: Stick with 1-2B models for full SDPO training
4. **Gradient accumulation**: Reduce batch size and accumulate gradients (not yet implemented)

---

## H100 - For Large Models (Not Yet Tested)

**Status**: 🔧 Configuration defined but not tested

**Configuration**:
- `gpu_memory_utilization`: 0.50
- `max_model_len`: 8192

---

## Command Reference

```bash
# Default (A100-40GB, Kimina 1.7B)
modal run training/lean_sdpo_modal.py \
  --dataset "amitayusht/PutnamBench" \
  --model "AI-MO/Kimina-Prover-RL-1.7B" \
  --problem-idx 0 \
  --max-iterations 5

# A100-80GB with Goedel 8B (experimental)
modal run training/lean_sdpo_modal.py \
  --dataset "amitayusht/PutnamBench" \
  --model "Goedel-LM/Goedel-Prover-V2-8B" \
  --problem-idx 0 \
  --max-iterations 3 \
  --gpu A100-80GB
```

---

## Changelog

- **2026-02-20**: Initial documentation
  - Documented A100-80GB OOM issues with 8B models
  - Tested 8-bit quantization (NaN loss issue)
  - Added LoRA + 4-bit quantization approach (in progress)
  - ✅ Verified A100-40GB works with Kimina 1.7B
    - Tested on PutnamBench problems 0 and 5
    - SDPO training loop completes successfully
    - Gradient flow verified (grad norms ~5-8)
    - Model weights saved correctly
