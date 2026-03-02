# Generation Speed — Current Engineering Specs & Optimization Options

**Date:** 2026-02-28  
**Topics:** vllm, generation, performance

---

This document captures the current vLLM/inference configuration in `lean_sdpo_goedel_8b_modal.py` that affects **token generation speed**, and proposes changes to improve throughput.

**Timing:** Per-step timings (setup + each iteration) are now recorded; see [20260301_timing_analysis.md](20260301_timing_analysis.md) for a one-iteration run and bottleneck analysis.

---

## 1. Current vLLM Engine Configuration

| Setting | Current value | Effect on speed |
|--------|----------------|-----------------|
| **Engine** | Legacy V0 (`VLLM_USE_V1=0`) | V1 would use torch.compile (3+ min first run) but can be faster once warm. We force V0 for predictable startup. |
| **`enforce_eager`** | **`False`** | **CUDA graphs enabled.** First request may take ~20–60s to capture; decode is ~2–4× faster (e.g. ~73 tok/s observed). |
| **`gpu_memory_utilization`** | **0.4** | vLLM gets ~32 GB on an 80 GB GPU. Rest is left for Unsloth (4-bit + LoRA + optimizer). |
| **`max_model_len`** | **10240** | Max sequence length (prompt + generated). Drives KV cache size. |
| **`max_num_seqs`** | **1** | Only one sequence in flight. No batching; no opportunity to hide latency with concurrent sequences. |
| **`enable_lora`** | True | Required for SDPO; LoRA adapter overlay has some overhead. |
| **`max_lora_rank`** | 16 | Matches training; affects LoRA matmul cost. |
| **`dtype`** | bfloat16 | No quantization for vLLM (keeps quality; fp8 would speed up at cost of precision). |

**Relevant code** (in `_setup_trainer`):

```python
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

trainer_self.vllm_engine = LLM(
    model=trainer_self.model_name,
    dtype="bfloat16",
    gpu_memory_utilization=gpu_memory_utilization,  # 0.4
    max_model_len=max_model_len,                     # 10240
    max_num_seqs=1,
    enable_lora=True,
    max_lora_rank=lora_rank,
    enforce_eager=True,   # <-- no CUDA graphs
)
```

---

## 2. FlashInfer / Custom Sampler

- **`VLLM_USE_FLASHINFER_SAMPLER`** is set to **`"0"`** so vLLM does **not** use FlashInfer-based sampling.
- **Reason:** FlashInfer (and building it) can depend on `nvcc`; the Modal image is `debian_slim` without a CUDA toolkit, so we disabled it to avoid build/runtime errors.
- **Effect:** We fall back to vLLM's default (PyTorch) sampling path, which is slower than fused FlashInfer kernels.

---

## 3. Sampling / Generation Parameters

| Parameter | Value | Effect |
|-----------|--------|--------|
| **`max_tokens`** | **8192** (`config.max_new_tokens`) | Each request can generate up to 8K tokens. Long runs amplify the cost of slow decode (no CUDA graphs). |
| **`temperature`** | 0.6 | Non-zero → sampling, not greedy; same kernel path. |
| **`top_p`** | 0.95 | Nucleus sampling. |
| **`stop`** | List of EOS tokens | Early stop can shorten generation; no impact on per-token speed. |

Generation is **single-prompt**: `generate([prompt], sampling_params, lora_request=...)` — no batching.

---

## 4. Why These Choices Were Made

- **`enforce_eager=True`:** Avoid CUDA graph capture to reduce **peak memory** and avoid OOM when sharing the GPU with Unsloth. Eager also avoids graph capture time and potential failures with LoRA.
- **`gpu_memory_utilization=0.35`:** Leave enough GPU memory for the 4-bit Unsloth model, LoRA, optimizer state, and activations so training doesn't OOM.
- **`max_num_seqs=1`:** SDPO loop is single-problem; no need for multi-sequence scheduling. Keeps memory predictable.
- **FlashInfer off:** Image doesn't provide `nvcc`; turning it off avoids build and runtime issues.

So the current setup **trades off decode speed for stability and memory safety** on one A100-80GB with Unsloth + vLLM.

---

## 5. What You're Likely Seeing

- **If the metric is "27 tokens / minute"** (0.45 tok/s): That's pathologically slow. Possible causes: CPU-bound post-processing, very long prompt processing, or a run where vLLM was still compiling/loading. Check logs for "est. speed output: X toks/s" to see vLLM's reported decode rate.
- **If the metric is "~27 tokens / second"** (e.g. "28.05 toks/s" in logs): That's consistent with **eager mode + no CUDA graphs + single sequence**. For 8192 max tokens, 8K / 27 ≈ **5+ minutes** per full generation, which matches "really slow" for interactive use.

---

## 6. Proposed Optimizations (by impact vs risk)

### High impact, try first

1. **Turn off eager mode (enable CUDA graphs)**  
   - Set **`enforce_eager=False`** (or remove the argument and use vLLM default).  
   - **Risk:** Higher peak memory during capture; possible OOM with Unsloth.  
   - **Mitigation:** Temporarily raise `gpu_memory_utilization` a bit (e.g. 0.38–0.40) if the run fits, or reduce `max_model_len` (e.g. 8192) to free KV cache.  
   - **Expected:** **2–4×** higher decode throughput (e.g. 27 → 60–100+ tok/s in favorable cases).

2. **Lower `max_new_tokens` for SDPO**  
   - Many proofs finish well before 8K tokens. Cap at **4096** or **2048** for experiments; increase again if you see truncation.  
   - **Effect:** Same per-token speed, but **shorter wall-clock time** per call and less KV cache pressure.

3. **Reduce `max_model_len`**  
   - If prompt + 4096 (or 2048) is enough, set **`max_model_len=6144`** or **5120**.  
   - **Effect:** Smaller KV cache, more room for CUDA graphs and/or higher `gpu_memory_utilization`.

### Medium impact, more setup

4. **Enable FlashInfer (fused sampling)**  
   - Use a **custom Modal image** that includes CUDA toolkit and builds **FlashInfer** (or use a pre-built wheel if available).  
   - Set **`VLLM_USE_FLASHINFER_SAMPLER=1`** (or vLLM's current env for FlashInfer).  
   - **Effect:** Faster sampling and potentially better memory use; exact gain depends on vLLM version and hardware.

5. **Try vLLM V1 engine**  
   - Set **`VLLM_USE_V1=1`** and accept **longer first-request latency** (torch.compile).  
   - **Effect:** After warmup, V1 can be faster; measure end-to-end for 1–2 full SDPO iterations.

6. **Slightly increase vLLM memory**  
   - If OOM is not observed, try **`gpu_memory_utilization=0.38`** or **0.40** with `enforce_eager=False`.  
   - **Effect:** More KV cache and graph memory, better decode speed.

### Lower impact / structural

7. **Keep `max_num_seqs=1`**  
   - Batching would require changing the SDPO loop to submit multiple prompts per call; only worth it if you move to multi-problem or multi-worker generation.

8. **FP8 / quantization for vLLM**  
   - **`quantization="fp8"`** (or int8) in LLM() can speed up and shrink the vLLM model.  
   - **Risk:** Possible quality/accuracy drop and LoRA compatibility; needs validation.

9. **Profile and log**  
   - Log vLLM's **throughput (tokens/s)** and **time to first token** per request.  
   - Distinguish "slow decode" from "long prompt encoding" or "slow Python/verification" after generation.

---

## 7. Quick reference: where it's set

| What | Where in code |
|------|----------------|
| `enforce_eager`, `max_num_seqs`, `gpu_memory_utilization`, `max_model_len` | `_setup_trainer()` → `LLM(...)` |
| `VLLM_USE_V1`, `VLLM_USE_FLASHINFER_SAMPLER` | `_setup_trainer()` → `os.environ` before `import vllm` |
| `max_new_tokens`, `temperature`, `top_p`, `stop` | `SDPOConfig` + `_generate_proof()` → `SamplingParams` |
| Single prompt per call | `_generate_proof()` → `self.vllm_engine.generate([prompt], ...)` |

---

## 8. Suggested next step

1. Set **`enforce_eager=False`** and **`max_model_len=8192`** (or 6144), and optionally **`max_new_tokens=4096`**.  
2. Run one SDPO iteration and watch for OOM.  
3. If stable, measure "est. speed output" in logs and wall-clock time for one generation; then consider FlashInfer + V1 and `gpu_memory_utilization` tweaks.
