# Generation Speed — Current Engineering Specs & Optimization Options

**Date:** 2026-02-28  
**Last updated:** 2026-03-03 (aligned with current Goedel/Kimina/DeepSeek Modal code)  
**Topics:** vllm, generation, performance

---

This document captures the current vLLM/inference configuration that affects **token generation speed**. The primary reference is **Goedel 8B** (`lean_sdpo_goedel_8b_modal.py`); Kimina and DeepSeek pipelines are summarized in §1.1.

**Timing:** Per-step timings (setup + each iteration) are now recorded; see [20260301_timing_analysis.md](20260301_timing_analysis.md) for a one-iteration run and bottleneck analysis.

---

## 1. Current vLLM Engine Configuration (Goedel 8B)

| Setting | Current value | Effect on speed |
|--------|----------------|-----------------|
| **Engine** | Legacy V0 (`VLLM_USE_V1=0`) | V1 would use torch.compile (3+ min first run) but can be faster once warm. We force V0 for predictable startup. |
| **`enforce_eager`** | **`False`** | **CUDA graphs enabled.** First request may take ~20–60s to capture; decode is ~2–4× faster (e.g. ~70+ tok/s observed). |
| **`max_seq_len_to_capture`** | **`max_model_len`** (10240) | Ensures CUDA graphs are used for full length; without this, long sequences fall back to eager and slow to ~5–27 tok/s. |
| **`gpu_memory_utilization`** | **0.4** | vLLM gets ~32 GB on an 80 GB GPU. Rest is left for Unsloth (4-bit + LoRA + optimizer). |
| **`max_model_len`** | **10240** | Max sequence length (prompt + generated). Drives KV cache size. |
| **`max_num_seqs`** | **1** | Only one sequence in flight. No batching; no opportunity to hide latency with concurrent sequences. |
| **`enable_lora`** | True | Required for SDPO; LoRA adapter overlay has some overhead. |
| **`max_lora_rank`** | 16 | Matches training; affects LoRA matmul cost. |
| **`dtype`** | bfloat16 | No quantization for vLLM (keeps quality; fp8 would speed up at cost of precision). |

**Relevant code** (in `lean_sdpo_goedel_8b_modal.py` → `_setup_trainer`):

```python
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

trainer_self.vllm_engine = LLM(
    model=trainer_self.model_name,
    dtype="bfloat16",
    trust_remote_code=True,
    download_dir="/cache",
    gpu_memory_utilization=gpu_memory_utilization,  # 0.4
    max_model_len=max_model_len,                     # 10240
    max_seq_len_to_capture=max_model_len,            # full-length CUDA graphs
    max_num_seqs=1,
    enable_lora=True,
    max_lora_rank=lora_rank,
    enforce_eager=False,   # CUDA graphs enabled for faster decode
)
```

### 1.1 Other pipelines (Kimina 2B, Kimina Distill 1.7B, DeepSeek 7B)

| Pipeline | File | GPU | gpu_mem_util | max_model_len | max_new_tokens | enforce_eager / CUDA |
|----------|------|-----|--------------|---------------|----------------|----------------------|
| **Kimina 2B** | `lean_sdpo_kimina_2b_modal.py` | A100-40GB | 0.25 | 8096 | 8096 | vLLM defaults (no LoRA) |
| **Kimina Distill 1.7B** | `lean_sdpo_kimina_distill_1_7b_modal.py` | A100-40GB | 0.25 | 8096 | 16384 | vLLM defaults (no LoRA) |
| **DeepSeek 7B** | `lean_sdpo_deepseek_7b_modal.py` | A100-80GB | 0.45 | 36000 | 32768 | `enforce_eager=False`, V0, no `max_seq_len_to_capture` |

Kimina pipelines use full-model inference (no LoRA in vLLM) and smaller `max_model_len`; DeepSeek matches Goedel’s CUDA-graph approach with a larger context (32K gen, 36K max len).

---

## 2. FlashInfer / Custom Sampler

- **`VLLM_USE_FLASHINFER_SAMPLER`** is set to **`"0"`** so vLLM does **not** use FlashInfer-based sampling.
- **Reason:** FlashInfer (and building it) can depend on `nvcc`; the Modal image is `debian_slim` without a CUDA toolkit, so we disabled it to avoid build/runtime errors.
- **Effect:** We fall back to vLLM's default (PyTorch) sampling path, which is slower than fused FlashInfer kernels.

---

## 3. Sampling / Generation Parameters

| Parameter | Value (Goedel 8B) | Effect |
|-----------|-------------------|--------|
| **`max_tokens`** | **8192** (`config.max_new_tokens`) | Each request can generate up to 8K tokens. Kimina 2B: 8096; Kimina Distill: 16384; DeepSeek: 32768. |
| **`temperature`** | 0.6 | Non-zero → sampling, not greedy; same kernel path. |
| **`top_p`** | 0.95 | Nucleus sampling. |
| **`stop`** | List of EOS tokens | Early stop can shorten generation; no impact on per-token speed. |

Generation is **single-prompt**: `generate([prompt], sampling_params, lora_request=...)` (Goedel/DeepSeek) or `generate([prompt], sampling_params)` (Kimina) — no batching.

---

## 4. Why These Choices Were Made (Goedel 8B)

- **`enforce_eager=False`:** CUDA graph capture is enabled for **2–4× faster decode** (~70+ tok/s). First request pays ~20–60s capture; subsequent decodes use the graph. Peak memory during capture is managed by `gpu_memory_utilization=0.4` and `max_seq_len_to_capture=max_model_len`.
- **`max_seq_len_to_capture=max_model_len`:** Ensures vLLM captures graphs for the full sequence length; otherwise long prompts/generations fall back to eager and drop to ~5–27 tok/s.
- **`gpu_memory_utilization=0.4`:** Leaves enough GPU memory for the 4-bit Unsloth model, LoRA, optimizer state, and activations so training doesn't OOM on A100-80GB.
- **`max_num_seqs=1`:** SDPO loop is single-problem; no need for multi-sequence scheduling. Keeps memory predictable.
- **FlashInfer off:** Image doesn't provide `nvcc`; turning it off avoids build and runtime issues.

The current Goedel 8B setup **prioritizes decode speed** (CUDA graphs) while fitting Unsloth + vLLM on one A100-80GB.

---

## 5. What You're Likely Seeing

- **With CUDA graphs enabled (Goedel/DeepSeek):** Decode can reach **~70+ tok/s** after the first request (which may take ~20–60s for graph capture). For 8192 max tokens, 8K / 70 ≈ **~2 minutes** per full generation.
- **If the metric is "~27 tokens / second" or lower:** May indicate eager fallback (e.g. sequence longer than `max_seq_len_to_capture`), or a Kimina run using vLLM defaults. Check logs for "est. speed output: X toks/s".
- **If the metric is "27 tokens / minute"** (0.45 tok/s): Pathologically slow—check for CPU-bound post-processing, very long prompt encoding, or cold start before vLLM is ready.

---

## 6. Proposed Optimizations (by impact vs risk)

### Already applied (Goedel 8B / DeepSeek 7B)

- **CUDA graphs:** `enforce_eager=False` is set; decode is ~70+ tok/s when graphs are used.
- **`max_seq_len_to_capture=max_model_len`** (Goedel only): Ensures full-length decode uses graphs; DeepSeek could add this if long generations slow down.

### High impact, try if needed

1. **Lower `max_new_tokens` for SDPO**  
   - Many proofs finish well before 8K tokens. Cap at **4096** or **2048** for experiments; increase again if you see truncation.  
   - **Effect:** Same per-token speed, but **shorter wall-clock time** per call and less KV cache pressure.

2. **Reduce `max_model_len`**  
   - If prompt + 4096 (or 2048) is enough, set **`max_model_len=6144`** or **5120**.  
   - **Effect:** Smaller KV cache, more room for CUDA graphs and/or higher `gpu_memory_utilization`.

### Medium impact, more setup

3. **Enable FlashInfer (fused sampling)**  
   - Use a **custom Modal image** that includes CUDA toolkit and builds **FlashInfer** (or use a pre-built wheel if available).  
   - Set **`VLLM_USE_FLASHINFER_SAMPLER=1`** (or vLLM's current env for FlashInfer).  
   - **Effect:** Faster sampling and potentially better memory use; exact gain depends on vLLM version and hardware.

4. **Try vLLM V1 engine**  
   - Set **`VLLM_USE_V1=1`** and accept **longer first-request latency** (torch.compile).  
   - **Effect:** After warmup, V1 can be faster; measure end-to-end for 1–2 full SDPO iterations.

5. **Slightly increase vLLM memory**  
   - If OOM is not observed, try **`gpu_memory_utilization=0.38`** or **0.40** with `enforce_eager=False`.  
   - **Effect:** More KV cache and graph memory, better decode speed.

### Lower impact / structural

6. **Keep `max_num_seqs=1`**  
   - Batching would require changing the SDPO loop to submit multiple prompts per call; only worth it if you move to multi-problem or multi-worker generation.

7. **FP8 / quantization for vLLM**  
   - **`quantization="fp8"`** (or int8) in LLM() can speed up and shrink the vLLM model.  
   - **Risk:** Possible quality/accuracy drop and LoRA compatibility; needs validation.

8. **Profile and log**  
   - Log vLLM's **throughput (tokens/s)** and **time to first token** per request.  
   - Distinguish "slow decode" from "long prompt encoding" or "slow Python/verification" after generation.

---

## 7. Quick reference: where it's set

| What | Where in code (Goedel 8B) |
|------|----------------------------|
| `enforce_eager`, `max_num_seqs`, `max_seq_len_to_capture`, `gpu_memory_utilization`, `max_model_len` | `_setup_trainer()` → `LLM(...)` |
| `VLLM_USE_V1`, `VLLM_USE_FLASHINFER_SAMPLER` | `_setup_trainer()` → `os.environ` before `import vllm` |
| `max_new_tokens`, `temperature`, `top_p`, `stop` | `SDPOConfig` + `_generate_proof()` → `SamplingParams` |
| Single prompt per call | `_generate_proof()` → `self.vllm_engine.generate([prompt], ...)` |

Kimina 2B / Distill: `_setup_trainer()` in `lean_sdpo_kimina_2b_modal.py` (and distill) sets only `gpu_memory_utilization` and `max_model_len`; no LoRA, no `enforce_eager` (vLLM defaults).

---

## 8. Suggested next steps

1. **Goedel/DeepSeek:** Current config (CUDA graphs, `max_seq_len_to_capture` on Goedel) is in place. Monitor "est. speed output" in logs; if long generations slow down on DeepSeek, add `max_seq_len_to_capture=max_model_len` there too.
2. **Kimina:** If decode is slow, consider setting `enforce_eager=False` and `max_seq_len_to_capture=max_model_len` in the Kimina `LLM(...)` call (and ensure GPU memory allows it).
3. Optional: Try **FlashInfer** (custom image with `nvcc`) or **vLLM V1** for further gains; measure end-to-end before/after.
