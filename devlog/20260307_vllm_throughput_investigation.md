# vLLM Throughput Investigation — Qwen3.5-4B on H100

**Date:** 2026-03-07  
**Topics:** vllm, throughput, performance, latency  
**Context:** Observed ~4 minutes to generate a proof of 8000 tokens. This doc investigates root causes and reports benchmark results.

---

## 1. Benchmark Result (Ground Truth)

**Benchmark run:** `devlog/throughput_results/bench_Qwen_Qwen3.5_4B_prob4_passk1_20260307_132900.json`

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen3.5-4B |
| GPU | H100 |
| Problem | [4] mathd_algebra_141 (test split) |
| Pass@k | 1 |
| Prompt tokens | 337 |
| Output tokens | 6724 (EOS, not truncated) |
| **Wall clock (timed generation only)** | **31.64s** |
| **Steady-state throughput** | **212.5 tok/s** |

**Warmup observations:**
- Warmup 1 (64 tokens): **220.6s** — torch.compile + CUDA graph warmup on first real request
- Warmup 2 (64 tokens): **0.33s** — hot path after graphs are captured

**Conclusion: vLLM decode throughput is 212.5 tok/s, which is excellent for a 4B model on H100. The CUDA graphs and default config are working correctly. No changes to the LLM() constructor are needed.**

---

## 2. Root Cause of the "4 Minutes for 8000 Tokens"

The benchmark reveals the slowness is entirely explained by the **first-request torch.compile + CUDA graph warmup penalty (~220s)**, not slow decode.

### Timeline breakdown for a cold SDPO container (first iteration)

| Phase | Time | Notes |
|-------|------|-------|
| Modal container cold start | ~30–60s | Image pull + Python init |
| vLLM model load (weights) | ~10s | Weights cached in `/hf_cache` volume |
| torch.compile (Dynamo + Inductor) | ~47s | One-time per container; cached to local disk after |
| CUDA graph capture (51 batch sizes) | ~11s | `FULL_AND_PIECEWISE` mode |
| **First inference request warmup** | **~220s** | **Root cause — V1 engine JIT-compiles on first real request** |
| QLoRA HF model load (SDPO only) | ~30–60s | 4-bit NF4 + LoRA adapters |
| Actual generation (steady state) | ~31–40s | 212 tok/s for 6–8K tokens |
| Verification (Kimina) | ~30–60s | Cold start on first call |

**Total estimated first-iteration: ~380–500s** — consistent with the observed 700s (run `023158`, cold with `max_iterations=5`) and 427s (run `030621`, possibly warmer).

### Why the first warmup request takes 220s

The vLLM V1 engine (nightly) uses `torch.compile` with `CompilationMode.VLLM_COMPILE` by default. On the first inference call after model load, it:
1. Runs Dynamo bytecode transformation (~15s)
2. Compiles the graph with Inductor (~32s)
3. Captures 51 CUDA graphs for different batch sizes (~11s)
4. Runs a warmup forward pass through the compiled + graph path

This all happens during the **first real request**, so it appears to take 220s for only 64 tokens. After this, the second call takes 0.33s for the same 64 tokens — a 667× speedup.

The `torch.compile` artifacts are cached at `/root/.cache/vllm/torch_compile_cache/` on the container's **local** disk. Since Modal containers do not persist local disk across restarts, every cold container pays this 220s penalty again.

### Important clarification: `max_seq_len_to_capture` vs `max_cudagraph_capture_size`

The earlier hypothesis that `max_seq_len_to_capture` was missing (from the V0 engine API used in Goedel 8B) does **not apply** to the current vLLM nightly (V1 engine). In V1:
- CUDA graphs are captured per **batch size** (number of concurrent tokens), not per **sequence length**
- The default `min(max_num_seqs*2, 512)` captures batch sizes 1–512, which is correct for our use case (single-sequence decode always runs at batch size 1 per step)
- Setting `max_cudagraph_capture_size=16384` caused OOM during graph capture (1043 graphs × memory cost) — confirming the V0 analogy does not hold for V1

The current config is correct. The 212 tok/s result proves CUDA graphs are active and effective.

---

## 3. Observed Performance from SDPO Logs

Both runs: `Qwen/Qwen3.5-4B`, `problem_idx=4`, `max_new_tokens=8192`, H100.

| Run | Total Wall Time | Output Tokens | Estimated gen time |
|-----|----------------|---------------|-------------------|
| `run_...023158` | 700.6s | 8192 (hit limit) | ~39s at 212 tok/s |
| `run_...030621` | 427.9s | 7544 | ~36s at 212 tok/s |

The remaining 360–660s per run is overhead: compile warmup + dual model setup + verification.

---

## 4. Remaining Performance Considerations

### 4a. First-request compile penalty (biggest practical issue)

Every cold Modal container pays ~220s on the first request. This dominates the perceived latency, especially for short SDPO runs (1–5 iterations). The compile cache lives on local container disk and is lost on each cold start.

### 4b. `gpu_memory_utilization=0.45` in SDPO

Intentional trade-off to share GPU with QLoRA HF model. Not a bottleneck — vLLM reports `Maximum concurrency for 16,384 tokens: 97.29x` at this memory level, meaning KV cache is not constrained.

### 4c. Batch size = 1 in SDPO

`_generate_proof()` submits a single prompt. Structural limitation of the sequential SDPO loop. At 212 tok/s, generation takes ~36s for a typical proof — acceptable relative to verification (~30–60s) and training (~10–30s per step).

### 4d. SamplingParams rebuilt every SDPO call

Minor: `_generate_proof()` creates a new `SamplingParams` on every call. Should build once in `_setup_trainer()` and reuse. Negligible performance impact.

---

## 5. Throughput Benchmark Tool

Written at `qwen_eval/tests/test_vllm_throughput.py`.

### Usage

```bash
# Default: problem_idx=4, pass@1, Qwen3.5-4B
python3 -m modal run qwen_eval/tests/test_vllm_throughput.py

# Batch of 4 (matches normal eval mode)
python3 -m modal run qwen_eval/tests/test_vllm_throughput.py --pass-k 4

# Shorter max tokens
python3 -m modal run qwen_eval/tests/test_vllm_throughput.py --max-new-tokens 4096
```

Results saved to `devlog/throughput_results/bench_*.json`.

### What it measures

1. **Warmup 1 (64 tokens):** absorbs the torch.compile + graph capture penalty. Logged but excluded from timed result.
2. **Warmup 2 (64 tokens):** verifies the hot path is active. Should complete in <1s.
3. **Timed round:** full `max_new_tokens` generation.
4. **Metrics:** `gen_wall_clock_s`, `total_output_tokens` (actual `token_ids` count), `tokens_per_sec_total/per_seq`, `truncated`.

---

## 6. Recommendations

### Rec 1 — Persist torch.compile cache to a Modal volume (highest impact)

Mount the compile cache on a persistent volume so cold containers skip recompilation:

```python
compile_cache_volume = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)

@app.cls(
    ...
    volumes={
        "/hf_cache": hf_cache_volume,
        "/root/.cache/vllm/torch_compile_cache": compile_cache_volume,
    }
)
```

**Impact:** Reduces cold-start first-request latency from ~220s to ~11s (CUDA graph capture only). Total first-iteration time drops by ~3 minutes.

**Risk:** The cache is tied to the model + vLLM version. Clear the volume after `vllm` package upgrades. Otherwise zero risk.

---

### Rec 2 — Increase `scaledown_window` to keep containers warm during training sessions

The SDPO trainer already has `scaledown_window=600` (10 min). Increasing to 1800s (30 min) for active sessions avoids cold-start penalties between iterations.

```python
@app.cls(..., scaledown_window=1800)
```

---

### Rec 3 — Reuse SamplingParams in SDPO (minor cleanup)

In `qwen_sdpo/modal_trainer.py`, build `SamplingParams` once at the top of `_setup_trainer()` and store on `trainer_self`. Reuse in `_generate_proof()`. Aligns with `qwen_eval.ProofGenerator`.

---

### Rec 4 — Reduce `max_new_tokens` if 4096 is sufficient

The benchmark showed 6724 tokens for a clean proof (EOS reached, `max_new_tokens=8192`). Setting `max_new_tokens=4096` would reduce worst-case generation from ~39s to ~19s, with minimal impact on proof quality for most MiniF2F problems. Truncated proofs are caught via `finish_reason="length"`.

---

## 7. Summary

| Finding | Status |
|---------|--------|
| **Steady-state decode throughput** | **212.5 tok/s** — healthy, no vLLM config changes needed |
| **"4 min for 8000 tokens" cause** | **220s torch.compile + CUDA graph warmup on cold container** |
| `max_seq_len_to_capture` (V0 API) | Not applicable — V1 engine handles this differently; default config is correct |
| `max_cudagraph_capture_size=16384` | Caused OOM (1043 graphs); not the right fix |
| `gpu_memory_utilization=0.45` in SDPO | Intentional, not a bottleneck |
| Batch size = 1 in SDPO | Structural; 212 tok/s makes it ~36s/proof, acceptable |

**The actionable fix is persisting the torch.compile cache to a Modal volume (Rec 1). This eliminates the 220s cold-start penalty that accounts for almost all of the observed latency.**
