# Timing Analysis — One-Iteration Run (2026-03-01)

Instrumentation was added to record every step in the SDPO pipeline (setup + per-iteration). This document summarizes one run with **max_iterations=1**, **problem_idx=0** (mathd_algebra_478), **CUDA graphs enabled** (`enforce_eager=False`), and **gpu_memory_utilization=0.4**.

---

## 1. Setup timings (one-time)

| Phase | Time (s) | % of setup |
|-------|----------|------------|
| Unsloth load (4-bit + LoRA) | 20.27 | 23% |
| Initial LoRA save to disk | 5.68 | 7% |
| vLLM init (bf16 + CUDA graph capture) | 38.47 | 44% |
| **Total setup** | **86.74** | 100% |

- vLLM init includes model load (~7s), memory profile (~6s), and **CUDA graph capture (~6s)**. Graph capture completed successfully.
- Setup is one-time per Modal worker; subsequent runs on the same container skip it until scaledown.

---

## 2. Iteration 1 timings (success path: proof found)

| Step | Time (s) | % of iteration |
|------|----------|----------------|
| **vLLM generate** | 13.43 | **26%** |
| Tokenize (post-process) | 0.0 | 0% |
| Extract tactics + build full Lean code | 0.0 | 0% |
| **Verify (Kimina Lean Server)** | **38.20** | **74%** |
| **Iteration total** | **51.63** | 100% |

- **Throughput:** 73.5 tok/s (vLLM reported ~73.6 output tok/s). CUDA graphs are active and much faster than previous eager ~27 tok/s.
- Verification includes **Kimina Lean Server cold start** (~18s from logs: "Kimina Lean Server ready after 18.0s") plus the actual verification call. So ~20s is verification work, ~18s is server startup on first use.

---

## 3. Bottleneck

**Primary bottleneck in this run: verification (74% of iteration time).**

- **Verify:** 38.2s — largest share. For the first iteration this includes starting the Kimina Lean Server (cold). Subsequent iterations would only pay the verification RTT (~20s or less depending on proof complexity).
- **vLLM generate:** 13.43s — second. With CUDA graphs and 73.5 tok/s, generation is no longer the main limiter for short/medium outputs.
- Extract/build and tokenize are negligible (<0.01s).

**For multi-iteration runs (failed attempts):**  
Each failed attempt adds: generate + extract + verify + (SDPO loss + optional optimizer step). Verify will still dominate per iteration unless proofs are trivial and Kimina is warm. **Recommendation:** keep Kimina Lean Server warm (e.g. longer `scaledown_window`) or run verification in a pre-warmed pool to avoid cold start on first request.

**Setup:** For cold runs, vLLM init (38.5s) is the largest one-time cost; Unsloth load (20s) is second. Both are acceptable for 5–10+ iteration runs.

---

## 4. Where timings are recorded

- **Logs:** `logs["setup_timings"]` and `logs["iteration_logs"][i]["timing"]` in `logs.json` (Modal volume and local copy under `sdpo_results/goedel_8b/run_*`).
- **Fields:** `unsloth_load_s`, `initial_lora_save_s`, `vllm_init_s`, `total_setup_s`; per iteration: `vllm_generate_s`, `tokenize_s`, `generate_total_s`, `tokens_per_sec`, `extract_and_build_s`, `verify_s`, `sdpo_loss_backward_s`, `optimizer_step_s`, `lora_save_s`, `iteration_total_s`.
- **Console:** Setup phases and iteration timing summary are printed during the run.

---

## 5. Config used for this run

- `enforce_eager=False` (CUDA graphs enabled)
- `gpu_memory_utilization=0.4`
- `max_model_len=10240`, `max_new_tokens=8192`
- One iteration; problem 0 (mathd_algebra_478); success on first try.
