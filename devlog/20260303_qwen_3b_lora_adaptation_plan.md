# Qwen 3B SDPO LoRA adaptation — plan and requirements

**Date:** 2026-03-03  
**Topics:** sdpo, qwen, lora, unsloth, modal, oom, memory, config

---

## TL;DR

- **Goal:** Add a LoRA-based training path for `lean_sdpo_qwen_3b_modal.py` to avoid CUDA OOM during `optimizer.step()` on A100-40GB (full-model + vLLM + Adam states exceed 40GB).
- **Approach:** New script `lean_sdpo_qwen_3b_lora_modal.py` adapted from the current Qwen 3B script, using the **Goedel 8B** pattern: Unsloth (4-bit + LoRA) for gradients, vLLM (bf16 + LoRA overlay) for inference; LoRA weights saved after each optimizer step and loaded by vLLM via `LoRARequest`.
- **LoRA save location (important):** Save LoRA to a **single fixed path** every time (overwrite), not versioned paths (v0, v1, v2). This avoids vLLM generating new CUDA graphs when the LoRA path changes; the same path is used for every `LoRARequest`, and we overwrite the files on disk after each optimizer step.
- **Status:** Part 2–7 completed (2026-03-03). E2E run: no OOM; proof found on iteration 1 (minif2f-lean4 problem 0).

---

## 1. Package and environment requirements

### 1.1 Goedel 8B (reference) — Unsloth + vLLM

- **Image:** `debian_slim`, Python **3.11** (Unsloth stack is pinned to 3.11 in Modal examples).
- **Install (uv):**
  - `unsloth[cu128-torch270]==2025.7.8` — 4-bit quantized training, LoRA, must be imported **before** `transformers`.
  - `unsloth_zoo==2025.7.10`
  - `trl==0.19.1`
  - `vllm>=0.6.0`
  - `transformers==4.53.2`
  - `accelerate==1.9.0`
  - `peft==0.16.0`
  - Plus: `sentencepiece`, `protobuf`, `datasets`, `matplotlib`, `httpx`.
- **Env:** `VLLM_USE_V1=0`, `VLLM_USE_FLASHINFER_SAMPLER=0` so vLLM uses legacy engine and respects `enforce_eager` if needed.
- **GPU:** A100-80GB; `gpu_memory_utilization=0.4` so vLLM and Unsloth fit.

### 1.2 Qwen 3B current (no LoRA)

- **Image:** `debian_slim`, Python 3.12; no Unsloth. `torch`, `transformers>=4.40.0`, `vllm>=0.6.0`, `accelerate`, `datasets`, `matplotlib`, `httpx`.
- **Model load:** Full bfloat16 via `AutoModelForCausalLM.from_pretrained(..., torch_dtype=dtype, device_map="auto")`.
- **vLLM:** Same model name, bf16, no LoRA. Single process holds full model + vLLM engine → OOM at optimizer step when Adam needs extra memory.

### 1.3 What we need for Qwen 3B LoRA

- **Unsloth:** We use **Qwen/Qwen2.5-Coder-3B-Instruct** only (no fallback to non-Coder). The Coder model is required for our Lean theorem-proving task; Unsloth supports it (see e.g. unsloth/Qwen2.5-Coder-3B-Instruct, Qwen 2.5 Coder fine-tuning docs). Use same Unsloth/peft/transformers pinning as Goedel for compatibility.
- **vLLM:** Must load base model with `enable_lora=True`, `max_lora_rank=...`; generation with `LoRARequest(lora_name, lora_int_id, lora_path)`. Use a **fixed** `lora_path` (same directory every time) so vLLM does not regenerate CUDA graphs.
- **Python:** Prefer 3.11 for Unsloth compatibility unless we confirm 3.12 works with the same Unsloth build.

---

## 2. Essential components (what changes from Qwen 3B → Qwen 3B LoRA)

| Component | Qwen 3B (current) | Goedel 8B (LoRA reference) | Qwen 3B LoRA (target) |
|-----------|-------------------|----------------------------|------------------------|
| **Modal image** | No Unsloth, py3.12 | Unsloth + uv, py3.11 | Unsloth + uv, py3.11 (match Goedel) |
| **Config** | No LoRA params | `lora_rank`, `lora_alpha`, `gradient_accumulation_steps` | Add same LoRA/grad-accum params |
| **Model load** | `AutoModelForCausalLM` full bf16 | `FastLanguageModel.from_pretrained(load_in_4bit=True)` then `get_peft_model(...)` | Same as Goedel (Unsloth 4-bit + LoRA) |
| **Tokenizer** | From `AutoTokenizer` | From `FastLanguageModel.from_pretrained` (returns model, tokenizer) | From Unsloth load |
| **Initial LoRA** | N/A | Save to `/tmp/sdpo_lora_weights/v0` for vLLM | Save to **fixed** path e.g. `/tmp/sdpo_lora_weights/current` (no versioning) |
| **vLLM** | `LLM(..., no enable_lora)` | `LLM(..., enable_lora=True, max_lora_rank=...)` | Same as Goedel |
| **Generation** | `vllm_engine.generate([prompt], sampling_params)` | `generate(..., lora_request=LoRARequest(...))` when `lora_version > 0` | **Always** pass same `LoRARequest` (same path); no version bump |
| **After optimizer.step()** | N/A | `_save_lora_and_bump_version()` → save to `.../v{N}`, increment `lora_version` | **Overwrite** same path (e.g. `.../current`); no new dirs, no CUDA graph rebuild |
| **Optimizer** | AdamW over `self.model.parameters()` | AdamW over `self.model.parameters()` (only LoRA params trainable) | Same |
| **Gradient accumulation** | None (1 step per failed iteration) | `gradient_accumulation_steps` (e.g. 4); loss scaled by 1/accum_steps; step at boundary | Optional: keep 1 for Qwen 3B to simplify, or add for stability |
| **Saving results** | `final_model/` (full model) | `final_lora/` (adapter + tokenizer) | `final_lora/` |
| **Debug / verification** | Same LeanVerifier, Kimina | Same | No change |

### 2.1 LoRA target modules

- Goedel uses: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (attention + MLP).
- Qwen2.5-Coder uses the same standard names; use the same list.

### 2.2 LoRA path: fixed location, overwrite (no versioned paths)

- **Goedel 8B (reference):** Saves to versioned dirs `.../v0`, `.../v1`, `.../v2` and passes a new `LoRARequest(..., lora_path=.../v{N}, lora_int_id=N)` each time. vLLM then compiles/loads a new LoRA for each path, which can trigger new CUDA graph generation.
- **Qwen 3B LoRA (target):** Save to a **single fixed directory** (e.g. `/tmp/sdpo_lora_weights/current`) every time. After each optimizer step, overwrite the files in that directory. Always pass the same `LoRARequest(lora_name, lora_int_id=1, lora_path=.../current)` so vLLM reuses the same LoRA slot and does not regenerate CUDA graphs. Implementation: (1) On setup, save initial LoRA to the fixed path. (2) After each optimizer step, call `model.save_pretrained(same_fixed_path)` (and tokenizer); do **not** increment a version or create new directories.

### 2.3 Memory (Qwen 3B on A100-40GB)

- With LoRA: vLLM (bf16 base, e.g. ~0.25 util) + Unsloth (4-bit + LoRA) + optimizer (only LoRA) + activations should fit in 40GB.
- If needed: reduce vLLM `gpu_memory_utilization` (e.g. 0.25) and/or `max_model_len` to leave room for Unsloth + optimizer.

---

## 3. Step-by-step plan (do one part at a time; wait for your “done” before next)

### Part 1 — Copy and plan (DONE)

- [x] Create `training/lean_sdpo_qwen_3b_lora_modal.py` as a copy of `lean_sdpo_qwen_3b_modal.py`.
- [x] Write this devlog (requirements, components, plan).
- [x] Update devlog README with this entry.

### Part 2 — Image and config (DONE)

- [x] In `lean_sdpo_qwen_3b_lora_modal.py`:
  - Replace the Modal **image** with the Unsloth-based image (Python 3.11, uv, same deps as Goedel: unsloth, unsloth_zoo, trl, vllm, transformers, accelerate, peft, etc.).
  - Add **config** fields: `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_bias`; `gradient_accumulation_steps` (default 1).
  - Set **app name** to `lean-sdpo-qwen-lora` and **output_dir** to `qwen_3b_lora`.
- [x] Image build and container start confirmed (build-check run; helper removed afterward).

### Part 3 — Trainer setup: Unsloth model + vLLM with LoRA (DONE)

- [x] Replace `_setup_trainer` with Unsloth + vLLM LoRA setup:
  - Set env: `HF_HOME`, `VLLM_USE_V1=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`, token if present.
  - Import Unsloth before transformers; load with `FastLanguageModel.from_pretrained(..., load_in_4bit=True)`.
  - Attach LoRA with `FastLanguageModel.get_peft_model(..., target_modules=LORA_TARGET_MODULES, use_gradient_checkpointing=True)`.
  - Save initial LoRA to **fixed path** `LORA_WEIGHT_PATH` (overwrite same dir every time; no versioning).
  - Initialize vLLM with `enable_lora=True`, `max_lora_rank=lora_rank`, `max_seq_len_to_capture`, `max_num_seqs=1`.
- [x] Add `LORA_WEIGHT_DIR`, `LORA_WEIGHT_PATH` (fixed), `LORA_TARGET_MODULES`.
- [x] Add Modal parameters `lora_rank`, `lora_alpha` on SDPOTrainer. No `lora_version` (fixed path).

### Part 4 — Generation and LoRA save (fixed path, overwrite) (DONE)

- [x] In `_generate_proof`: pass `lora_request=LoRARequest(lora_name="sdpo_adapter", lora_int_id=1, lora_path=LORA_WEIGHT_PATH)` every time.
- [x] Add `_save_lora_weights()`: overwrite `LORA_WEIGHT_PATH` with `model.save_pretrained` and `tokenizer.save_pretrained`.
- [x] Add `_strip_special_tokens_from_generation` for Qwen (`<|im_end|>`, `<|endoftext|>`, `<|im_start|>assistant`).

### Part 5 — Training loop (DONE)

- [x] After each `optimizer.step()`, call `_save_lora_weights()` (overwrite fixed path).
- [ ] Gradient accumulation: not added; default `gradient_accumulation_steps=1`. Can add later if needed.
- [ ] Ensure `_compute_sdpo_loss` and optimizer use only the Unsloth model (no change to loss math; Unsloth model already has LoRA and is the one we step).

### Part 6 — Saving and entrypoint (DONE)

- [x] In `_save_results`: save to `final_lora/` (adapter + tokenizer) instead of full model; same logs/metrics/plots.
- [x] In `main()`: add CLI args `--lora-rank`, `--lora-alpha` (and optionally `--gradient-accumulation-steps`); pass into config_dict; print LoRA settings.
- [x] Fix deprecation: replace `allow_concurrent_inputs` with `@modal.concurrent(max_inputs=100)` on KiminaLeanServer.

### Part 7 — Cleanup and devlog (DONE)

- [x] Add top-of-file docstring for the LoRA script (purpose, two-instance design, usage).
- [x] Run one end-to-end test (e.g. `modal run training/lean_sdpo_qwen_3b_lora_modal.py --problem-idx 0 --max-iterations 2`) and confirm no OOM.
- [x] Append to this devlog: “Part 2–7 completed”, any gotchas (e.g. Unsloth/Qwen Coder compatibility, memory settings).
- [x] Devlog README already has categories and tags (lora, unsloth, oom); no change needed.

---

## 4. Files touched

| File | Role |
|------|------|
| `training/lean_sdpo_qwen_3b_lora_modal.py` | New LoRA variant (copy of Qwen 3B, then adapted). |
| `training/lean_sdpo_goedel_8b_modal.py` | Reference only (Unsloth + vLLM LoRA pattern). |
| `training/lean_sdpo_qwen_3b_modal.py` | Unchanged; remains full-model (for reference or smaller runs). |
| `devlog/20260303_qwen_3b_lora_adaptation_plan.md` | This file. |
| `devlog/README.md` | Index updated with this devlog. |

---

## 5. Risks and notes (expanded)

This section documents implementation decisions, every exception/fallback in the pipeline, and what can go wrong. Use it for debugging and to avoid introducing silent fallbacks or wrong defaults.

### 5.1 Model and Coder requirement

- **Model:** We use **Qwen/Qwen2.5-Coder-3B-Instruct** only. There is **no fallback** to a non-Coder model (e.g. `Qwen/Qwen2.5-3B-Instruct`). The Coder variant is required for the Lean/code task; substituting a chat-only Qwen would change behavior. Unsloth supports Qwen2.5-Coder-3B-Instruct (see §1.3 and Unsloth blog/collection).
- **CLI `--model`:** The user can pass any HuggingFace ID via `main(model=...)`. The script does **not** validate that the ID is a Coder model. If someone passes a non-Coder model, training will run but results may be poor for Lean. Consider adding a warning or check when `model_name` does not contain `"Coder"` (optional).

### 5.2 Exceptions and fallbacks in the pipeline

- **`apply_chat_template` (base + feedback prompt):**  
  In `_create_base_prompt` and `_create_feedback_prompt`, if `tokenizer.apply_chat_template(...)` raises, we catch with a **bare `except:`** and fall back to a plain string prompt: `"System: ...\n\nUser: ...\n\nAssistant:"`.  
  **Risk:** Any tokenizer error (e.g. wrong message format, missing template) is swallowed and the model sees a non-Qwen format. That can hurt generation quality or cause subtle bugs. Prefer `except Exception as e:` and at least log; consider failing fast if the template is required for Qwen.

- **Kimina server kill (`_start_lean_server`):**  
  Before starting the server we `server_proc.kill()` and `server_proc.wait(timeout=5)` in a `try/except Exception: pass`. So if the previous process is already dead or does not exit in time, we ignore and continue. Safe for restarts; no raise.

- **Kimina health check (`_start_lean_server`):**  
  The loop that waits for the Lean server uses `try/except Exception: pass` and `time.sleep(2)`. If the server never responds within 60s, we only **print** a warning and return; we do not raise. So the container can continue with a server that is not actually ready.  
  **Risk:** First `verify` may then fail or hang. The subsequent verify retries (in the SDPO loop) may mask the issue.

- **Kimina `_ensure_server_alive`:**  
  Same pattern: if the health POST fails, we restart the server and return. No raise.

- **Kimina `verify` (inside KiminaLeanServer):**  
  Connect/timeout/other errors are caught; we retry up to 3 times with server restart, then return `{"error": "...", "is_server_error": True}`. So we never raise; the caller (LeanVerifier) always gets a dict.

- **LeanVerifier.verify:**  
  The whole body is in `try/except Exception as e`. On any exception we return a synthetic failure dict with `success=False`, `feedback=str(e)`, `is_server_error=True`. So **no exception propagates** from verification to the trainer.  
  **Risk:** Real bugs (e.g. bad response shape, network) look like “server error”; we skip training for that attempt and retry. Good for resilience, but can hide bugs if logs are not inspected.

- **Debug JSONL write (verify_start / verify_end):**  
  Writing to `debug_lean_path` is wrapped in `try/except Exception: pass`. So if the volume is full or path is bad, we **silently skip** writing. No raise, no warning.  
  **Risk:** Debug artifacts may be missing with no indication.

- **Dataset loading (`main`):**  
  We try, in order: (1) `load_dataset(dataset, subset?, split)`, (2) with `trust_remote_code=True`, (3) load full dataset and pick split or first available split. The third path can **silently switch** the split (e.g. user asked for `test`, we use `train` if `test` missing). We print “Using split: …” but do not require user confirmation.  
  **Risk:** Wrong split used; results may not be comparable to expectations.

- **`problem_idx` out of range:**  
  If `problem_idx >= len(ds)`, we print a message and set `problem_idx = 0`. We do **not** raise. So we silently run on problem 0 instead of failing.  
  **Risk:** User thinks they ran on problem N but actually ran on 0.

- **`_get_field` / dataset field extraction:**  
  Returns `default=""` when no field matches. So missing theorem code yields empty string; later code may build invalid Lean (e.g. empty `lean4_code`). The run continues; verification or extraction may then fail in a less obvious way.

### 5.3 Design decisions and their implications

- **Fixed LoRA path (`LORA_WEIGHT_PATH`):**  
  We always save to and load from the same path. vLLM receives the same `lora_path` every time, so it does not create new LoRA slots/CUDA graphs. If vLLM **caches** LoRA in memory and does not re-read from disk after we overwrite the files, the next `generate()` could use **stale** weights. This is implementation-dependent; we have not verified vLLM’s reload behavior. If you see no learning across iterations, consider forcing a new path or checking vLLM docs for LoRA hot-reload.

- **No `lora_version`:**  
  We do not bump a version or create new dirs. Simpler and avoids CUDA graph rebuilds; see above for possible staleness.

- **`LORA_TARGET_MODULES`:**  
  Hardcoded to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Correct for Qwen2; if the repo ever uses a different architecture, this list must be updated or LoRA will not apply to the right layers.

- **Env vars in setup:**  
  `VLLM_USE_V1=0`, `VLLM_USE_FLASHINFER_SAMPLER=0` are set so vLLM uses the legacy engine and we can use `enforce_eager=False` for CUDA graphs. If Unsloth or vLLM change behavior by default, these may need to be revisited.

- **`pad_token` fallback:**  
  If `tokenizer.pad_token is None`, we set `tokenizer.pad_token = tokenizer.eos_token`. Standard and correct for Qwen; no fallback to a different model.

- **Header for Lean code:**  
  If the dataset does not provide a header and the model does not generate imports, we use `config.default_header`. That is a **content** fallback (default Lean imports), not a code-path exception.

- **`_save_results` output dir:**  
  Saves to `run_dir / "final_lora"` (Part 6 done). We save with `self.model.save_pretrained(...)`; for a PEFT model that is the adapter + config (LoRA weights + tokenizer), which is correct.

### 5.4 Memory, Python, and dependencies

- **A100-40GB:** With LoRA, vLLM (bf16) + Unsloth (4-bit + LoRA) + optimizer (LoRA only) + activations should fit. If OOM persists, reduce `max_model_len` or vLLM `gpu_memory_utilization`, or ensure gradient checkpointing is enabled (already set in Unsloth `get_peft_model`).

- **Python 3.11:** The Modal image uses 3.11 for Unsloth compatibility. Using 3.12 may require a different Unsloth wheel or may fail at install/import; we have not tested 3.12.

- **Unsloth / vLLM / transformers versions:** Pinned in the image (Unsloth 2025.7.8, transformers 4.53.2, vLLM ≥0.6.0). Upgrading any of these can change LoRA format, request shapes, or reload behavior; test after upgrades.

### 5.5 Summary of recommendations

- Avoid adding new silent fallbacks (e.g. model ID, split, or `problem_idx`) without at least logging or warning.
- Consider logging when `apply_chat_template` falls back to the plain prompt.
- Consider validating or warning when `model_name` does not look like a Coder model.
- When debugging “no learning,” verify that vLLM is loading updated LoRA from disk (or that the fixed path does not cause stale cached weights).
- Part 6 done: `_save_results` now writes to `final_lora/` (adapter + tokenizer).

---
## 6. Part 2–7 completed (2026-03-03)

- **Parts 2–5:** Image (Unsloth stack, Python 3.11), config (lora_rank/alpha/dropout, gradient_accumulation_steps), Unsloth + vLLM LoRA setup, fixed LoRA path, generation + save after each step, training loop.
- **Part 6:** `_save_results` → `final_lora/`; CLI `--lora-rank`, `--lora-alpha`, `--gradient-accumulation-steps`; `@modal.concurrent(max_inputs=100)` on KiminaLeanServer.
- **Part 7:** Top-of-file docstring (two-instance design, usage with `training/lean_sdpo_qwen_3b_lora_modal.py`); E2E test run.

**E2E test (2026-03-03):** `modal run training/lean_sdpo_qwen_3b_lora_modal.py --dataset cat-searcher/minif2f-lean4 --dataset-split test --problem-idx 0 --max-iterations 2`. Result: no OOM; Unsloth + vLLM LoRA loaded on A100-40GB; proof verified on iteration 1; final LoRA saved to `qwen_3b_lora/run_*/final_lora`.

**Gotchas / notes:**
- Unsloth/Qwen2.5-Coder-3B-Instruct: works with Unsloth 2025.7.8, transformers 4.53.2, vLLM 0.9.2.
- Modal: run with `python -m modal run` or ensure `modal` is on PATH (e.g. from venv).
- `allow_concurrent_inputs` is deprecated; use `@modal.concurrent(max_inputs=100)` on the class.
- Local results go to `sdpo_results/qwen_3b_lora/<dataset>/run_<problem_idx>_<timestamp>/`.

---

## 7. Proof tactics parsing investigation (2026-03-03)

**Context:** Run log `sdpo_results/qwen_3b_lora/minif2f-lean4/run_0_20260303_003126/logs.json` — iteration 1 shows `raw_output` with a long English preamble and `extracted_tactics` that looked like they might not come from `raw_output`. Investigation: where are tactics extracted from, and does the raw output actually contain the proof?

### 7.1 Where are the tactics extracted from?

- **Source:** `_extract_proof_tactics(raw_output)` in `lean_sdpo_qwen_3b_lora_modal.py` (SDPOTrainer).
- **Flow:**  
  1. If output has `<think>` but no `</think>`, return `"sorry"` (no extraction).  
  2. **Strategy 1:** If `</think>` present, look for ```lean4 / ```lean / ```tactics code blocks in the text *after* `</think>`; take the first block whose `_extract_tactics_from_code_block(...)` result is non-empty and not "sorry"/"by".  
  (No fallbacks: we do not search the entire output or use a ":=" by " heuristic.)

- For the run in question there is **no** `</think>` in `raw_output`. With **strict** parsing (current behavior), we do **not** search the rest of the output: we return `"sorry"`. So run_0 would now fail verification (no tactics extracted). Previously we had fallbacks (Strategy 2 / 4) that took the first code block in the whole output; those are removed.

- **Conclusion:** Tactics are extracted from the **first** ```lean4 (or ```lean / ```tactics) code block in `raw_output`. In this log, that block is the one containing `import tactic`, `variable (b h v : ℝ)`, the two `have` lines, and `exact v_eq_65`.

### 7.2 Does the raw output contain the proof tactics?

- **Yes.** In the log, `raw_output` is one long string. It contains:
  - English explanation (e.g. "To solve this problem in Lean 4...", "Here's the Lean 4 code to achieve this:").
  - Then a **markdown code block:** `\n\n```lean4\nimport tactic\n\n-- Define the variables\nvariable (b h v : ℝ)\n...\nhave h₁_sub : v = 1 / 3 * (30 * (13 / 2)) := by rw [h₂, h₃]\n\n-- Simplify...\nhave v_eq_65 : v = 65 := by rw [h₁_sub, mul_comm 30 13 / 2, div_by_2 65]\n\n-- Show...\nexact v_eq_65\n```\n\nThis code defines...`
- So the Lean proof *is* inside `raw_output`, inside that single ```lean4 block. The parser correctly identifies that block and runs `_extract_tactics_from_code_block` on it. Any confusion likely comes from (a) most of the text being English, or (b) the code block being easy to miss in a single-line or truncated JSON view.

### 7.3 Parsing bug: one tactic line was incorrectly dropped

- **Observed:** `extracted_tactics` in the log contains `have h₁_sub : v = 1 / 3 * (30 * (13 / 2)) := by rw [h₂, h₃]` and `exact v_eq_65`, but **not** the middle line `have v_eq_65 : v = 65 := by rw [h₁_sub, mul_comm 30 13 / 2, div_by_2 65]`.
- **Cause:** In `_extract_tactics_from_code_block`, the filter intended to skip “type signature” lines like `4 * x^3 - 7 * y^3 ≠ 2003 :=` uses the regex `r"[≠=<>]\s*\d+\s*:="`. The line `have v_eq_65 : v = 65 := by rw [...]` contains the substring `= 65 :=`, so it **matches** this regex and was incorrectly skipped.
- **Fix (in script):** Only apply that skip when the line does **not** start with tactic-style prefixes `have `, `suffices `, or `show ` (such lines have a tactic body after `:= by` and should be kept). Applied in `training/lean_sdpo_qwen_3b_lora_modal.py`.
- **Effect:** Future runs will retain lines like `have v_eq_65 : v = 65 := by rw [...]` in `extracted_tactics`. The run in question still verified successfully because the assembled `full_code` (theorem statement + tactics) already contained enough for the prover (e.g. `exact v_eq_65`), and the dropped line was an intermediate `have` that the verifier could infer or that did not change the final result for this particular problem.

### 7.4 Run run_0_20260303_003126: “No code block after reasoning trace”

For the run in `sdpo_results/qwen_3b_lora/minif2f-lean4/run_0_20260303_003126/logs.json`, two possibilities were raised: (1) the raw_output in the log is incomplete, or (2) extraction is wrong and we did not extract from the solution properly.

**Checked with the actual log file:**

- **Is the logged raw_output complete?**  
  Yes. The same `raw_output` string that is passed to `_extract_proof_tactics` is written to the iteration log with no truncation (`json.dump(..., default=str)`). The logged string has length 1316 and is the full model reply.

- **Does this run have a “reasoning trace”?**  
  No. The logged `raw_output` contains **no** opening think tag and **no** closing think tag. So there is no “reasoning trace” (no think block) in this output. The model produced: [English prose] then [one fenced lean4 code block] then [more English]. So we never use “extract after reasoning” for this run — Strategy 1 does not apply.

- **Is there a lean4 code block in the raw_output?**  
  Yes. There is exactly one fenced code block matching our pattern (lean4/lean/tactics). It starts with `import tactic` and ends with `exact v_eq_65`. So the “solution” (the Lean proof) **is** in the log, inside that one block. It appears in the middle of the string after “Here’s the Lean 4 code to achieve this:” and before “This code defines…”.

- **Where do we extract from now (strict)?**  
  With **strict** parsing we only extract from after the closing think tag. This run has no think tags, so we would return `"sorry"` and verification would fail. The previous fallback (Strategy 2: first code block in entire output) is removed.

### 7.5 Where raw_output is produced (code path) and where the lean block is in the log

**Where raw_output comes from (everything after the prompt):**

- **File:** `training/lean_sdpo_qwen_3b_lora_modal.py`
- **Generation:**  
  - **Lines 1038–1040:** `outputs = self.vllm_engine.generate([prompt], ...)` then `generated_text = outputs[0].outputs[0].text`. vLLM returns **only the model’s completion** (the assistant reply), not the prompt. Then `generated_text = self._strip_special_tokens_from_generation(generated_text)` strips Qwen special tokens.  
  - **Line 1046:** `return generated_text, generated_ids`  
  - **Line 662:** `raw_output, generated_ids = self._generate_proof(config, base_prompt)`  
- So **raw_output** is exactly the model’s generated text: everything after the prompt (reasoning trace if present, then any text/code after it). The prompt is not included in raw_output.

**Where the lean block is in the stored log (run_0):**

- In `sdpo_results/qwen_3b_lora/minif2f-lean4/run_0_20260303_003126/logs.json`, the string `raw_output` has length **1316** characters.
- The **fenced lean4 block** (the substring between the opening backticks and closing backticks) runs from **character index 547** to **1081** (inclusive of the backtick lines). So:
  - Characters 0–546: English preamble (“To solve this problem…”, “Here’s the Lean 4 code to achieve this:”).
  - Characters 547–1081: the block (starts with `` ```lean4 `` then newline then `import tactic` … `exact v_eq_65` then newline then `` ``` ``).
  - Characters 1082–1315: trailing English (“This code defines…”).
- **extracted_tactics** is a filtered version of the **content inside** that block (the 522 characters between the backticks): `_extract_tactics_from_code_block` drops blank lines, `--` comments, and a few other patterns, then joins the rest. So the “connected clean block” you see in extracted_tactics is the same block as in raw_output, just with comments and blank lines removed.

**Summary:** The raw_output in the log is complete. There is no reasoning trace in this run, so there is no “code block after the reasoning trace” — there is only a code block after the English preamble. Under strict parsing, such outputs get no tactics (sorry) and verification fails. If the UI or JSON viewer renders the fenced block as a code block and hides the raw text, the block can look “missing” even though it is present in the string.

---

## 8. How the parser parses the Lean solution

**Goal:** The Lean proof should appear *after* the model's reasoning. The parser should find that proof and extract it. This section describes how that works in plain language.

**Note:** There is **no special think token** in generation. We do not add or request `<think>`/`</think>` in the prompt; Qwen2.5-Coder-3B-Instruct does not emit them by default. The parser is written to *support* think tags if a model outputs them; with the current model and strict parsing, outputs have no think tags, so we always return sorry unless we change the prompt or use a model that emits think tags.

---

### Step 1: What the parser sees

- The model (vLLM) returns one long string: the assistant's reply.
- We strip special tokens (e.g. Qwen's end-of-turn and assistant-start tokens).
- The result is stored as **raw_output**. The rest of the parser only ever sees this single string.

---

### Step 2: Truncation rule (incomplete reasoning)

- Some models output a *reasoning* section inside **think tags**: an opening tag, the reasoning text, then a closing tag.
- **Rule:** If the output contains the *opening* think tag but **not** the *closing* think tag, we do **not** try to parse anything. We immediately set tactics to `"sorry"`.
- So we never use text from inside an unclosed reasoning block. If the model got cut off mid-reasoning, we don't treat whatever is there as the proof.

---

### Step 3: How we find the proof (strict: no fallbacks)

The function `_extract_proof_tactics(raw_output)` uses **one** rule. There are no fallbacks.

**Only: proof after the reasoning trace**

- **When:** The output contains the *closing* think tag (so the reasoning block is complete).
- **What we do:**
  - Take everything that comes **after** the last closing think tag. Call that slice **after_think**.
  - We look **only** in after_think for a Lean code block. We do **not** look anywhere else in the output.
  - A "Lean code block" means a fenced block labeled lean4, lean, or tactics (the usual markdown-style triple backticks).
  - We take the **first** such block in after_think whose content, after a cleanup step, counts as valid tactics.
- **If there is no closing think tag:** We return `"sorry"` and do **not** search the rest of the output. So if the model does not use think tags at all (e.g. run_0 in §7.4), verification will get sorry and fail.
- **If there is a closing think tag but no valid code block after it:** We return `"sorry"`.

---

### Step 4: How we clean the chosen block (not verbatim)

- Once we have picked one code block (from the text after the closing think tag), we don't use it raw.
- We take the **text inside** the block (between the backticks) and run **`_extract_tactics_from_code_block`** on it:
  - Split into lines.
  - **Keep:** import/open/set_option lines; lines that look like tactic steps (e.g. have ... := by ..., exact ..., rw [...]). We do *not* drop lines that start with have, suffices, or show even if they match a "type signature" pattern.
  - **Drop:** blank lines, comment-only lines, theorem/lemma declarations, parameter-only lines, := sorry, and a few other patterns meant to skip non-tactic lines.
- The lines we keep are joined into one string. That string is **extracted_tactics**.

---

### Step 5: What we do with the result

- **extracted_tactics** is then passed to **`_create_full_lean_code`**, which combines it with the dataset's theorem statement and header to build the full Lean file that gets sent to the verifier.

---

### How this matches "proof after reasoning" and "extract directly"

| Your ideal | What the code does |
|------------|--------------------|
| **Proof after reasoning** | When the closing think tag is present, we look **only** at the text after it for the first lean4 block. So the intended usage (reasoning first, then proof in a lean4 block) is already supported. |
| **Extract the proof directly** | We take the proof only from the first lean4 block after the closing think tag. There are no fallbacks (no search over the whole output). "Directly" here means: take that block's content, then run the line-by-line filter above. We do **not** use the block verbatim with zero filtering. |

If by "directly" you mean **no filtering at all** (use the content between the lean4 backticks as-is), that would require a small change: e.g. when the closing think tag is present, optionally take the first lean4 block after it and use it verbatim instead of passing it through `_extract_tactics_from_code_block`. Right now we *do* take the proof from the right place but then apply the same cleanup everywhere.

---

## 9. Migration: Unsloth → Hugging Face + PEFT (2026-03-03)

### 9.1 Rationale

Unsloth’s training path (both `use_gradient_checkpointing=True` and `"unsloth"`, and even with `use_gradient_checkpointing=False`) still triggers:

`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [1, 644, 2048]], which is output 0 of MulBackward0, is at version 1; expected version 0 instead.`

The in-place modification comes from **Unsloth’s patched model forward** (e.g. custom attention/RoPE kernels), not only from their gradient checkpointing. Disabling checkpointing did not fix it. To avoid depending on Unsloth’s internals, we replace the **training** stack with standard Hugging Face + bitsandbytes (4-bit) + PEFT LoRA. vLLM for inference is unchanged; it already loads Hugging Face PEFT adapter format from disk.

### 9.2 Scope

- **Backup:** Current Unsloth-based script copied to `training/lean_sdpo_qwen_3b_lora_modal_unsloth_backup.py`.
- **In scope:** `lean_sdpo_qwen_3b_lora_modal.py` — replace Unsloth model load and LoRA setup with HF + PEFT; replace Modal image to drop Unsloth and add bitsandbytes + compatible HF stack.
- **Out of scope:** Kimina, vLLM, SDPO loss logic, parsing, dataset loading, CLI — no changes except where they touch model load/save.

### 9.3 Plan (implementation order)

1. **Modal image**
   - Remove: `unsloth[cu128-torch270]`, `unsloth_zoo`, `trl`.
   - Keep: `vllm`, `datasets`, `matplotlib`, `httpx`, `sentencepiece`, `protobuf`.
   - Add: `transformers`, `peft`, `bitsandbytes`, `accelerate` with versions compatible with each other and with vLLM. Use PyTorch with CUDA (e.g. install `torch` with CUDA index or use a base image that provides it).

2. **Model loading (`_setup_trainer`)**
   - Replace `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` with:
     - `AutoTokenizer.from_pretrained(model_name)`
     - `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")` (or similar)
     - `AutoModelForCausalLM.from_pretrained(model_name, quantization_config=..., device_map="auto", torch_dtype=torch.bfloat16)`
   - Replace `FastLanguageModel.get_peft_model(...)` with:
     - `prepare_model_for_kbit_training(model)` (from PEFT)
     - `LoraConfig(r=..., lora_alpha=..., target_modules=LORA_TARGET_MODULES, lora_dropout=0, bias="none")`
     - `get_peft_model(model, lora_config)` (from PEFT)
   - Remove all Unsloth-specific calls (`gradient_checkpointing_disable`, etc.). Optionally call `model.gradient_checkpointing_enable()` (standard PyTorch) if memory is tight; for 3B 4-bit it is likely unnecessary.
   - Keep: same `LORA_WEIGHT_PATH`, same `save_pretrained(LORA_WEIGHT_PATH)` and tokenizer save; vLLM continues to load from this path.

3. **vLLM**
   - No code change. vLLM’s `LoRARequest(lora_path=...)` accepts a directory with Hugging Face PEFT adapter files (`adapter_config.json`, `adapter_model.safetensors`), which PEFT’s `save_pretrained` produces.

4. **Docstrings and comments**
   - Update top-level docstring and any comments that say “Unsloth” for the training path to “Hugging Face 4-bit + PEFT LoRA”.
   - Update image comment to “Hugging Face 4-bit + PEFT LoRA, vLLM with LoRA overlay”.

### 9.4 Risks and notes

- **Memory:** 4-bit + LoRA via bitsandbytes + PEFT is comparable to Unsloth; A100-40GB should still be sufficient. If OOM appears, enable standard `gradient_checkpointing_enable()` (no Unsloth custom checkpointing).
- **Compatibility:** Ensure `bitsandbytes` and `accelerate` versions are compatible with the installed `transformers` and CUDA driver on Modal (e.g. use versions that support the same PyTorch/CUDA).
- **LoRA format:** PEFT writes `adapter_config.json` and `adapter_model.safetensors`; vLLM reads this format — no extra conversion.
- **Backup:** The Unsloth version remains in `lean_sdpo_qwen_3b_lora_modal_unsloth_backup.py` for reference or rollback.

### 9.5 Implementation (done)

- **Backup:** `training/lean_sdpo_qwen_3b_lora_modal_unsloth_backup.py` created.
- **Image:** Unsloth/unsloth_zoo/trl removed; `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `vllm`, plus datasets/matplotlib/httpx/sentencepiece/protobuf added.
- **Model load:** `AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(...), device_map="auto")`, `AutoTokenizer.from_pretrained`, `prepare_model_for_kbit_training`, `LoraConfig`, `get_peft_model`. Same `LORA_WEIGHT_PATH` and save/load; vLLM unchanged.
- **Docstrings:** Updated to "Hugging Face 4-bit + PEFT LoRA" where appropriate.
