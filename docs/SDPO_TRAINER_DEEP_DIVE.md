# SDPO Trainer — Deep Dive (qwen_sdpo)

This document describes the **current** SDPO pipeline: **qwen_sdpo** — Qwen 3.5 (4B/9B) with QLoRA on Modal, verification and orchestration on your machine (Kimina Docker). For a high-level workflow and diagram, see **[docs/QWEN_SDPO_WORKFLOW.md](QWEN_SDPO_WORKFLOW.md)**.

---

## 1. Architecture: Local vs Modal

| Where | What |
|-------|------|
| **Local** | Entrypoint (`modal run qwen_sdpo/modal_app.py`), dataset load, prompt building, **parsing** (lean4 block), **verification** (Kimina HTTP), payload construction, saving logs and plots |
| **Modal (H100)** | vLLM generation, HuggingFace QLoRA model for SDPO loss, optimizer step, **LoRA → vLLM weight sync** so the next generation uses the updated policy |

The trainer class **QwenSDPOTrainer** lives on Modal. The **local** driver in `entrypoint.py` runs the loop: call Modal to generate → parse and verify locally → build teacher payloads → call Modal to run one SDPO step (loss + backward + sync).

---

## 2. SDPOConfig (qwen_sdpo)

Defined in **[qwen_sdpo/config.py](../qwen_sdpo/config.py)**.

- **Model / dataset:** `model_name`, `dataset_name`, `dataset_subset`, `dataset_split`, `problem_idx`.
- **Generation:** `max_new_tokens`, `temperature`, `top_p`, `minibatch_size` (samples per iteration).
- **SDPO:** `max_iterations`, `learning_rate`, `distillation_topk` (top-K for KL), `teacher_response_mode` (full_output / answer_only / code_only), `kl_mask_final_code_only`, `feedback_errors_only`.
- **LoRA:** `lora_r`, `lora_alpha`, `lora_dropout`, etc.
- **Verification:** `kimina_url`, `verify_timeout` (used locally).

---

## 3. QwenSDPOTrainer (Modal) — Main Methods

### 3.1 Setup (`@modal.enter()`)

- **vLLM:** `LLM(..., tensor_parallel_size=1)` for **generation only** (no gradients).
- **HuggingFace:** Same base model with **QLoRA** (`BitsAndBytesConfig` 4-bit), `device_map="auto"`, `train()`, gradient checkpointing. Used for SDPO loss.
- **Optimizer:** AdamW on trainable (LoRA) parameters.
- **Weight sync:** After each gradient step, LoRA weights are copied from the HF model into vLLM’s weight tensors in-place ([_weight_sync.py](../qwen_sdpo/_weight_sync.py)), so the next `generate_batch` uses the updated policy (online RL).

### 3.2 `generate_batch(prompts)` (Modal)

- **Input:** List of strings (e.g. `[student_prompt]` for minibatch_size=1).
- **vLLM:** `SamplingParams` from config (temperature, max_tokens, stop); `llm.generate(prompts, sampling_params)`.
- **Output:** For each sample: `raw_output` (text), `generated_ids` (token ids from HF tokenizer, for loss), `finish_reason`. Used by the local driver to parse, verify, and build payloads.

### 3.3 `run_sdpo_step_minibatch(cfg_dict, payloads)` (Modal)

- **Input:** `cfg_dict` (SDPOConfig as dict), `payloads` = list of dicts from local: each has `student_input_ids`, `teacher_input_ids`, `response_mask` / `teacher_response_ids`, optional `cot_len` (for answer_only/code_only), etc.
- **Skip conditions:** If any payload is marked success / server_error / truncated, the step is skipped (no backward).
- **Loss:** For each payload, [sdpo_loss.compute_sdpo_loss](../qwen_sdpo/sdpo_loss.py): student and teacher forward on HF model; KL(student ‖ teacher) on **top-K + tail** logits over response tokens; optional masking (e.g. final code block only). Loss = mean over batch.
- **Backward:** `loss.backward()`, `clip_grad_norm_`, `optimizer.step()`, `optimizer.zero_grad()`.
- **Sync:** `sync_lora_weights_to_vllm()` so vLLM uses the new LoRA weights for the next iteration.

### 3.4 `finalize_run(cfg_dict, logs)` (Modal)

- Saves the current model (HF + LoRA) and logs to the Modal volume (`/output/...`).

---

## 4. Local driver (entrypoint.run_main)

In **[qwen_sdpo/entrypoint.py](../qwen_sdpo/entrypoint.py)**:

1. Load dataset (HuggingFace), pick problem by `problem_idx`.
2. Build **student_prompt** (problem only; fixed for the whole run).
3. **Loop** (up to `max_iterations`):
   - Call **Modal** `trainer.generate_batch.remote([student_prompt] * minibatch_size)`.
   - For each sample: **parse** ([parsing.py](../qwen_sdpo/parsing.py): `extract_full_lean_block_parsed`, `create_full_lean_code`) → **verify** ([_verifier.py](../qwen_sdpo/_verifier.py): HTTP to Kimina) → if success, break and finalize.
   - If failure: build **teacher_prompt** (problem + compiler errors, and optionally failed proof) and build payload (tokenized student/teacher + response masks).
   - Call **Modal** `trainer.run_sdpo_step_minibatch.remote(cfg_dict, payloads)`.
4. After loop or on success: `trainer.finalize_run.remote(...)`, then locally `save_local_run()`, `plot_training_curves()`.

---

## 5. SDPO loss (compute_sdpo_loss)

In **[qwen_sdpo/sdpo_loss.py](../qwen_sdpo/sdpo_loss.py)**:

- **Inputs:** Student and teacher input IDs, response mask or explicit `teacher_response_ids`, optional `cot_len` (for answer_only/code_only modes).
- **Student:** Forward through HF model; logits at **response** positions only.
- **Teacher:** Forward (no grad); same response positions.
- **Top-K + tail:** For each position, keep top-K logits from student; get teacher log-probs at those indices; add a tail bucket so the distribution sums to 1.
- **KL:** `F.kl_div(teacher_probs, student_log_probs, ...)` per position, then sum or mask (e.g. final code only).
- Returns loss (and optional reward/entropy for logging).

---

## 6. Workflow diagram

End-to-end and per-iteration flow: **[docs/QWEN_SDPO_WORKFLOW.md](QWEN_SDPO_WORKFLOW.md)** (ASCII + Mermaid). Rendered image: [qwen_sdpo_workflow.png](qwen_sdpo_workflow.png).

---

## 7. Legacy: lean_sdpo_modal (training/)

The older **training/lean_sdpo_modal.py** pipeline runs **entirely on Modal**: one `SDPOTrainer` class does generation, verification (LeanVerifier on Modal), and SDPO steps in one process. No local verification; no separate “generate_batch” vs “run_sdpo_step_minibatch” RPC. The **qwen_sdpo** pipeline replaces this as the main SDPO path: verification is local (Kimina), and the split between local (parse, verify, orchestrate) and Modal (generate + train + sync) is as above. For reference, the legacy trainer used a single `run_sdpo(config_dict, problem)` with internal generate → extract tactics → full Lean file → verify → feedback prompt → `_compute_sdpo_loss` → backward → step, all on the same container.
