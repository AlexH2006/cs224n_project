# SDPO Trainer — Deep Dive (Current Code)

This document reflects the **current** state of `lean_sdpo_modal.py` (1545 lines) and provides a line-by-line style breakdown of the trainer plus a workflow diagram.

---

## 1. Recent / Notable Updates (vs earlier version)

| Area | Change |
|------|--------|
| **Model reset** | **Removed.** No more `initial_state` or `reload_model()`. The comment in `run_sdpo` states: *"Model weights are NOT reset between problems - this is intentional! SDPO is a test-time training method where the model accumulates learning across problems."* So the same Modal container can run multiple problems and the model keeps updated weights. |
| **Generation** | `max_new_tokens` increased to **8192** (from 4096) for long thinking outputs. |
| **Sorry replacement** | **Only the last `sorry`** in the theorem file is replaced with tactics (fixes Putnam-style problems with `abbrev ... := sorry` and `theorem ... := sorry`). Explicit patterns `:= by sorry` and `:= by\n  sorry` are still handled first. |
| **Config** | Config now carries **field name lists**: `theorem_fields`, `informal_fields`, `header_fields`, `id_fields`. Optional overrides: `system_prompt`, `default_header`. |
| **main()** | More robust dataset loading (e.g. `trust_remote_code`, fallback splits). CLI overrides: `--theorem-field`, `--informal-field`, `--header-field`, `--system-prompt`, `--default-header`. Built `config_dict` includes the field lists and optional overrides. |
| **Iteration logs** | Each `iter_log` stores `student_prompt` and `teacher_prompt` (or `current_teacher_prompt`) for analysis. |

---

## 2. SDPOConfig (used by the trainer)

- **Model:** `model_name`.
- **Dataset:** `dataset_name`, `dataset_subset`, `dataset_split`, `problem_idx`.
- **Field mapping:** `theorem_fields`, `informal_fields`, `header_fields`, `id_fields` (lists of candidate keys for dataset-agnostic access).
- **Generation:** `max_new_tokens` (8192), `temperature`, `top_p`, `stop_tokens` (list).
- **RL:** `max_iterations`, `learning_rate`, `distillation_topk` (top-K for KL).
- **Prompts:** `system_prompt`, `default_header` (Lean imports when no dataset header and no model imports).
- **Feedback:** `feedback_include_failed_proof`, `feedback_attempt_template`, `feedback_attempt_template_errors_only`, `feedback_separator`.
- **Output:** `output_dir`.

---

## 3. SDPOTrainer — Detailed Breakdown

### 3.1 Class and `setup` (lines ~370–438)

- **Modal:** `@app.cls` with A100-40GB, 1h timeout, 10 min scaledown, volumes `/cache` (HF), `/output` (results), secret `huggingface`.
- **Parameter:** `model_name` so callers can pass a different HF model ID.
- **`setup` (`@modal.enter()`):**
  - Sets `HF_HOME=/cache` and copies `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN` if set.
  - **Tokenizer:** `AutoTokenizer.from_pretrained(model_name)`, left padding, pad = eos if missing.
  - **vLLM:** `LLM(..., gpu_memory_utilization=0.25, max_model_len=4096)` for **generation only** (no gradients).
  - **HF model:** Same weights via `AutoModelForCausalLM`, bfloat16, `device_map="auto"`, `train()`, gradient checkpointing if available.
  - No snapshot of initial weights; the model is trained in place and not reset after a run.

### 3.2 `_get_field` (static, ~441–455)

- **Purpose:** Dataset-agnostic field access.
- **Logic:** Given `data` and a list of keys (e.g. `theorem_fields`), returns the first key’s value that exists and is non-empty. If value is list/tuple, returns `str(value[0])`; if str, returns it. Else `default`.
- Used everywhere the trainer needs theorem code, header, or informal text from a `problem` dict.

### 3.3 `run_sdpo` (lines ~457–630)

**Signature:** `run_sdpo(config_dict, problem, verifier_results=None)`. `verifier_results` is unused.

**Setup:**

- `config = SDPOConfig(**config_dict)`.
- **metrics:** lists for iterations, losses, rewards, kl_divs, entropies, grad_norms, timestamps.
- **logs:** problem, config, iteration_logs, start_time (later: end_time, success, best_proof, metrics, model_save_path).
- **optimizer:** `AdamW(model.parameters(), lr=config.learning_rate)`.
- **feedback_history:** list of `(feedback_str, tactics_str)` for failed attempts.
- **base_prompt:** built once with `_create_base_prompt(config, problem)`. This is the **student** prompt (problem only) and never changes during the run.

**Per iteration:**

1. **Generate**  
   `raw_output, generated_ids = _generate_proof(config, base_prompt)`.  
   Generation is **always** from `base_prompt` (student context only).

2. **Extract tactics**  
   `tactics = _extract_proof_tactics(raw_output)`.

3. **Dataset fields**  
   `lean4_code = _get_field(problem, config.theorem_fields)`, `header = _get_field(problem, config.header_fields)`.

4. **Full Lean file**  
   `full_code = _create_full_lean_code(config, lean4_code, tactics, header)`.

5. **Verify**  
   `LeanVerifier().verify.remote(full_code)` up to 3 times on server errors.

6. **Success**  
   `is_success = verification["success"] and verification["complete"]`.  
   If `tactics.strip().lower() == "sorry"`, force failure and set feedback to forbid `sorry`.

7. **Logging**  
   `current_teacher_prompt = _create_feedback_prompt(...)` if there is feedback_history (for logging).  
   `iter_log` = iteration, student_prompt, teacher_prompt (or current_teacher_prompt), raw_output, extracted_tactics, full_code, verification, success.

8. **If success**  
   Set best_proof, append to logs/metrics (loss/reward/kl/entropy/grad_norm as 0 or None), **break**.

9. **If server error**  
   Append iter_log with server_error, **continue** (no training).

10. **Else (failed proof)**  
    - Append `(feedback, tactics)` to feedback_history.  
    - `teacher_prompt = _create_feedback_prompt(config, problem, feedback_history)`.  
    - `per_token_kl, reward, avg_kl, entropy = _compute_sdpo_loss(config, base_prompt, teacher_prompt, generated_ids)`.  
    - `loss = per_token_kl.mean()`, backward, grad norm, `clip_grad_norm_(1.0)`, optimizer step.  
    - Append iter_log (with full teacher_prompt, loss, reward, kl, entropy, grad_norm, feedback) and same to metrics/timestamps.

**After loop:** Set logs (end_time, success, best_proof, metrics), call `_save_results(config, logs, metrics)` (saves model + logs + plots + metrics), set `logs["model_save_path"]`, **return logs** (no model reset).

### 3.4 `_create_base_prompt` (lines ~672–719)

- **Role:** Student prompt only; no informal, no feedback.
- Gets `lean4_code` and `header` via config field lists; `has_header = bool(header.strip())`.
- **User text:** “Prove the following Lean 4 theorem.” + lean4 block + “output ONLY the proof tactics in a ```lean4 code block”, replace sorry; then either “Do NOT include import/theorem/:= sorry” (if has_header) or “Include necessary import… Do NOT include theorem or := sorry”.
- Renders with `apply_chat_template` (system = `config.system_prompt`, user = user_content, add_generation_prompt=True) or a simple "System: ... User: ... Assistant:" fallback.

### 3.5 `_create_feedback_prompt` (lines ~721–800)

- **Role:** Teacher prompt: same theorem + informal (if any) + all previous failed attempts.
- Gets lean4_code, informal, header via config field lists.
- **User text:** “Prove the following Lean 4 theorem.” + optional “Problem: {informal}” + lean4 block + if feedback_history: “The following N proof attempt(s) failed:” then for each attempt either `feedback_attempt_template` (attempt_num, feedback, failed_proof) or `feedback_attempt_template_errors_only` (attempt_num, feedback), joined by feedback_separator, then “Analyze the errors and provide a corrected proof.” Then “Provide corrected proof tactics” (and optional “include any necessary imports” if no header).
- **System:** `config.system_prompt + " After reasoning, output the proof tactics in a ```lean4 code block."`  
- Same chat-template vs fallback as base.

### 3.6 `_generate_proof` (lines ~802–819)

- Builds vLLM `SamplingParams` from config (temperature, top_p, max_tokens, stop=config.stop_tokens).
- `vllm_engine.generate([prompt], sampling_params)` → first output text.
- Tokenizes that text with the **HF tokenizer** (add_special_tokens=False) → `generated_ids` for the HF model (SDPO loss uses HF model + tokenizer).
- Returns `(generated_text, generated_ids)`.

### 3.7 `_is_degenerate_output` (lines ~821–845)

- Detects repetitive output: if any phrase of length 10/15/20 words appears ≥ threshold (default 5), returns True.
- Used in `_extract_proof_tactics` to return `"sorry"` and avoid training on looping garbage.

### 3.8 `_extract_tactics_from_code_block` (lines ~847–893)

- Input: one code block string.
- Keeps: lines starting with `import `, `open `, `set_option `, and lines that are not filtered out.
- Skips: theorem/lemma lines, parameter lines `(x : T)`, “:= sorry”, conclusion/type-signature lines (several regexes), lines ending with “:=” or “:= by”.
- Returns joined kept lines (tactics + optional header-like lines).

### 3.9 `_extract_proof_tactics` (lines ~895–1006)

- If `_is_degenerate_output(output)`: return `"sorry"`.
- **1)** After `</think>`: look for lean4/lean code blocks or use raw text; run through `_extract_tactics_from_code_block`; take first valid tactics.
- **2)** If no tactics, inside `<think>...</think>`: extract from code blocks there, join all tactic-like blocks.
- **3)** If still none, any code block in full output.
- **4)** If still none, “:= by” pattern: take lines after last “:= by” (up to 10, no comments).
- Clean: strip leading “by”, reject “:= sorry” or tactics that are just “sorry” or too short.
- No keyword heuristics (e.g. “simp” in reasoning); if nothing found, return `"sorry"`.

### 3.10 `_create_full_lean_code` (lines ~1008–1078)

- Splits `proof_tactics` into model_imports, model_opens, model_set_options vs tactic lines; builds `tactics_clean` and `indented_proof`.
- **Sorry replacement (main-theorem only):**
  - If `":= by sorry"` in theorem_code → replace that with `":= by\n  " + indented_proof`.
  - Elif `":= by\n  sorry"` → replace that with `":= by\n  " + indented_proof`.
  - Else: replace **only the last** `"sorry"` (rfind) with indented_proof, so earlier sorries (e.g. abbrev) stay.
- **Header:** If dataset header: use it. Else if model gave imports: use them (+ set_options, opens). Else `config.default_header`.
- Returns `final_header + "\n\n" + theorem_with_proof`.

### 3.11 `_add_tail` (static, lines ~1080–1087)

- Given log-probs over vocab, appends a “tail” bucket so the distribution over [top-k + tail] sums to 1 for KL.

### 3.12 `_compute_sdpo_loss` (lines ~1089–1185)

- Tokenizes base_prompt and teacher_prompt (truncate 2048), concats each with generated_ids → student_input_ids, teacher_input_ids.
- **Student:** HF forward on student_input_ids; logits over **response** positions only.
- **Entropy:** From student logits on response (logsumexp and -p*log(p)).
- **Top-K:** For each position, keep top-K student logits and indices; teacher logits at those indices (no grad).
- **Tail:** `_add_tail` on student and teacher top-k log-probs.
- **KL:** `F.kl_div(teacher, student, log_target=True)` per bucket, sum → per_token_kl.
- **Reward (logging):** From student vs teacher log-prob of actual response tokens (no grad).
- Returns `(per_token_kl, total_reward, avg_kl, entropy)`.

### 3.13 `_save_results` (lines ~1187–1260)

- Creates `/output/{output_dir}/run_{timestamp}/`, saves **current model + tokenizer** in `run_dir/final_model/`, writes logs.json, metrics.json, and if there are iterations a 2×2 plot (loss, grad norm, entropy, KL) as training_curves.png. Returns path to final_model.

---

## 4. main() (local entrypoint)

- Parses CLI (model, dataset, subset, split, problem_idx, max_iterations, lr, temperature, feedback_errors_only, system_prompt, default_header, theorem_field, informal_field, header_field).
- Loads dataset (with fallbacks: trust_remote_code, different splits).
- Builds theorem_fields, informal_fields, header_fields with user overrides at front; gets problem dict and prints id / lean4 / informal / header.
- Builds config_dict (including those field lists and optional system_prompt, default_header).
- `SDPOTrainer(model_name=model).run_sdpo.remote(config_dict, problem)`.
- Prints results and writes local copy under `sdpo_results/run_{problem_idx}_{timestamp}/` (logs, metrics, training_curves.png).

---

## 5. Workflow Illustration

See `SDPO_WORKFLOW.md` in this directory for the Mermaid diagram of the end-to-end and per-iteration flow.
