# Devlog: `lean_sdpo_deepseek_7b_modal.py`

Single-file devlog for the **DeepSeek-Prover-V2-7B** SDPO training script on Modal. Check here for purpose, design, and a dated changelog.

---

## Quick reference

| Item | Value |
|------|--------|
| **File** | `training/lean_sdpo_deepseek_7b_modal.py` |
| **Model** | `deepseek-ai/DeepSeek-Prover-V2-7B` |
| **Run (Modal)** | `modal run lean_sdpo_deepseek_7b_modal.py --problem-idx 0` |
| **Run (local test)** | `python lean_sdpo_deepseek_7b_modal.py --test-extraction` |
| **Output (remote)** | Modal volume `sdpo-output` → `deepseek-sdpo-results/` |
| **Output (local)** | `sdpo_results/deepseek_7b/run_{problem_idx}_{timestamp}/` |

---

## Changelog (newest first)

Use this section to see what changed and when. Add new entries at the **top** under the next date.

### 2026-03-01 (Unsloth + config compatibility)

- **rope_scaling:** DeepSeek-Prover-V2-7B `config.json` uses integer `factor` (16), `beta_fast` (32), `beta_slow` (1). Transformers/Unsloth expect floats, so loading logged warnings and could break. **Fix:** Pre-download model to a temp dir and rewrite `config.json` so all `rope_scaling` numeric fields are floats.
- **Chat template / add_generation_prompt:** Unsloth’s `fix_chat_template()` requires the tokenizer’s chat_template to contain the literal `{% if add_generation_prompt %}`. DeepSeek’s template uses `{% if add_generation_prompt and not ns.is_last_user ... %}`, so Unsloth’s check fails and raises RuntimeError. **Fix:** In the same pre-download step, patch `tokenizer_config.json` by appending a no-op `{% if add_generation_prompt %}{% endif %}` so the check passes; generation still uses the rest of the template.
- **Implementation:** Added `_patch_deepseek_model_for_unsloth()` (downloads via `huggingface_hub.snapshot_download`, patches config + tokenizer, returns path). Setup loads Unsloth and vLLM from this patched path when the model name is DeepSeek-Prover. See “Unsloth / vLLM compatibility” below.

### 2026-03-01 (naming)

- **8b → 7b:** Renamed script to `lean_sdpo_deepseek_7b_modal.py`, devlog to `LEAN_SDPO_DEEPSEEK_7B_MODAL.md`. App name `lean-sdpo-deepseek-7b`, local output dir `sdpo_results/deepseek_7b`. All usage examples and references updated.

### 2026-03-01

- **Adaptation from Goedel:** Script created by adapting `lean_sdpo_goedel_8b_modal.py` to train DeepSeek-Prover-V2-7B instead of Goedel-Prover-V2-8B.
- **Model & tokens:** Default model set to `deepseek-ai/DeepSeek-Prover-V2-7B`. Stop tokens and strip logic updated for DeepSeek tokenizer (LlamaTokenizerFast, `</think>`-style EOS/chat tags). Added both ASCII and Unicode variants for EOS and User/Assistant tags in config and in `_strip_special_tokens_from_generation()`.
- **Parsing:** DeepSeek can output think/assistant tags in ASCII (`</think>`, `<think>`) or Unicode (`\u300c...\u300d`). Added `_THINK_END_MARKERS`, `_THINK_START_MARKERS`, `_ASSISTANT_START_MARKERS` and helpers `_split_after_last_think_end()`, `_has_think_start()`, `_has_think_end()`. Tactic extraction (A/B) and truncation detection now use these so both tag styles are handled.
- **Prompt:** Wording aligned with DeepSeek-Prover-V2-7B README quick start; prompt structure unchanged (problem in ` ```lean4 ``` ` + proof plan instruction).
- **Branding:** App name `lean-sdpo-deepseek-7b`, output dir `deepseek-sdpo-results`, local dir `sdpo_results/deepseek_7b`; docstrings and CLI messages updated.

---

## 1. File purpose

- **What:** SDPO (Self-Distilled Policy Optimization) for Lean 4 theorem proving using **DeepSeek-Prover-V2-7B** on Modal.
- **How:** One GPU runs two model instances: **vLLM** (bf16 + LoRA overlay) for inference, **Unsloth** (4-bit + LoRA) for gradients. Only LoRA is trained; base is frozen. Gradient accumulation over `gradient_accumulation_steps` (default 4) before each optimizer step.
- **Student/Teacher:** Student sees problem only; teacher sees problem + error feedback. Loss = KL(student || teacher) on generated proof tokens.

---

## 2. Unsloth / vLLM compatibility (DeepSeek-Prover)

The repo `deepseek-ai/DeepSeek-Prover-V2-7B` is not fully compatible with Unsloth and strict config checks out of the box:

| Issue | Cause | Fix in script |
|-------|--------|----------------|
| **rope_scaling type** | `config.json` has `rope_scaling.factor` (16), `beta_fast` (32), `beta_slow` (1) as integers; libraries expect floats. | Before load: download model to `/tmp/deepseek_prover_7b_patched`, convert those keys to float in `config.json`. |
| **add_generation_prompt** | Unsloth checks for exact substring `{% if add_generation_prompt %}` in the chat template. DeepSeek uses `{% if add_generation_prompt and ... %}`, so the check fails. | In the same patched copy, append `{% if add_generation_prompt %}{% endif %}` to the template in `tokenizer_config.json`. |

Setup only runs this patch when the model name indicates DeepSeek-Prover; other models load from Hugging Face as usual.

### 2.1 Bug 1: `rope_scaling` type mismatch (detailed)

**What happens:** When Unsloth (and the underlying transformers config) load `config.json`, they validate `rope_scaling`. The DeepSeek-Prover-V2-7B repo ships a config like:

```json
"rope_scaling": {
  "type": "yarn",
  "factor": 16,
  "beta_fast": 32,
  "beta_slow": 1,
  "original_max_position_embeddings": 4096,
  "mscale": true
}
```

Here `factor`, `beta_fast`, and `beta_slow` are **integers**. The transformers/Unsloth code paths that build RoPE (rotary position embeddings) expect these to be **floats** (e.g. `16.0`, `32.0`, `1.0`). The validation is strict: it checks the type and logs (or can raise):

- `'rope_scaling''s factor field must be a float >= 1, got 16`
- `'rope_scaling''s beta_fast field must be a float, got 32`
- `'rope_scaling''s beta_slow field must be a float, got 1`

So the **bug** is a schema/type mismatch between what the model card ships and what the library expects. The values are numerically correct; only the type is wrong.

**Why it matters:** Depending on library version, this can be warnings only or can cause later code to fail when it uses these fields (e.g. in tensor creation or scaling math). Fixing the types removes the warnings and avoids any downstream type-sensitive logic.

**Fix (what we do):** Before calling `FastLanguageModel.from_pretrained()`, we download the model to a local directory (e.g. `/tmp/deepseek_prover_7b_patched`). We then:

1. Read `config.json` from that directory.
2. If `rope_scaling` exists and is a dict, we iterate over `factor`, `beta_fast`, and `beta_slow`; for any key that is present and has an **integer** value, we set `rs[key] = float(rs[key])`.
3. Write the modified config back to `config.json`.

We do **not** change any other keys (e.g. `type`, `original_max_position_embeddings`, `mscale`). Unsloth and vLLM are then pointed at this patched directory, so they load the same numeric values but as floats. Behavior of RoPE is unchanged; only the type seen by the library is corrected.

---

### 2.2 Bug 2: Unsloth chat-template check for `add_generation_prompt` (detailed)

**What happens:** When Unsloth loads a tokenizer, it runs a helper called `fix_chat_template()` (in `unsloth/tokenizer_utils.py`). That function **requires** the tokenizer’s chat template (a Jinja2 string) to contain the **exact** substring:

```text
{% if add_generation_prompt %}
```

If that substring is not found, Unsloth raises:

```text
RuntimeError: Unsloth: The tokenizer `deepseek-ai/DeepSeek-Prover-V2-7B`
does not have a {% if add_generation_prompt %} for generation purposes.
```

So the **bug** is that Unsloth’s check is a simple string match: it looks for that exact phrase. It does **not** accept variants like `{% if add_generation_prompt and ... %}`.

**What DeepSeek’s template actually has:** The DeepSeek-Prover-V2-7B `tokenizer_config.json` uses a chat template that *does* use `add_generation_prompt`, but inside a longer condition, for example:

```jinja2
{% if add_generation_prompt and not ns.is_last_user and not ns.is_tool %}{{'</think>'}}{% endif %}
```

So the template:

- Does respect `add_generation_prompt` for when to add the assistant turn.
- Does **not** contain the exact substring `{% if add_generation_prompt %}` (there is no `%}` immediately after `add_generation_prompt`; instead there is ` and not ...`).

So the **root cause** is the combination of (1) Unsloth’s strict substring check and (2) DeepSeek’s template using a compound condition. Functionally the template is correct; only the check fails.

**Why it matters:** Without passing this check, Unsloth refuses to continue and we never get to load the model. So we must make the template string contain that exact substring somewhere.

**Fix (what we do):** We do **not** rewrite or simplify DeepSeek’s full template (that could change behavior). We only make the check pass:

1. In the same patched directory, we read `tokenizer_config.json` and get the `chat_template` string.
2. If that string **does not** already contain `{% if add_generation_prompt %}`, we **append** a no-op block to the end of the template:
   ```jinja2
   {% if add_generation_prompt %}{% endif %}
   ```
3. We write the updated `tokenizer_config.json` back.

Effects:

- The template string now contains the exact substring Unsloth looks for, so `fix_chat_template()` passes.
- The **rest** of the template is unchanged. All of DeepSeek’s original logic (including `{% if add_generation_prompt and not ns.is_last_user and not ns.is_tool %}...`) still runs first. The appended block is a no-op (it renders nothing when `add_generation_prompt` is true or false), so it does not change the formatted output. So generation and training behavior stay the same; we only satisfy the checker.

---

### 2.3 How the patch is applied (implementation flow)

1. **When:** In `_setup_trainer()`, before any `FastLanguageModel.from_pretrained()` or `LLM()` call. We only run the patch when the model name indicates DeepSeek-Prover (e.g. `"DeepSeek-Prover" in trainer_self.model_name` or `"deepseek-ai/DeepSeek-Prover" in trainer_self.model_name`).

2. **Download:** We call `huggingface_hub.snapshot_download(model_name, local_dir=patched_dir, local_dir_use_symlinks=False)`. That puts a full copy of the repo (config, tokenizer, weights, etc.) into e.g. `/tmp/deepseek_prover_7b_patched`.

3. **Patch config:** We open `patched_dir/config.json`, fix `rope_scaling.factor` / `beta_fast` / `beta_slow` to floats as above, and save.

4. **Patch tokenizer:** We open `patched_dir/tokenizer_config.json`, and if the chat_template does not contain `{% if add_generation_prompt %}`, we append `{% if add_generation_prompt %}{% endif %}` and save.

5. **Load from patched dir:** We set `model_to_load = patched_dir` and pass that to:
   - `FastLanguageModel.from_pretrained(model_name=model_to_load, ...)` (Unsloth).
   - `LLM(model=model_to_load, ...)` (vLLM).

So both Unsloth and vLLM see the **same** patched config and tokenizer; neither loads the original Hugging Face repo directly when we use DeepSeek-Prover. The original repo is never modified; only our local copy in `patched_dir` is.

---

## 3. Model and tokenizer (DeepSeek-specific)

- **Model:** `deepseek-ai/DeepSeek-Prover-V2-7B` (7B is the small Prover).
- **Tokenizer:** LlamaTokenizerFast. EOS/pad use `</think>`-style tokens (Unicode U+300C / U+300D possible in decoded text). Chat format: `</think>` (User) then `</think>` (Assistant) for generation.
- **Stop tokens (config):** ASCII `<|endofsentence|>`, `<|endoftext|>`, `</think>`, `</think>`; Unicode `\u300cendofsentence\u300d`, `\u300cUser\u300d`, `\u300cAssistant\u300d`; plus common fallbacks. vLLM stops on these so generation doesn’t start a new turn.
- **Strip:** `_strip_special_tokens_from_generation()` removes trailing EOS variants and leading assistant/think markers (ASCII and Unicode) before tactic extraction.

---

## 4. Parsing (tactics and think tags)

- **Think/assistant tags:** Model may output ASCII (`</think>`, `<think>`) or Unicode (`\u300c/think\u300d`, `\u300cthink\u300d`, `\u300cAssistant\u300d`). Constants and helpers in the file support both; extraction (A: after last `</think>`, B: inside `<think>...</think>`) and truncation (`<think>` without `</think>`) use them.
- **Tactic extraction:** Same priority as Goedel script: (A) after last think-end, (B) inside think region, (C) any code block (best by score), (D) `:= by` fallback. Best block = fewer `sorry` then longer. `_extract_tactics_from_code_block()` strips theorem header through `:= by` and keeps import/open/set_option.
- **Truncation:** `is_truncated = _has_think_start(raw_output) and not _has_think_end(raw_output)`; then quick-reject (no Lean verify).

---

## 5. Prompts

- **Student:** `_create_base_prompt()` — one user message: “Complete the following Lean 4 code:” + ` ```lean4 ``` ` with full statement (header + theorem) + “Before producing the Lean 4 code… provide a detailed proof plan…” (matches DeepSeek-Prover README).
- **Teacher:** `_create_feedback_prompt()` — same structure + “Previous proof attempts produced these errors: … Avoid these errors.” + same proof-plan instruction. Feedback is errors-only by default (`feedback_include_failed_proof=False`).
- **Chat format:** `tokenizer.apply_chat_template(..., add_generation_prompt=True)` so the model gets correct User/Assistant framing.

---

## 6. Output and layout

- **Remote:** Modal volume `sdpo-output`, prefix `deepseek-sdpo-results/run_{timestamp}/` (iter artifacts, `logs.json`, `final_lora`, `training_curves.png`, `metrics.json`).
- **Local:** `sdpo_results/deepseek_7b/run_{problem_idx}_{timestamp}/` (logs, metrics, plots).

---

## 7. Where things live in the file

| Concern | Location / symbol |
|--------|--------------------|
| Config | `SDPOConfig` (top) |
| DeepSeek Unsloth/config patch | `_patch_deepseek_model_for_unsloth()`, `DEEPSEEK_PATCHED_DIR` |
| DeepSeek think/assistant tags | `_THINK_*_MARKERS`, `_split_after_last_think_end`, `_has_think_start`, `_has_think_end` |
| Tactic extraction | `_extract_proof_tactics()`, `_extract_tactics_from_code_block()` |
| Modal app / images | `app`, `inference_image`, `kimina_image` |
| Trainer setup | `_setup_trainer()` |
| SDPO loop | `SDPOTrainer.run_sdpo()` |
| Prompts | `_create_base_prompt()`, `_create_feedback_prompt()` |
| Generation / strip | `_generate_proof()`, `_strip_special_tokens_from_generation()` |
| Full Lean code | `_create_full_lean_code()` |
| Loss / step | `_compute_sdpo_loss()`, optimizer at accumulation boundary |
| Results | `_save_results()`, main() local copy |

---

## 8. Related devlogs

- **Bugfixes (Goedel/Modal):** `devlog/BUGFIXES_20260301.md` — verification timeouts, LoRA targets, quick-reject, Kimina response handling (same pipeline ideas apply).
- **Goedel script summary:** `devlog/GOEDEL_8B_SDPO_CHANGES.md` — same SDPO design; this devlog is the DeepSeek-7B counterpart.

---

*Last updated: 2026-03-01. Add new changelog entries at the top under the current date.*
