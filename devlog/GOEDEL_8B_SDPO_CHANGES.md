# Goedel-8B SDPO Modal Pipeline — Summary of Important Changes

This document summarizes the important changes in `lean_sdpo_goedel_8b_modal.py` relative to the original Kimina-based SDPO setup. The philosophy is **modular, containerized code** with clear comments so failures are localized.

---

## 1. File purpose (TLDR)

- **What:** Self-Distilled Policy Optimization (SDPO) for Lean 4 theorem proving, using **Goedel-LM/Goedel-Prover-V2-8B** on Modal.
- **How:** Two model instances on one GPU — **vLLM** for fast inference (bf16 + LoRA overlay), **Unsloth** for 4-bit quantized LoRA training. Only LoRA weights are trained; base model is frozen. Gradients are accumulated over a configurable minibatch (default 4 problems) before each optimizer step.
- **Student/Teacher:** Student gets problem only; teacher gets problem + error feedback. Student learns by minimizing KL to the teacher distribution.

---

## 2. Model and prompt format

- **Model:** Switched from Kimina-Prover-1.7B to **Goedel-LM/Goedel-Prover-V2-8B** (Qwen2/Qwen3 chat, tokenizer: `eos_token=<|im_end|>`, `pad_token=<|endoftext|>`).
- **Prompt format (Goedel):**
  - **User-only** messages (no system prompt).
  - Full formal statement (imports + theorem) in a ` ```lean4 ... ``` ` block.
  - Instruction to provide a **proof plan before** producing Lean 4 code.
- **Student prompt:** Built in `_create_base_prompt()` — problem block + "provide a detailed proof plan…".
- **Teacher prompt:** Same structure in `_create_feedback_prompt()`, with an extra section: "Previous proof attempts produced these errors: … Avoid these errors." Feedback can be errors-only or include failed proof snippets (configurable).

---

## 3. Output and results layout

- **Output directory:** Dedicated dir for Goedel runs: `output_dir = "goedel-sdpo-results"` (Modal volume path prefix).
- **Local copy:** Results are also written under `sdpo_results/goedel_8b/run_{problem_idx}_{timestamp}/` (logs, metrics, plots). Path is configurable via the entrypoint.

---

## 4. LoRA with Unsloth (no full fine-tuning)

- **Library:** **Unsloth** for 4-bit quantized base + LoRA adapters; no PEFT used for the adapter definition (Unsloth provides the LoRA layers).
- **Config (in `SDPOConfig`):**
  - `lora_rank` (default 16), `lora_alpha` (default 32), `lora_dropout` (0), `lora_bias` ("none").
- **Training:** Base model frozen; only LoRA parameters are trainable. Unsloth saves LoRA weights to a temp dir (e.g. `/tmp/sdpo_lora_weights/v0`, `v1`, …).
- **Inference:** vLLM loads the **same** LoRA via `LoRARequest(..., lora_path=...)` so that inference uses the updated adapter after each optimizer step (or at start, the initial LoRA).
- **Weight flow:** Unsloth saves → path passed to vLLM's `LoRARequest`; vLLM overlays LoRA on the bf16 base. No full-model copy for training.

---

## 5. Gradient accumulation (minibatch)

- **Before:** Effectively one problem per step (update every iteration).
- **After:** **Gradient accumulation** over `gradient_accumulation_steps` (default 4). For each "step" we still do one inference (one problem), but we only call `optimizer.step()` and reload LoRA into vLLM when we have accumulated `gradient_accumulation_steps` gradients (e.g. after 4 failed attempts). Success on any iteration still triggers early exit and optional saving.
- **Config:** `SDPOConfig.gradient_accumulation_steps = 4`.
- **Code:** Accumulation counter and step/zero_logic live in the main SDPO loop; "accum 1/4", "accum 2/4", etc., are logged.

---

## 6. Proof tactic extraction (parser)

Two main improvements so the **correct** proof block is used and tactic text is not over-filtered.

### 6.1 Choosing the right code block

- **Issue:** The model often outputs a **sketch** (with `sorry`) first and the **complete proof** in a later ` ```lean4 ``` ` block. Taking the *first* block led to verification failure (e.g. two `sorry` lines).
- **Change:** In Strategies 1 and 3 of `_extract_proof_tactics()`, instead of stopping at the first "valid" block we **score every** code block and pick the **best** one:
  - Score = `(-sorry_count, length)` (fewer `sorry` is better; then longer is better).
  - So the full proof block (0 sorry, long) wins over the sketch (several sorry, short).

### 6.2 Extracting tactics from a single block

- **Issue:** The old `_extract_tactics_from_code_block()` used many regex filters to drop "non-tactic" lines. Those filters removed valid tactic lines (e.g. `have h_final : 4 * x^3 - 7 * y^3 ≠ 2003 := by ...` was dropped due to patterns like `≠ ... :=`).
- **Change:** Rewritten to a **header-stripping** strategy:
  - Scan for the theorem/lemma declaration and the `:= by` that starts the tactic body.
  - **Keep** everything after `:= by` (and preserve `import`/`open`/`set_option` for header use). No content-based regex that deletes lines containing `≠`, `:= sorry`, etc., so valid `have` lines and full tactic bodies are preserved.

### 6.3 Final validation

- **Before:** Rejecting any tactics string containing `":= sorry"` could reject useful partial proofs.
- **After:** Only reject **trivially** empty bodies (e.g. exactly `"sorry"`, `"by"`, `"by sorry"` or very short). Proofs that still contain some `sorry` in sub-goals are allowed; the Lean verifier and the block-scoring logic handle them.

---

## 7. GPU and memory (Modal)

- **GPU:** **A100-80GB** (script is written for 80GB; 40GB was insufficient for vLLM + Unsloth without truncation).
- **vLLM:** `gpu_memory_utilization=0.35` so vLLM and Unsloth fit on the same GPU. vLLM runs in a spawned process and sees the full GPU; the cap leaves room for Unsloth's 4-bit + LoRA training memory.
- **vLLM options:** `max_model_len=10240`; no `enforce_eager` by default (CUDA graphs allowed for speed). No FlashInfer in the image (nvcc not in the build env); sampling uses vLLM's built-in path.
- **LoRA:** Initial LoRA is saved to a temp dir; after training, final LoRA is written to the Modal volume under `goedel-sdpo-results/run_.../final_lora`.

---

## 8. Dependency and env notes

- **Unsloth**, **vLLM**, **transformers** (e.g. 4.53.2), **trl** (e.g. 0.19.1) are pinned in the Modal image to avoid `ConstantLengthDataset` and `aimv2` config conflicts.
- **Special tokens:** `_strip_special_tokens_from_generation()` strips `<|im_end|>`, `<|endoftext|>`, and any leading `<|im_start|>assistant` from vLLM output before parsing.

---

## 9. Entrypoint and CLI

- **Entrypoint:** `run_sdpo_goedel()` (or equivalent) loads dataset, picks the problem by index, builds `SDPOConfig`, and launches the trainer on Modal.
- **CLI:** e.g. `--problem-idx`, `--max-iterations`, `--lora-rank`, `--gradient-accumulation-steps`. Local results directory is derived from problem index and timestamp.

---

## 10. Where things live in the code (modular layout)

| Concern              | Location / symbol |
|----------------------|-------------------|
| Config               | `SDPOConfig` (top of file) |
| Modal app / images   | `app`, `inference_image` |
| Trainer setup        | `_setup_trainer()` (Unsloth + vLLM init) |
| SDPO loop            | `SDPOTrainer.run_sdpo()` |
| Student/teacher prompts | `_create_base_prompt()`, `_create_feedback_prompt()` |
| Generation           | `_generate_proof()` (vLLM + LoRARequest) |
| Parsing              | `_extract_proof_tactics()`, `_extract_tactics_from_code_block()` |
| Full Lean code       | `_create_full_lean_code()` |
| Loss / training step| `_compute_sdpo_loss()`, optimizer step at accumulation boundary |
| Results / persistence| `_save_results()`, volume + local copy |

Keeping these boundaries clear makes it easier to fix a single part (e.g. parser or LoRA reload) without touching the rest.
