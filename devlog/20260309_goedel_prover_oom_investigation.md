# OOM Investigation: Goedel Prover SDPO Training

**Date:** 2026-03-09  
**Topics:** goedel, sdpo, oom, memory, modal, config, weight_sync, bugfixes

---

## Identified OOM location

The codebase explicitly documents OOM **during the optimizer step** for 8B models:

- **Source:** [sdpo_modal_local_verify_goedel/utils.py](sdpo_modal_local_verify_goedel/utils.py) line 531:  
  `"8B models have memory issues (OOM during optimizer step)."`

So the failure is in the **training step** on Modal, not at vLLM init or at verification time.

## Where exactly OOM can happen (same GPU, one process)

Everything runs on one GPU in this order inside `run_sdpo_step`:

1. **`compute_sdpo_loss(...)`**  
   - **Student forward** (with grad): `student_input_ids` = prompt (≤2048) + response (≤8192) → up to **~10,240 tokens**.  
   - **Teacher forward** (no_grad): same length.  
   - **Backward** through the student forward only.  
   - **Peak here:** Activations for the student forward (with gradient checkpointing) plus gradients and optimizer state. For an 8B model and 10k tokens, this is the **most likely OOM point**.

2. **`optimizer.step()`**  
   - Updates LoRA parameters; Adam state is already allocated.  
   - Small extra temporary memory.  
   - Possible OOM if the previous step left the GPU almost full, but usually the **backward** is the main consumer.

3. **`sync_lora_weights_to_vllm_goedel(...)`**  
   - **Trainer side:** Builds `weight_dict` on CPU (merged LoRA weights), one layer at a time; no large GPU spike.  
   - **Worker side** (same process with `VLLM_ENABLE_V1_MULTIPROCESSING=0`):  
     In [sdpo_modal_local_verify_goedel/_weight_sync_goedel.py](sdpo_modal_local_verify_goedel/_weight_sync_goedel.py) `update_weights_from_numpy`, we do:
     ```python
     for param_name, arr in weight_dict.items():
         tensor = torch.from_numpy(...).to(device=self.device, dtype=torch.bfloat16)
         weights.append((param_name, tensor))
     self.model_runner.model.load_weights(weights=weights)
     ```
     So **all** merged weight tensors (~224 layers × ~32MB ≈ **~7 GB**) are on GPU at once before `load_weights`.  
   - **Second OOM candidate:** If vLLM + HF training already use ~73GB on an 80GB H100, this **+7 GB** during sync can trigger OOM.

## Memory breakdown (H100 80GB, order-of-magnitude)

| Component | Size (GB) |
|-----------|-----------|
| vLLM (0.4 util, max_model_len=16384) | ~18–20 (weights + KV) |
| HF 4-bit base + LoRA + Adam state | ~5–6 |
| **HF activations (forward + backward, 10k seq)** | **~15–25** (dominant) |
| Weight-sync peak (merged tensors on GPU) | ~7 |
| **Total peak** | **~45–58** (best case) to **~55–58** (long seq + sync) |

With long generations (e.g. 8k-token responses), activation memory for the **student forward + backward** dominates and can push you over 80GB, especially if vLLM has already reserved a large chunk.

## Root cause summary

- **Primary:** **SDPO forward/backward** in `compute_sdpo_loss` with **long sequences** (prompt ≤2048 + response ≤8192). Gradient checkpointing is on but 8B × 10k tokens still needs a lot of activation memory.
- **Secondary:** **Weight-sync** temporarily materializing all merged LoRA weights on GPU in `update_weights_from_numpy` adds ~7 GB peak and can tip an already tight run over.

## Mitigation strategies

### 1. Reduce SDPO sequence length (most effective)

- **Cap response length used in the loss** (e.g. first 2048 or 4096 tokens of `generated_ids` for the backward).  
  You still generate up to 8192 for the proof, but only backprop through a shorter span so activation memory drops.
- **Lower `max_new_tokens`** for the SDPO run (e.g. 4096) so both generation and loss use shorter sequences.
- **Shorten prompt truncation** in [sdpo_modal_local_verify_goedel/sdpo_loss.py](sdpo_modal_local_verify_goedel/sdpo_loss.py) (e.g. `max_length=1536` for student/teacher prompts) to reduce total sequence length.

**Where:** [sdpo_modal_local_verify_goedel/sdpo_loss.py](sdpo_modal_local_verify_goedel/sdpo_loss.py) (truncation `max_length` and optionally slicing `response_ids` / `generated_ids` before the forward).

### 2. Lower vLLM memory share

- Reduce **`gpu_memory_utilization`** for vLLM (e.g. from 0.4 to 0.35) so more GPU memory is left for HF training and weight sync.  
  Slightly smaller KV cache; may reduce max batch or context length a bit.

**Where:** [sdpo_modal_local_verify_goedel/modal_app.py](sdpo_modal_local_verify_goedel/modal_app.py) `GPU_VLLM_MEMORY` / `DEFAULT_GPU_MEMORY`.

### 3. Chunked weight sync

- In **`update_weights_from_numpy`**, apply weights in **chunks** (e.g. 32 layers at a time): build a small `weights` list, call `load_weights`, clear the list, repeat.  
  Peak GPU memory during sync drops from ~7 GB to ~1–2 GB.

**Where:** [sdpo_modal_local_verify_goedel/_weight_sync_goedel.py](sdpo_modal_local_verify_goedel/_weight_sync_goedel.py) (worker extension and/or the code that builds `weight_dict` and calls `collective_rpc`).

### 4. Clear cache before sync

- After `optimizer.step()`, call `torch.cuda.empty_cache()` before `sync_lora_weights_to_vllm_goedel` to return fragmented memory to the allocator.  
  Helps if the OOM is right at the sync step.

**Where:** [sdpo_modal_local_verify_goedel/modal_trainer.py](sdpo_modal_local_verify_goedel/modal_trainer.py) right before `sync_lora_weights_to_vllm_goedel`.

### 5. Reduce LoRA rank or target modules

- Use **smaller LoRA rank** (e.g. `r=8` instead of 16) or **fewer target modules** (e.g. only `q_proj`, `v_proj`, `o_proj`) to shrink optimizer state and sync size.  
  Slightly less capacity, but can make the difference for OOM.

**Where:** [sdpo_modal_local_verify_goedel/modal_app.py](sdpo_modal_local_verify_goedel/modal_app.py) `LoraConfig(r=..., target_modules=...)`.

### 6. Confirm exact OOM site (optional)

- Add **`torch.cuda.synchronize()`** and **`torch.cuda.memory_allocated()` / `torch.cuda.max_memory_allocated()`** around:
  - end of `compute_sdpo_loss` (after backward),
  - after `optimizer.step()`,
  - after `sync_lora_weights_to_vllm_goedel`.  
  Log these in Modal; the last successful print before the crash is right before the OOM.

---

**Recommended order:** (1) cap SDPO sequence length or lower `max_new_tokens`, then (2) lower vLLM `gpu_memory_utilization`, then (3) chunked weight sync and (4) `empty_cache` if still OOM.
