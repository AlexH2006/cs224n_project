# LoRA → vLLM Weight Sync + Cold-Start Latency Fix

**Date:** 2026-03-07  
**Affects:** `qwen_sdpo/`, `sdpo_modal_local_verify_kimina/`

---

## Problem

The SDPO pipeline held two model instances on the same GPU:

1. **vLLM engine** — fast batched generation (bf16, CUDA graphs on)
2. **HuggingFace QLoRA model** — 4-bit base + LoRA adapters for gradient training

These were completely independent. LoRA gradient updates modified only the HF model.
The vLLM engine had no mechanism to receive those updates, so it always generated with
the original frozen base weights. The online RL loop was effectively broken: the model
trained but the generator never improved.

Additionally, every cold Modal container paid a ~220s `torch.compile` warmup penalty
because the compile artifacts were never persisted — the compile cache lived only in
the container's ephemeral filesystem.

---

## Root Cause Analysis

### Frozen generator

After `optimizer.step()` in `run_sdpo_step()`, the LoRA A and B matrices in the HF
model were updated but had nowhere to go. The vLLM engine's weights, stored in GPU
memory at fixed addresses, were never touched.

```
Before fix:
  optimizer.step()   → HF model updated (LoRA A, B change)
  generate_only()    → vLLM uses original frozen base weights  ← bug
```

This meant SDPO was doing real gradient computation and backprop, but generating with
the same policy for all iterations. The teacher-student KL loop was producing valid
loss signals, but they were never improving the generator.

### Compile cache not persisted

vLLM's first inference call on a cold container triggers two sequential phases:
- `torch.compile`: CPU-side JIT compilation of GPU kernels (~200s)
- CUDA graph capture: GPU-side recording of the op sequence (~11s)

Only the compile phase is cacheable across restarts (its output is files on disk).
CUDA graph capture must always re-run because GPU memory addresses change each time.
Previously, the compile cache at `/root/.cache/vllm/torch_compile_cache/` was written
to the container's ephemeral filesystem and discarded on shutdown.

---

## Changes

### 1. `qwen_sdpo/_weight_sync.py` (new file)

Core of the fix. Two components:

**`QwenSDPOWorker(Worker)`** — subclass of `vllm.worker.worker.Worker`, registered
at `LLM()` init via `worker_cls=QwenSDPOWorker`. Adds:

- `get_device_uuid()` — returns the GPU UUID; used to key IPC handle dicts so handles
  are only opened on the GPU that created them.
- `update_weights_from_ipc_handles(ipc_handles)` — opens CUDA IPC handles from the
  training process, reads the merged bf16 weight tensors, and calls
  `self.model_runner.model.load_weights(weights)` to copy values in-place into the
  vLLM weight tensors. The tensor memory addresses do not change — CUDA graphs remain
  valid, no recapture.
- `check_weights_updated(param_name, expected_checksum)` — verifies a specific weight
  tensor matches an expected L2 norm (used for debugging/testing).

**`sync_lora_weights_to_vllm(hf_model, vllm_engine, lora_target_modules)`** — called
by the trainer after each `optimizer.step()`. Steps:

1. `hf_model.merge_adapter()` — dequantizes the 4-bit base weight and folds in the
   LoRA delta: `W_merged = dequant(W_base_4bit) + B @ A * (alpha / r)` in bf16.
   Only LoRA-modified layers are affected.
2. Extract the bf16 `.weight` tensors for all LoRA target modules. Build CUDA IPC
   handles via `torch.multiprocessing.reductions.reduce_tensor()`, keyed by GPU UUID.
3. `vllm_engine.collective_rpc("update_weights_from_ipc_handles", args=(handles,))`
   — dispatches to the vLLM worker process, which opens the IPC handles and calls
   `load_weights()` in-place.
4. `hf_model.unmerge_adapter()` — always called (in a `finally` block), restores the
   4-bit base + separate LoRA A/B matrices so the HF model stays trainable.

The entire sync takes ~milliseconds (GPU arithmetic + memcpy within one GPU). No disk
I/O, no CUDA graph recapture.

**Why CUDA IPC instead of vLLM's native `enable_lora=True`:**

The native LoRA path requires saving adapter weights to disk between steps (~3-5s)
and causes vLLM to recapture CUDA graphs for each new `LoRARequest` ID (~11s per step).
CUDA IPC avoids both: the in-place `load_weights()` preserves tensor addresses, so
existing CUDA graphs remain valid indefinitely.

### 2. `qwen_sdpo/_modal_infra.py`

Added `compile_cache_volume`:

```python
compile_cache_volume = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)
```

This persistent Modal volume is mounted at `/root/.cache/vllm/torch_compile_cache/`
in the trainer container. On the first run ever, `torch.compile` writes its compiled
kernels there. Every subsequent cold start finds them already present and skips
recompilation.

Expected cold-start latency: 220s → ~11s (CUDA graph capture only, unavoidable).

### 3. `qwen_sdpo/modal_trainer.py`

Three changes:

- **Import**: `from qwen_sdpo._weight_sync import QwenSDPOWorker, sync_lora_weights_to_vllm`
- **`_setup_trainer()`**: Added `worker_cls=QwenSDPOWorker` to the `LLM()` constructor.
  Also imported `compile_cache_volume` from `_modal_infra` and mounted it at
  `/root/.cache/vllm/torch_compile_cache` in the `@app.cls` volumes dict.
- **`run_sdpo_step()`**: Added `sync_lora_weights_to_vllm(self.model, self.vllm_engine, _QWEN35_LORA_TARGET_MODULES)` immediately after `self._optimizer.step()`.

After the fix:

```
optimizer.step()                      → HF model updated (LoRA A, B change)
sync_lora_weights_to_vllm(...)        → merged bf16 weights pushed into vLLM in-place
generate_only() [next iteration]      → vLLM generates with updated policy  ← fixed
```

### 4. `sdpo_modal_local_verify_kimina/modal_app.py`

Same frozen-generator bug existed here. Additional issues:

- Used a `is_large_model` heuristic to decide whether to add LoRA (only for 7B+ models).
  For the Qwen3.5-4B use case this would skip LoRA entirely. Replaced with unconditional
  QLoRA always applied.
- LoRA target modules were missing all `Qwen3_5GatedDeltaNet` modules (`in_proj_*`,
  `out_proj`). Updated to the full `_QWEN35_LORA_TARGET_MODULES` list.
- `transformers>=4.40.0` — updated to `>=5.2.0` (required for `qwen3_5` model type).
- Added `worker_cls=QwenSDPOWorker` to `LLM()`.
- Added `compile_cache_volume` and its volume mount.
- Stored `trainer_self.lora_target_modules` so `modal_trainer.py` can pass it to
  `sync_lora_weights_to_vllm` without hard-coding the list a second time.
- Added `enforce_eager=False` (was missing — CUDA graphs were off by default).

### 5. `sdpo_modal_local_verify_kimina/modal_trainer.py`

- Imported `sync_lora_weights_to_vllm` from `qwen_sdpo._weight_sync`.
- Imported `compile_cache_volume` from `modal_app` and added it to the volumes dict.
- Added `sync_lora_weights_to_vllm(self.model, self.vllm_engine, self.lora_target_modules)`
  after `self._optimizer.step()` in `run_sdpo_step()`.

### 6. `qwen_sdpo/tests/test_weight_sync.py` (new file)

CPU-only unit tests, no GPU/Modal/vLLM installation required. Three groups:

**Group 1 — LoRA delta math:**
- `test_lora_delta_shape`: `B @ A` produces shape matching base weight.
- `test_lora_delta_values`: merged weight equals `base + B @ A * scale` within tolerance.
- `test_merged_weight_differs_from_base`: merge actually changes the tensor.

**Group 2 — IPC handle plumbing (vLLM mocked via `sys.modules`):**
- `test_collective_rpc_called_once_per_sync`: `collective_rpc` is called exactly once
  with `"update_weights_from_ipc_handles"` as the method name.
- `test_unmerge_called_even_if_push_fails`: `unmerge_adapter()` is always called (the
  `finally` block) even when `collective_rpc` raises.

**Group 3 — Worker-side in-place update:**
- `test_worker_load_weights_updates_target`: correct tensor overwritten with new values.
- `test_worker_load_weights_is_inplace`: tensor object identity preserved (same address)
  — critical for CUDA graph validity.
- `test_worker_unknown_param_ignored`: unknown param names do not crash or corrupt others.

All 8 tests pass locally (`python3 qwen_sdpo/tests/test_weight_sync.py`).

---

## What is NOT tested locally (requires real H100 on Modal)

- That CUDA graphs actually remain valid after the in-place weight overwrite (address
  preservation is verified by `test_worker_load_weights_is_inplace`, but graph validity
  requires a real CUDA graph capture and replay).
- That vLLM's collective_rpc actually dispatches to the worker process (the IPC
  plumbing test mocks the engine).
- That generation quality measurably improves across SDPO iterations (requires a full
  run on a problem where the base model fails iteration 1 but the updated model
  succeeds on a later iteration).

---

## Per-Step Overhead Estimate

| Operation | Time |
|---|---|
| `merge_adapter()` | ~100-200ms (dequantize + matmul on GPU) |
| `reduce_tensor()` × N layers | ~1ms total |
| `collective_rpc` + `load_weights()` | ~50-100ms (GPU memcpy) |
| `unmerge_adapter()` | ~100-200ms |
| **Total sync overhead** | **~300-500ms per step** |

Generation time per step is ~30s. Sync adds <2% overhead.
