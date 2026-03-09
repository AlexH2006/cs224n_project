# Weight Sync: Final Fix — In-Process collective_rpc + Param Name Mapping

**Date:** 2026-03-08  
**Affects:** `qwen_sdpo/_weight_sync.py`, `qwen_sdpo/modal_trainer.py`  
**Status:** Fixed and verified — full 5-iteration SDPO run completed successfully.

---

## Context

This log continues from `20260307_sdpo_debugging_and_redesign.md`, which left two open
issues:

1. The `collective_rpc` weight transfer (numpy dict through msgpack) was described as a
   "serialization ceiling" concern — potentially hitting encoding failures for large
   payloads.
2. The vLLM parameter name mapping (`model.layers...` vs. what vLLM actually uses) had
   not been verified against the running model.

Both issues manifested as runtime crashes on the first end-to-end run with weight sync
enabled. This log documents the root causes and fixes.

---

## Bug 1 — collective_rpc serialization failure via ZMQ/msgpack

### Symptom

After the weight sync path was enabled, `collective_rpc("update_weights_from_numpy", ...)`
failed with a serialization error. The previous devlog attributed this to msgpack's
inhomogeneous-shape handling, which was partially addressed by switching from
`args=(list_of_tuples,)` to `kwargs={"weight_dict": dict}`. However, the fundamental
problem was not in the encoding of the dict values themselves.

### Root cause (confirmed via vLLM V1 source)

By default, vLLM V1 sets `VLLM_ENABLE_V1_MULTIPROCESSING=1`. This means `LLM()` spins
up a separate engine-core subprocess and communicates with it over a ZMQ socket using
msgpack. The full data path for `collective_rpc` is:

```
LLMEngine.collective_rpc()
  → EngineCoreClient (= SyncMPClient)
  → SyncMPClient.collective_rpc()
      → self.call_utility("collective_rpc", method, timeout, args, kwargs)
          → self._send_input(UTILITY, (0, call_id, "collective_rpc", (method, timeout, args, kwargs)))
              → MsgpackEncoder.encode(...)  ← serialization happens here
                  → ZMQ socket → EngineCore subprocess
                      → UniProcExecutor.collective_rpc()
                          → run_method(worker, method, args, kwargs)
                              → QwenSDPOWorkerExtension.update_weights_from_numpy(weight_dict=...)
```

The entire `(method_name, timeout, args_tuple, kwargs_dict)` payload is msgpack-encoded.
On a single H100 with tensor parallelism = 1, the subprocess provides no benefit — it
exists solely for scheduling decoupling in multi-GPU deployments.

Key sources confirmed:
- `vllm/v1/executor/uniproc_executor.py`: `UniProcExecutor.collective_rpc()` calls
  `run_method(self.driver_worker, method, args, kwargs)` directly — no serialization
  when in-process.
- `vllm/v1/engine/core_client.py`: `EngineCoreClient.make_client()` returns
  `SyncMPClient` when `VLLM_ENABLE_V1_MULTIPROCESSING=1` (the default), which adds the
  ZMQ+msgpack layer.
- `vllm/envs.py`: confirms `VLLM_ENABLE_V1_MULTIPROCESSING` defaults to `"1"`.

### Fix

Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` in the Modal container environment. This makes
`EngineCoreClient.make_client()` return an `InprocClient` (direct Python call, no
subprocess, no ZMQ, no msgpack). The `collective_rpc` call becomes:

```
LLMEngine.collective_rpc()
  → InprocClient → InprocEngineCore
      → UniProcExecutor.collective_rpc()
          → run_method(driver_worker, method, args, kwargs)  ← pure Python call
              → QwenSDPOWorkerExtension.update_weights_from_numpy(weight_dict=...)
```

The numpy arrays are passed by reference with zero serialization overhead. CUDA graphs
are unaffected because we never cross a process boundary.

**Change in `qwen_sdpo/modal_trainer.py`:**

```python
@app.cls(
    ...
    # Disable vLLM V1 multiprocessing so collective_rpc uses direct in-process
    # calls instead of ZMQ + msgpack. With TP=1 on a single H100 there is no
    # benefit to multiprocessing, and keeping it on forces all collective_rpc
    # kwargs (including large numpy weight arrays) through the msgpack serializer,
    # which does not support arbitrary dict[str, ndarray] payloads without
    # VLLM_ALLOW_INSECURE_SERIALIZATION. In-process mode: direct Python call,
    # zero serialization overhead, zero risk of encoding failures.
    env={"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
)
```

---

## Bug 2 — vLLM parameter name mismatch: `model.*` vs `language_model.model.*`

### Symptom

After the serialization fix, `collective_rpc` reached the worker successfully, but
`load_weights()` raised:

```
ValueError: There is no module or parameter named 'model' in
Qwen3_5ForConditionalGeneration. The available parameters belonging to
(Qwen3_5ForConditionalGeneration) are: {'language_model.model.layers.0...', ...}
```

### Root cause

The param name translation in `sync_lora_weights_to_vllm()` stripped the PEFT prefix
but produced the wrong root namespace:

| Stage | Name |
|---|---|
| HF PEFT (named_modules) | `base_model.model.model.layers.0.self_attn.q_proj` |
| After `.removeprefix("base_model.model.")` | `model.layers.0.self_attn.q_proj` |
| We appended `.weight` to get | `model.layers.0.self_attn.q_proj.weight` |
| **vLLM actually expects** | `language_model.model.layers.0.self_attn.q_proj.weight` |

The discrepancy exists because `LLM()` with `language_model_only=True` still loads the
full `Qwen3_5ForConditionalGeneration` wrapper class. Even though vLLM skips the vision
encoder tower, the top-level module namespace is `language_model` (wrapping the bare
language model). The HF model is just the language model directly, so its namespace is
`model`.

### Fix

Replace the bare `model.` root with `language_model.` when constructing the vLLM param
name:

```python
# Before (wrong):
vllm_param_name = module_name.removeprefix("base_model.model.") + ".weight"
# → "model.layers.0.self_attn.q_proj.weight"  ← vLLM rejects this

# After (correct):
inner_name = module_name.removeprefix("base_model.model.")
vllm_param_name = "language_model." + inner_name + ".weight"
# → "language_model.model.layers.0.self_attn.q_proj.weight"  ← vLLM accepts this
```

**Change in `qwen_sdpo/_weight_sync.py`** (the param name construction block):

```python
# Strip PEFT's "base_model.model." prefix and replace the bare "model." root
# with "language_model." to match vLLM's Qwen3_5ForConditionalGeneration naming.
#
# HF PEFT name:  "base_model.model.model.layers.0.self_attn.q_proj"
# After strip:   "model.layers.0.self_attn.q_proj"
# vLLM expects:  "language_model.model.layers.0.self_attn.q_proj.weight"
#
# This is because vLLM loads the full Qwen3_5ForConditionalGeneration wrapper
# (even with language_model_only=True) whose top-level module is "language_model".
inner_name = module_name.removeprefix("base_model.model.")
vllm_param_name = "language_model." + inner_name + ".weight"
```

---

## Verification

A full 5-iteration run on Qwen3.5-4B (problem idx 13) completed with exit code 0 in
~10 minutes total. The vLLM log confirmed the extension was injected:

```
INFO Injected <class 'qwen_sdpo._weight_sync.QwenSDPOWorkerExtension'> into
<class 'vllm.v1.worker.gpu_worker.Worker'> for extended collective_rpc calls
['check_weights_updated', 'update_weights_from_numpy']
```

Weight sync ran after every gradient step without errors. Iteration results:

| Iter | Tokens | Finish | Loss | Reward | Grad norm |
|------|--------|--------|------|--------|-----------|
| 1 | 4096 | length | 0.0116 | 10.63 | 1.325 |
| 2 | 2952 | stop | 0.0118 | 10.85 | 0.418 |
| 3 | 4096 | length | 0.0033 | 22.95 | 0.196 |
| 4 | 3822 | stop | 0.0126 | -6.88 | 0.759 |
| 5 | 4096 | length | 0.0029 | 22.99 | 0.170 |

Generation speed on iterations 2+ was ~216 tok/s (close to baseline), confirming CUDA
graphs stayed valid across weight syncs (no recapture).

---

## Summary of all weight sync bugs (cumulative across both devlogs)

| # | Error | Root cause | Fix |
|---|-------|------------|-----|
| 1 | `worker_cls` type ValidationError | Pydantic field requires string, not class object | Pass fully-qualified string: `"qwen_sdpo._weight_sync.QwenSDPOWorker"` |
| 2 | `QwenSDPOWorker() takes no arguments` / `no attr init_device` | `worker_cls` in vLLM V1 controls an internal driver wrapper, not the model worker | Use `worker_extension_cls` with a plain class (no `Worker` inheritance) |
| 3 | CUDA IPC `function` not serializable | `reduce_tensor()` returns a callable tuple; msgpack blocks non-primitive types | Switch to CPU numpy arrays |
| 4 | No LoRA layers found | `isinstance(m, peft.tuners.lora.layer.Linear)` misses `lora.bnb.Linear4bit` | Duck-type: check for `base_layer` + `lora_A` + `lora_B` attributes |
| 5 | `requires_grad=True` on delta tensor | `lora_B @ lora_A` inherits gradient tracking from training params | Call `.detach()` on A and B weights before matmul |
| 6 | msgpack inhomogeneous shape | `list[tuple[str, ndarray]]` treated as a single ndarray with mixed-type rows | Use `dict[str, ndarray]` passed as `kwargs={"weight_dict": ...}` |
| 7 | `collective_rpc` serialization failure (ZMQ path) | `VLLM_ENABLE_V1_MULTIPROCESSING=1` default routes through ZMQ+msgpack subprocess | Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` → in-process direct call, no serialization |
| 8 | `ValueError: no module named 'model'` in `load_weights` | HF param root is `model.`, vLLM wraps it under `language_model.model.` | Prefix with `language_model.` after stripping PEFT's `base_model.model.` |
| 9 | `merge_adapter()` silent failure | Incompatible with gradient checkpointing; may leave model in inconsistent state | Bypass: compute `dequant(W_4bit) + lora_B @ lora_A * scale` directly |

---

## Open items

- The `language_model.` prefix mapping is Qwen3.5-specific. If adapting this pipeline
  to a different architecture, verify the vLLM top-level module name by inspecting
  `list(model_runner.model.named_parameters())` before building the weight dict.
- Generation is still truncated on many iterations (`finish=length`) because
  `max_new_tokens=4096` is often not enough for the model's current reasoning style. A
  future improvement is to add a short-circuit: if the model fills the context budget
  on 3 consecutive iterations, halt early and record as a failure rather than continuing
  to generate truncated proofs.
- The KL heatmap renderer raises `ParseException` on LaTeX math in theorem names (the
  `$$` in problem statements). This is cosmetic and does not affect training correctness.
