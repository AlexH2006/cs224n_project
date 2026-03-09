# SDPO Pipeline: Debugging Log and Architectural Redesign

**Date:** 2026-03-07  
**Affects:** `qwen_sdpo/`, `qwen_sdpo/_weight_sync.py`, `qwen_sdpo/entrypoint.py`

---

## Overview

This log documents all bugs encountered during the first end-to-end run of the SDPO pipeline, the root causes, the resolution strategy for each, and two deeper architectural issues that need to be fixed before the pipeline can be considered correct.

---

## Part 1: Weight Sync Bugs Encountered During Run

### Bug 1 — `worker_cls` must be a string, not a class object

**Error:** `ValidationError for ParallelConfig: worker_cls — Input should be a valid string`

**Root cause:**  
In vLLM V1, `ParallelConfig` is a Pydantic dataclass. The `worker_cls` field is typed as `str`, not `type`. We passed the Python class object (`QwenSDPOWorker`) directly. Pydantic rejected it at validation time before the engine even started.

**Resolution:** Pass the fully-qualified module path as a string: `"qwen_sdpo._weight_sync.QwenSDPOWorker"`.

---

### Bug 2 — `worker_cls` is the wrong API for vLLM V1

**Error:** `TypeError: QwenSDPOWorker() takes no arguments` / `AttributeError: 'QwenSDPOWorker' object has no attribute 'init_device'`

**Root cause:**  
vLLM V1 completely redesigned its worker architecture. The `worker_cls=` parameter still exists in the config but now controls `UniProcExecutor`'s *internal driver wrapper*, not the model-running worker. `worker_class(**kwargs)` is called with construction kwargs that the V1 `Worker` does not accept in `__init__` (it is initialized through a separate `init_worker()` call). Our subclass broke both the init and the interface contract.

The correct V1 API is `worker_extension_cls=`: a **plain Python class** (not a Worker subclass) whose methods vLLM mixes into its internal worker. This is documented in `vllm/examples/offline_inference/rlhf.py` and in the V1 PR notes.

**Resolution:** Replace `QwenSDPOWorker(Worker)` with `QwenSDPOWorkerExtension` (a plain class, no inheritance). Register via `worker_extension_cls="qwen_sdpo._weight_sync.QwenSDPOWorkerExtension"`.

---

### Bug 3 — `collective_rpc` serializes via msgpack, not pickle

**Error:** `TypeError: Object of type <class 'function'> is not serializableSet VLLM_ALLOW_INSECURE_SERIALIZATION=1`

**Root cause:**  
The CUDA IPC approach uses `torch.multiprocessing.reductions.reduce_tensor()`, which returns a `(function, args)` tuple — a Python callable, not serializable by msgpack. vLLM V1's `SyncMPClient` sends all `collective_rpc` arguments through a msgpack encoder that only knows about primitives, numpy arrays, and a small set of vLLM types. It explicitly blocks pickle to prevent code injection.

The CUDA IPC approach is fundamentally incompatible with vLLM V1's `collective_rpc` transport layer in this configuration.

**Resolution:** Do not use CUDA IPC handles. Instead, move the merged weights to CPU as numpy arrays and transmit through `collective_rpc`. Numpy arrays are natively msgpack-serializable in vLLM's encoder.

---

### Bug 4 — PEFT LoRA detection by isinstance failed

**Error:** `No LoRA-wrapped layers found for target modules`

**Root cause:**  
We initially checked `isinstance(module, peft.tuners.lora.layer.Linear)` to identify LoRA-wrapped layers. In newer PEFT versions (5.x), when LoRA is applied to a 4-bit `bitsandbytes.nn.Linear4bit` layer, the resulting wrapper class is `peft.tuners.lora.bnb.Linear4bit` — a different class entirely, not a subclass of `lora.layer.Linear`. The isinstance check silently missed every layer.

**Resolution:** Detect PEFT LoRA wrappers by duck-typing: check for the presence of `base_layer`, `lora_A`, and `lora_B` attributes. This is robust across all PEFT versions and quantization backends.

---

### Bug 5 — `reduce_tensor` blocked on `requires_grad=True`

**Error:** `RuntimeError: Cowardly refusing to serialize non-leaf tensor which requires_grad`

**Root cause:**  
The computed delta `lora_B @ lora_A` carries `requires_grad=True` because `lora_B.weight` and `lora_A.weight` are training parameters with gradient tracking enabled. `reduce_tensor` and `torch.multiprocessing` refuse to serialize gradient-carrying tensors across process boundaries because autograd cannot operate across process memory.

**Resolution:** Call `.detach()` on `lora_A.weight` and `lora_B.weight` before the matmul. The training graph for A/B is retained — we only detach the copy used for the weight merge.

---

### Bug 6 — msgpack inhomogeneous shape error for list-of-tuples

**Error:** `setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (3,) + inhomogeneous part.`

**Root cause:**  
When passing `named_weights` as a `list[tuple[str, ndarray]]` through `collective_rpc`, vLLM's msgpack encoder tried to encode the list as a numpy array (since it saw numpy arrays as elements). The encoder treated each `(str, ndarray)` tuple as an array row, producing an inhomogeneous shape error because string and ndarray have different sizes.

**Resolution:** Change the payload format from a list of tuples to a plain dict `{param_name: ndarray}`. Dicts with string keys and numpy array values are handled correctly by the msgpack encoder. Pass as `kwargs={"weight_dict": weight_dict}` rather than `args=(...)`.

---

### Bug 7 — `teacher_prompt=None` on iteration 1 crashed the loss function

**Error:** `ValueError: You need to specify either text or text_target` (from `tokenizer(None, ...)`)

**Root cause:**  
SDPO is a self-distillation method: the teacher prompt is the student prompt augmented with error feedback from the verifier. On iteration 1, there is no prior feedback yet, so `teacher_prompt` is `None`. The `entrypoint.py` still called `run_sdpo_step.remote()` with `teacher_prompt=None`, and `compute_sdpo_loss()` unconditionally called `tokenizer(teacher_prompt, ...)`, crashing on `None`.

**Resolution:** In `run_sdpo_step`, check `if not payload.get("teacher_prompt")` and return early with `skipped_no_teacher=True`. Additionally, fix the `entrypoint.py` display code to use `or 0.0` guards instead of `.get('loss', 0)` (which returns the `None` value when the key exists).

---

### Bug 8 — `merge_adapter()` silently failed with gradient checkpointing

**Error:** `No bf16 parameters matched LoRA target modules after merge`  
(Params still showed `base_layer.weight: torch.uint8` and separate `lora_A`/`lora_B` after calling `merge_adapter()`)

**Root cause:**  
When gradient checkpointing is enabled (via `gradient_checkpointing_enable()`), `merge_adapter()` may fail silently or produce unexpected state. The parameters still showed the unmerged structure because the merge did not complete, but no exception was raised. Even when merge succeeds, PEFT's parameter naming after merge may differ from what we expected to filter by suffix.

**Resolution:** Bypass `merge_adapter()` entirely. Compute the merged weight directly: dequantize the 4-bit base (`bnb_F.dequantize_4bit(...)`) and add the LoRA delta (`lora_B @ lora_A * scale`). This is numerically identical to what `merge_adapter()` does internally but is explicit, gradient-checkpointing-safe, and does not alter the model state.

---

## Part 2: ~~Unresolved Issue~~ RESOLVED — collective_rpc Serialization

**Status: Fixed 2026-03-08. See `20260308_weight_sync_final_fix.md` for full details.**

**Root cause (confirmed):** `VLLM_ENABLE_V1_MULTIPROCESSING=1` (default) routes all
`collective_rpc` calls through a ZMQ+msgpack subprocess. Large numpy weight arrays could
not be reliably serialized through this path.

**Fix:** Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` in the Modal container `env=` on
`@app.cls()`. This makes `collective_rpc` a direct in-process Python call — no ZMQ, no
msgpack, no serialization at all. On a single H100 (TP=1) the multiprocessing provides
zero benefit, so this is a free win.

---

## Part 3: Generation Latency Issue

**Observation:** The first generation per Modal container takes ~4-5 minutes for 8192 tokens (≈33 tok/s), while subsequent generations within the same container take ~3 minutes (≈45 tok/s).

**Root cause analysis:**

The slowdown on the first generation has two separable causes:

1. **Chunked prefill cold path.** The Qwen3.5-4B student prompt is long (includes the full theorem statement, informal hint, and system instruction). The first prefill processes the full prompt length through the chunked prefill scheduler. vLLM V1's asynchronous scheduler adds ~5-10s of overhead on the first request as internal queues and state machines warm up.

2. **Token budget is massively over-allocated for this task.** `max_new_tokens=8192` was chosen for the baseline eval (where we want to give the model every chance to produce a proof). For SDPO online training, **8192 tokens means 4-5 minutes of generation time per iteration, per step**. For a 5-iteration run that is potentially 25+ minutes of generation alone.

   The first generation is especially slow because the model (on iteration 1, before any LoRA training) tends to produce extremely long reasoning chains that fill the entire context window. These always `finish_reason=length` — meaning they are truncated before reaching a lean4 block, producing a `TRUNCATED` parse result that cannot be verified and cannot produce useful feedback. The model wastes the entire token budget and produces zero training signal.

**Recommended fix:** Reduce `max_new_tokens` significantly for SDPO (e.g., 2048-4096). The model should be incentivized to write concise proofs. If the proof requires more than 4096 tokens of reasoning, truncation feedback ("write shorter chain-of-thought") is the right signal anyway. This alone would reduce per-iteration wall time from ~4 minutes to ~1 minute.

---

## Part 4: Fundamental Logic Error — Self-Distillation Timing

**This is the most important bug. The current pipeline is algorithmically wrong.**

### What the pipeline currently does

```
Iteration N:
  1. Generate with student prompt (no feedback)   → proof attempt
  2. Verify proof attempt                         → error feedback F_N
  3. Build teacher prompt using F_{N-1}           ← uses PREVIOUS iteration's error
  4. Compute SDPO loss: KL(teacher_{N-1} || student)
  5. Gradient step
```

The teacher prompt given to `run_sdpo_step` is built from `latest_feedback`, which stores the error from iteration N-1. On iteration N, the model trains on the feedback from the previous iteration's failure — not the current one. On iteration 1, `latest_feedback = None`, so no gradient step occurs at all.

### What the pipeline should do

The core idea of SDPO / self-distillation is: **given what just went wrong, teach the model what to do instead — right now, using that signal.**

```
Iteration N:
  1. Generate with student prompt (no feedback)   → proof attempt
  2. Verify proof attempt                         → error feedback F_N
  3. Build teacher prompt using F_N               ← uses THIS iteration's error
  4. Compute SDPO loss: KL(teacher_N || student)
  5. Gradient step  →  model immediately learns from its own mistake
```

The teacher prompt for iteration N's gradient step should use F_N (the error we just received), not F_{N-1}. The current one-iteration lag means:

- Iteration 1: generates, fails with error E1, trains on nothing (no prior feedback)
- Iteration 2: generates, fails with error E2, trains on E1's feedback
- Iteration 3: generates, fails with error E3, trains on E2's feedback
- ...

This is a delayed feedback loop. The model trains on a signal that is already stale by the time the gradient step is applied. More importantly, iteration 1 — when the model is most wrong and the feedback is most informative — produces zero gradient signal.

### The correct loop structure

```python
for iteration in range(max_iterations):
    raw_output, generated_ids, finish_reason = generate()
    verification = verify(parse(raw_output))

    if is_success:
        break

    if not is_server_error and not parse_failed:
        # Build teacher prompt from THIS iteration's error
        feedback = verification["feedback"]
        teacher_prompt = build_teacher_prompt(..., feedback, ...)
        run_sdpo_step(student_prompt, teacher_prompt, generated_ids)
        # Weight sync → next generate() uses updated policy
```

The teacher prompt must be constructed from the current iteration's error and used in the same iteration's gradient step. There is no need for a `latest_feedback` lag variable.

### Why the current design had the lag

The comment in `entrypoint.py` (line 262) says:
> "Build teacher prompt for this step (uses the *previous* iteration's error)."

This was likely a misunderstanding of the algorithm. In some RL-from-feedback setups, you generate with a feedback-augmented prompt and compare to a baseline. But in SDPO self-distillation, the teacher IS the feedback-conditioned policy — and the conditioning must come from the current failure.

### Impact

- Iteration 1 always skips training (guaranteed wasted iteration)
- The model trains on stale signals
- The effective number of useful gradient steps is `max_iterations - 1`, not `max_iterations`
- Even within those `max_iterations - 1` steps, the feedback is one step behind, reducing gradient alignment

---

## Summary of Changes Needed

| Issue | Fix |
|---|---|
| `worker_cls` type error | Already fixed: use string path |
| vLLM V1 worker API mismatch | Already fixed: use `worker_extension_cls` |
| CUDA IPC not msgpack-safe | Switched to numpy dict; but see serialization ceiling concern |
| PEFT isinstance false negative | Already fixed: duck-typing on `base_layer`/`lora_A`/`lora_B` |
| `requires_grad` on delta tensor | Already fixed: `.detach()` before matmul |
| msgpack inhomogeneous list | Already fixed: dict + kwargs |
| `teacher_prompt=None` crash | Already fixed: skip guard in `run_sdpo_step` |
| `merge_adapter()` silent failure | Already fixed: direct dequant + delta computation |
| Generation too slow | ✓ Fixed: reduced `max_new_tokens` to 4096 |
| **Self-distillation timing wrong** | ✓ Fixed: use current iteration's feedback (see this devlog) |
| Weight sync serialization failure | ✓ Fixed: `VLLM_ENABLE_V1_MULTIPROCESSING=0` (see `20260308_weight_sync_final_fix.md`) |
| Weight sync param name mismatch | ✓ Fixed: prefix `language_model.` (see `20260308_weight_sync_final_fix.md`) |

---

## Part 5: Teacher Prompt Construction — Implementation Audit (2026-03-08)

### Where it is built

`qwen_sdpo/prompts.py` → `build_teacher_prompt()`, called from `qwen_sdpo/entrypoint.py` line 270.

### The four components

The teacher prompt is a chat-formatted string with the following structure, in order:

**1. System turn** — hardcoded in `qwen_eval/prompts.py::_SYSTEM_PROMPT`:
```
You are an expert in mathematics and Lean 4 theorem proving.
```
Identical to the student prompt's system turn.

**2. Base user message** — assembled from `qwen_eval/prompts.py::_USER_TEMPLATE` with three sub-components:
- `informal`: natural language problem statement from the dataset
- `header + theorem_code`: dataset-provided Lean imports (or `cfg.default_header` if absent) concatenated with the formal Lean 4 theorem containing `sorry`
- Fixed instructions: "Do NOT use sorry", "output a complete self-contained lean4 code block as the very last thing"

**3. Compiler feedback block** — appended to the user turn immediately after the instructions. Sourced from `verification["feedback"]` of **the current iteration's** verifier result — never from a prior step. Format depends on `cfg.feedback_errors_only`:

- **Errors-only** (`feedback_errors_only=True`, the default):
  ```
  The following proof attempt was INCORRECT. The Lean 4 compiler returned this error:
  {error}
  Avoid this mistake in your new attempt.
  ```
- **Errors + failed proof** (`feedback_errors_only=False`):
  ```
  The following proof attempt was INCORRECT:
  ```lean4
  {failed_proof}
  ```
  The Lean 4 compiler returned this error:
  {error}
  Avoid this mistake in your new attempt.
  ```

**4. Generation prompt** — `tokenizer.apply_chat_template(..., add_generation_prompt=True)` appends `<|im_start|>assistant\n<think>\n` to open the model's reasoning turn.

### What `error` actually contains

The `error` / `feedback` string is set by `entrypoint.py` at line 269:
```python
feedback = verification.get("feedback") or "Proof verification failed."
```

Three cases:
- **Truncated output** (no `</think>` found): synthetic feedback — "Your reasoning was cut off because it exceeded the token limit. Write a shorter chain-of-thought..."
- **No lean4 block found**: synthetic feedback — "No lean4 code block was found in the output..."
- **Real Lean compiler error**: the raw compiler error string from the Kimina verifier

### Concrete example (from `sdpo_results/Qwen3.5-4B/run_Qwen3.5-4B_13_20260307_164318/logs.json`)

**Problem:** amc12b_2021_p3 — iteration 1, output truncated (no closing `</think>` tag), `feedback_errors_only=True`.

**1. System turn**

    <|im_start|>system
    You are an expert in mathematics and Lean 4 theorem proving.<|im_end|>

**2. Base user message** (from `_USER_TEMPLATE` with `informal` + `header_and_theorem`)

    <|im_start|>user
    Think step-by-step to prove the following Lean 4 theorem.

    # Informal problem:
    Suppose $2+\frac{1}{1+\frac{1}{2+...}}=\frac{144}{53}$. What is the value of $x$?
    \textbf{(A) }\frac34 \qquad \textbf{(B) }\frac78 \qquad ... Show that it is \text{A}.

    # Lean 4 theorem to prove:
    ```lean4
    import Mathlib.Algebra.BigOperators.Basic
    ... (header + theorem amc12b_2021_p3 ... := sorry)
    ```

    Instructions:
    - Do NOT use `sorry`
    - At the very end of your response, output your final answer as exactly one lean4 code block...
    - Do not output any text after the closing ```

**3. Compiler feedback block** (appended; `feedback` = current iteration's truncated feedback)

    The following proof attempt was INCORRECT. The Lean 4 compiler returned this error:
    Your reasoning was cut off because it exceeded the token limit. Your response was too long and the final lean4 code block was never reached. Write a shorter chain-of-thought, then output the complete proof immediately in a ```lean4 block.
    Avoid this mistake in your new attempt.<|im_end|>

**4. Generation prompt**

    <|im_start|>assistant
    <think>

The only difference from the **student prompt** is the insertion of the feedback block between the instructions and `<|im_end|>`. The student prompt ends after "Do not output any text after the closing ```"; the teacher prompt adds the error block before closing the user turn.

### Correctness status

The implementation is algorithmically correct as of the fix applied on 2026-03-07:
- Feedback is always sourced from the **current** iteration's verifier output (not `latest_feedback` from the prior iteration — that bug was fixed)
- On every iteration, the teacher prompt is constructed and passed to `run_sdpo_step` regardless of whether the proof succeeded or failed
- The `build_teacher_prompt` in `qwen_sdpo` and `create_feedback_prompt` in `sdpo_modal_local_verify_kimina` follow the same pattern and both use current-iteration feedback
