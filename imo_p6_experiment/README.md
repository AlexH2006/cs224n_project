# SDPO Local Verify — Qwen3.5-4B

Test-time SDPO for Lean 4 with **online RL**: generate and train on Modal, verify locally (or via Kimina). After each gradient step, merged LoRA weights are pushed into the vLLM engine in-place so the next generation uses the updated policy.

## Overview

| Component | Role |
|-----------|------|
| **Modal** | vLLM generation + HuggingFace QLoRA training |
| **Local** | Parse model output, verify proofs (local `lake exe repl` or Kimina HTTP) |
| **Weight sync** | After each SDPO step: merge LoRA into base, push to vLLM via CUDA IPC |

## Architecture

```
generate_only.remote()  →  parse + verify (local)  →  run_sdpo_step.remote()
        ↑                                                        │
        │                                                        ▼
        └──────────────────  sync_lora_weights_to_vllm_kimina()  ─┘
```

- **Model**: `Qwen/Qwen3.5-4B` (`Qwen3_5ForConditionalGeneration` — natively multimodal, used text-only). vLLM loads the full ConditionalGeneration model; parameters live under `language_model.model.layers.*`.
- **Hybrid attention** (32 layers, 3:1 ratio): 24 Gated DeltaNet (linear attention) + 8 standard attention.
- **Teacher prompt** uses the **current** iteration's compiler feedback (never stale). The teacher is the feedback-conditioned policy: same problem, but with the current error appended.
- **QLoRA** (4-bit NF4 + LoRA) targets all trainable projections:
  - Standard attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - GDN linear attention: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`
  - MLP: `gate_proj`, `up_proj`, `down_proj`
- **KiminaSDPOWorkerExtension**: vLLM worker extension enabling in-place `load_weights()` via `collective_rpc`. CUDA graphs remain valid — no recapture after sync.
- **VLLM_ENABLE_V1_MULTIPROCESSING=0**: Required so `collective_rpc` is in-process (ZMQ+msgpack cannot serialize the weight dict).
- **compile_cache_volume**: Persists torch.compile artifacts; reduces cold-start first-inference from ~220s to ~11s.

## Requirements

- `transformers>=5.2.0` (Qwen3_5ForConditionalGeneration support)
- Modal (`pip install modal`, `modal token new`)
- For local verification: elan + mathlib4 (see `devlog/20260303_local_lean_verifier_setup.md`)
- For Kimina verification: Kimina Docker or `LEAN_VERIFY_BACKEND=kimina`, `LEAN_VERIFY_KIMINA_URL`

## Run

```bash
modal run training/lean_sdpo_local_verify_modal.py --problem-idx 0
modal run training/lean_sdpo_local_verify_modal.py --problem-idx 0 --max-iterations 5 --gpu A100-80GB
```

Results: `sdpo_results/local_verify/Qwen3.5-4B/minif2f-lean4/run_{idx}_{timestamp}/`

## Key files

| File | Purpose |
|------|---------|
| `modal_app.py` | App, inference image, volumes, `_setup_trainer` (vLLM + HF QLoRA via AutoModelForImageTextToText) |
| `modal_trainer.py` | `SDPOTrainer`: `generate_only`, `run_sdpo_step`, `finalize_run` |
| `entrypoint.py` | `run_main`: local loop, prompt building, verification orchestration |
| `_weight_sync_kimina.py` | `KiminaSDPOWorkerExtension`, `sync_lora_weights_to_vllm_kimina` (unfused → vLLM stacked routing) |
| `sdpo_loss.py` | SDPO loss (KL, reward, entropy) |
| `config.py` | `SDPOConfig` (defaults to `Qwen/Qwen3.5-4B`) |

## Volumes

| Volume | Path | Purpose |
|--------|------|---------|
| `sdpo-hf-cache` | `/cache` | Model weights |
| `sdpo-output-local-verify` | `/output` | Training results |
| `vllm-compile-cache` | `/root/.cache/vllm/torch_compile_cache` | torch.compile kernel cache |
