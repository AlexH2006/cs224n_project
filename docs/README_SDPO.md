# Test-Time Self-Distillation for Lean Theorem Proving

**TL;DR** — Test-time and full SDPO (Self-Distilled Policy Optimization) for Lean 4: generate proof attempts, verify with a Lean backend, use feedback for self-distillation and (on Modal) gradient updates.

## Overview

The pipeline works as follows:

```
┌─────────────────────────────────────────────────────────────┐
│                    Test-Time Self-Distillation              │
├─────────────────────────────────────────────────────────────┤
│  1. Generate N proof attempts (vLLM / transformers)         │
│                     ↓                                       │
│  2. Verify each proof (Kimina HTTP or local lake exe repl)   │
│                     ↓                                       │
│  3. If success → Return proof                               │
│     If failure → Extract compiler errors as feedback         │
│                     ↓                                       │
│  4. Reprompt with feedback (self-distillation)              │
│     - Include error messages from failed attempts            │
│     - Include successful proofs as demonstrations            │
│                     ↓                                       │
│  5. Compute KL divergence for policy regularization         │
│                     ↓                                       │
│  6. Repeat until success or max iterations                  │
└─────────────────────────────────────────────────────────────┘
```

### Verification backends

| Backend | Use case | Package / script |
|--------|----------|-------------------|
| **Kimina** (HTTP) | Cloud or local Docker Lean server | `sdpo_modal` (Modal), `verification/verify_proofs_kimina.py` |
| **Local Lean** (`lake exe repl`) | No Kimina; verify in mathlib4 workspace | `sdpo_modal_local_verify`, `sdpo_modal.local_lean_verifier` |

Modal pipelines that use **Kimina** live in `sdpo_modal/` and are invoked by `training/lean_sdpo_*_modal.py` (e.g. Kimina 2B, Distill 1.7B, Goedel 8B). The **local verification** pipeline lives in `sdpo_modal_local_verify/` and is invoked by `training/lean_sdpo_local_verify_modal.py` (generate/train on Modal; verification runs on your machine via `lake exe repl`).

## Requirements

**Local test-time only (no Modal):**
```bash
pip install torch transformers kimina-client
```

**Kimina server** (for Kimina-backed pipelines):
```bash
# Using Docker
docker run -d -p 80:80 projectnumina/kimina-lean-server:2.0.0

# Or from source (see setup/kimina-lean-server-setup)
```

**Local Lean verification** (for `sdpo_modal_local_verify` or `lean_sdpo_local_verify_modal.py`): install [elan](https://github.com/leanprover/elan) and build a mathlib4 workspace (e.g. `Goedel-Prover-main/mathlib4`). See `devlog/20260303_local_lean_verifier_setup.md`.

## Usage

Basic usage (from project root):
```bash
python training/lean_sdpo_ttt.py --n_problems 5 --n_samples 4 --max_iterations 3
```

Full options:
```bash
python training/lean_sdpo_ttt.py \
    --model "Qwen/Qwen3-1.7B" \
    --dataset "Goedel-Prover-V2/dataset/minif2f.jsonl" \
    --n_problems 10 \
    --n_samples 8 \
    --max_iterations 5 \
    --kimina_url "http://localhost:80" \
    --temperature 0.7 \
    --kl_coef 0.1 \
    --output "results/sdpo_results.json"
```

## Key Components

### 1. LeanVerifier
Wraps the Kimina client to verify Lean 4 proofs and extract compiler errors as rich feedback.

### 2. LeanSDPO
Main class implementing test-time self-distillation:
- `generate_proofs()`: Generate multiple proof attempts
- `compute_kl_divergence()`: Compute KL between policy and reference model
- `self_distill_iteration()`: One iteration of generation + verification + feedback
- `solve_theorem()`: Full solving loop with multiple iterations

### 3. Feedback Templates
The system uses two types of feedback:
- **Error feedback**: Lean compiler errors from failed attempts
- **Solution demonstrations**: Successful proofs to guide subsequent attempts

## Configuration

Key parameters in `SDPOConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_samples` | 4 | Proof attempts per iteration |
| `max_iterations` | 3 | Maximum self-correction iterations |
| `temperature` | 0.7 | Sampling temperature |
| `kl_coef` | 0.1 | KL divergence penalty coefficient |
| `alpha` | 0.5 | KL interpolation (0=forward, 1=reverse, 0.5=JSD) |

## Output Format

Results are saved as JSON:
```json
{
  "config": {
    "model": "Qwen/Qwen3-1.7B",
    "n_samples": 4,
    "max_iterations": 3,
    ...
  },
  "summary": {
    "n_problems": 5,
    "n_solved": 2,
    "solve_rate": 0.4,
    "total_iterations": 12,
    "total_attempts": 48,
    "elapsed_seconds": 120.5
  },
  "results": [
    {
      "problem_id": "mathd_algebra_478",
      "success": true,
      "best_proof": "simp [h₁, h₂, h₃]\nring",
      "iterations": 2,
      "total_attempts": 8
    },
    ...
  ]
}
```

## Differences from Full SDPO Training

This is a **test-time** self-distillation implementation, which differs from full SDPO training:

| Aspect | Test-Time (this script) | Full SDPO Training |
|--------|------------------------|-------------------|
| Model updates | No gradient updates | Online policy updates |
| Reference model | Frozen copy | EMA-updated teacher |
| KL divergence | Monitoring only | Used in loss function |
| Scale | Single problem | Batch training |

**Full SDPO training on Modal** (Kimina verification):
- `training/lean_sdpo_kimina_2b_modal.py` — Kimina-Prover-RL-1.7B
- `training/lean_sdpo_kimina_distill_1_7b_modal.py` — Kimina-Prover-Distill-1.7B
- `training/lean_sdpo_goedel_8b_modal.py` — Goedel-Prover-V2-8B (LoRA/Unsloth)
- `training/lean_sdpo_qwen_3b_modal.py` / `lean_sdpo_qwen_3b_lora_modal.py` — Qwen 3B
- `training/lean_sdpo_deepseek_7b_modal.py` — DeepSeek 7B

**Local Lean verification** (no Kimina on Modal; verify on your machine):
- `training/lean_sdpo_local_verify_modal.py` — uses `sdpo_modal_local_verify`; requires elan + mathlib4.

The `SDPO/` directory contains the verl framework used for batch training.

## References

- [SDPO Paper (arXiv:2601.20802)](https://arxiv.org/abs/2601.20802)
- [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server)
- [Goedel-Prover-V2](https://github.com/Goedel-LM/Goedel-Prover-V2)
