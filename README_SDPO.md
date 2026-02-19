# Test-Time Self-Distillation for Lean Theorem Proving

This implements a simplified test-time self-distillation pipeline based on SDPO (Self-Distilled Policy Optimization) for Lean 4 theorem proving.

## Overview

The pipeline works as follows:

```
┌─────────────────────────────────────────────────────────────┐
│                    Test-Time Self-Distillation              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Generate N proof attempts with Qwen3-1.6B               │
│                     ↓                                       │
│  2. Verify each proof with Kimina (Lean compiler)           │
│                     ↓                                       │
│  3. If success → Return proof                               │
│     If failure → Extract compiler errors as feedback        │
│                     ↓                                       │
│  4. Reprompt with feedback (self-distillation)              │
│     - Include error messages from failed attempts           │
│     - Include successful proofs as demonstrations           │
│                     ↓                                       │
│  5. Compute KL divergence for policy regularization         │
│                     ↓                                       │
│  6. Repeat until success or max iterations                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

```bash
pip install torch transformers kimina-client
```

Make sure Kimina server is running:
```bash
# Using Docker
docker run -d -p 80:80 projectnumina/kimina-lean-server:2.0.0

# Or from source (see kimina-lean-server/README.md)
```

## Usage

Basic usage:
```bash
python lean_sdpo_ttt.py --n_problems 5 --n_samples 4 --max_iterations 3
```

Full options:
```bash
python lean_sdpo_ttt.py \
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

For full SDPO training, see the `SDPO/` directory which uses the verl framework.

## References

- [SDPO Paper (arXiv:2601.20802)](https://arxiv.org/abs/2601.20802)
- [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server)
- [Goedel-Prover-V2](https://github.com/Goedel-LM/Goedel-Prover-V2)
