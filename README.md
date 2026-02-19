# CS 224N Project: Test-Time RL for Theorem Proving

Test-time reinforcement learning for Lean 4 theorem proving using **SDPO (Self-Distilled Policy Optimization)**, plus model-agnostic evaluation on the MATH dataset.

## Overview

- **SDPO**: The model improves at a single problem by distilling from itself: it sees compiler feedback only when computing the teacher distribution; at test time it uses only the problem (no feedback). See [`core_algo_explained.md`](core_algo_explained.md) for the algorithm details.
- **Lean verification**: Proofs are checked via [Kimina](https://projectnumina.ai) or a local Lean 4 toolchain.
- **MATH evaluation**: [`eval_nl_MATH.py`](eval_nl_MATH.py) runs few-shot MATH with local or [Modal](https://modal.com) inference.

## Setup

**Requirements:** Python 3.10+ (3.12 recommended), a CUDA-capable GPU for local training/inference.

```bash
# Clone and enter repo
git clone https://github.com/AlexH2006/cs224n_project.git
cd cs224n_project

# Create a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Optional (Lean verification):**

- **Kimina (cloud):** Set `KIMINA_API_KEY` or `LEAN_SERVER_API_KEY` for server-side verification.
- **Local Lean 4:** Install [elan](https://github.com/leanprover/elan) and Lean 4 if you want local fallback.

**Optional (Modal):** For cloud GPU runs (SDPO or MATH eval):

```bash
pip install modal
modal token new   # one-time auth
```

## Main Scripts

### 1. SDPO on Modal (recommended for full pipeline)

Runs SDPO test-time RL on Modal: GPU inference, Lean verification via Kimina (or local Lean in the image), persistent HF cache and output volumes.

```bash
# Default: Kimina-Prover-RL-1.7B, minif2f-lean4, problem index 0
modal run lean_sdpo_modal.py --model AI-MO/Kimina-Prover-RL-1.7B --problem-idx 0

# Custom model and dataset
modal run lean_sdpo_modal.py --model Goedel-LM/Goedel-Prover-V2-8B --dataset deepmind/math --problem-idx 5

# More iterations per problem
modal run lean_sdpo_modal.py --max-iterations 10 --problem-idx 0
```

Results and training curves are written to the Modal volume `sdpo-output` and synced to `sdpo_results/` (see script output for paths).

### 2. SDPO locally (`lean_sdpo_ttt.py`)

Local test-time RL loop; requires a Lean verification backend (Kimina with API key or local Lean).

```bash
python lean_sdpo_ttt.py --model AI-MO/Kimina-Prover-RL-1.7B --n_problems 10
python lean_sdpo_ttt.py --model Qwen/Qwen3-1.6B --max_iterations 5
```

### 3. MATH evaluation (`eval_nl_MATH.py`)

Few-shot evaluation on [MATH](https://github.com/hendrycks/math) (e.g. EleutherAI/hendrycks_math). Supports local GPU or Modal.

```bash
# Local inference
python eval_nl_MATH.py --model Qwen/Qwen3-1.7B --n-examples 20

# Modal inference
python eval_nl_MATH.py --model Qwen/Qwen3-1.7B --n-examples 20 --modal

# Compare two models
python eval_nl_MATH.py --model Qwen/Qwen3-1.7B --model2 AI-MO/Kimina-Prover-RL-1.7B --n-examples 50
```

Outputs (accuracy, sample solutions) are under `eval_nl_MATH/`.

## Project structure

```
.
├── README.md
├── requirements.txt
├── core_algo_explained.md      # SDPO algorithm and loss details
├── lean_sdpo_modal.py          # SDPO on Modal (GPU + Lean verification)
├── lean_sdpo_ttt.py            # Local SDPO test-time RL
├── eval_nl_MATH.py             # MATH dataset evaluation (local or Modal)
├── eval_nl_MATH/               # MATH eval outputs and sample solutions
├── sdpo_results/               # SDPO run logs, metrics, training curves
├── SDPO/                       # SDPO/verl-related training utilities
├── src/                        # Utilities (compile, inference, summarize)
└── lean_compiler/              # Lean REPL/scheduler helpers
```

## References

- Algorithm and loss: [`core_algo_explained.md`](core_algo_explained.md)
- Kimina: [projectnumina.ai](https://projectnumina.ai)
- Modal: [modal.com](https://modal.com)
- MATH: [hendrycks/math](https://github.com/hendrycks/math)

## License

See repository license file.
