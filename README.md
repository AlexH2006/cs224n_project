# CS 224N Project: Test-Time RL for Theorem Proving

Test-time reinforcement learning for Lean 4 theorem proving using **SDPO (Self-Distilled Policy Optimization)**, plus model-agnostic evaluation on the MATH dataset.

## Overview

- **SDPO**: The model improves at a single problem by distilling from itself: it sees compiler feedback only when computing the teacher distribution; at test time it uses only the problem (no feedback). See [Algorithm details](docs/core_algo_explained.md). Two Modal pipelines: **Kimina 2B** (full fine-tune) and **Goedel 8B** (LoRA with Unsloth).
- **Lean verification**: Proofs are checked via [Kimina](https://projectnumina.ai) or a local Lean 4 toolchain.
- **MATH evaluation**: [eval/eval_nl_MATH.py](eval/eval_nl_MATH.py) runs few-shot MATH with local or [Modal](https://modal.com) inference.

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

For Conda, use `requirements_conda.txt` if preferred.

**Optional (Lean verification):**

- **Kimina (cloud):** Set `KIMINA_API_KEY` or `LEAN_SERVER_API_KEY` for server-side verification.
- **Local Lean 4:** Install [elan](https://github.com/leanprover/elan) and Lean 4 if you want local fallback.

**Optional (Modal):** For cloud GPU runs (SDPO or MATH eval):

```bash
pip install modal
modal token new   # one-time auth
```

## Project structure

```
.
├── README.md
├── requirements.txt
├── requirements_conda.txt     # Optional Conda environment
├── dataset/                   # Input datasets
├── devlog/                    # Change logs and engineering notes
│   ├── GOEDEL_8B_SDPO_CHANGES.md   # Goedel-8B pipeline summary
│   ├── GPU_CONFIG_NOTES.md         # GPU config and OOM notes
│   └── GENERATION_SPEED_SPECS.md   # vLLM generation speed and optimizations
├── docs/                      # Documentation
│   ├── core_algo_explained.md
│   ├── README_SDPO.md
│   ├── SDPO_TRAINER_DEEP_DIVE.md
│   └── SDPO_WORKFLOW.md
├── eval/                      # Evaluation scripts
│   ├── eval_nl_MATH.py        # MATH dataset evaluation (local or Modal)
│   ├── eval_nl_MATH/          # MATH eval outputs and sample solutions
│   ├── eval_minif2f_kimina.py
│   └── eval_minif2f_qwen.py
├── training/                  # SDPO test-time RL
│   ├── lean_sdpo_kimina_2b_modal.py   # SDPO on Modal: Kimina-Prover 1.7B
│   ├── lean_sdpo_goedel_8b_modal.py   # SDPO on Modal: Goedel-Prover-V2-8B (LoRA/Unsloth)
│   └── lean_sdpo_ttt.py       # Local SDPO test-time RL
├── verification/              # Proof verification utilities
│   ├── verify_proofs_kimina.py
│   └── verify_single_proof.py
├── scripts/                   # Shell scripts and helpers
│   ├── pipeline.sh            # Inference → compile → summarize pipeline
│   └── modal_test.py
├── setup/                     # Server and environment setup
│   └── kimina-lean-server-setup/
├── src/                       # Pipeline utilities (compile, inference, summarize)
├── SDPO/                      # SDPO/verl-related training utilities (submodule)
├── sdpo_results/              # SDPO run outputs (gitignored)
│   ├── kimina_2b/             # Kimina 2B runs: run_{problem_idx}_{timestamp}/
│   └── goedel_8b/             # Goedel 8B runs: run_{problem_idx}_{timestamp}/
└── results/                   # Other run outputs (gitignored optional)
```

## Main scripts

All commands below are run from the **project root**.

### 1. SDPO on Modal (recommended for full pipeline)

Two Modal pipelines: **Kimina 2B** (full fine-tune, A100-40GB) and **Goedel 8B** (LoRA with Unsloth, A100-80GB). Both use GPU inference and Lean verification via Kimina; results are written to the `sdpo-output` volume and synced locally.

**Kimina 2B** (`lean_sdpo_kimina_2b_modal.py`) — Kimina-Prover-RL-1.7B, full model updates:

```bash
modal run training/lean_sdpo_kimina_2b_modal.py --problem-idx 0
modal run training/lean_sdpo_kimina_2b_modal.py --max-iterations 10 --problem-idx 0
```

Local results: `sdpo_results/kimina_2b/run_{problem_idx}_{timestamp}/`.

**Goedel 8B** (`lean_sdpo_goedel_8b_modal.py`) — Goedel-Prover-V2-8B with Unsloth LoRA, gradient accumulation (default 4), proof-plan prompt format. Requires A100-80GB.

```bash
modal run training/lean_sdpo_goedel_8b_modal.py --problem-idx 1 --max-iterations 4
modal run training/lean_sdpo_goedel_8b_modal.py --problem-idx 0 --lora-rank 32 --gradient-accumulation-steps 8
```

Local results: `sdpo_results/goedel_8b/run_{problem_idx}_{timestamp}/`. For pipeline details and generation-speed notes, see [devlog/GOEDEL_8B_SDPO_CHANGES.md](devlog/GOEDEL_8B_SDPO_CHANGES.md) and [devlog/GENERATION_SPEED_SPECS.md](devlog/GENERATION_SPEED_SPECS.md).

### 2. SDPO locally

Local test-time RL loop; requires a Lean verification backend (Kimina with API key or local Lean).

```bash
python training/lean_sdpo_ttt.py --model AI-MO/Kimina-Prover-RL-1.7B --n_problems 10
python training/lean_sdpo_ttt.py --model Qwen/Qwen3-1.6B --max_iterations 5
```

### 3. MATH evaluation

Few-shot evaluation on [MATH](https://github.com/hendrycks/math) (e.g. EleutherAI/hendrycks_math). Supports local GPU or Modal.

```bash
# Local inference
python eval/eval_nl_MATH.py --model Qwen/Qwen3-1.7B --n-examples 20

# Modal inference
python eval/eval_nl_MATH.py --model Qwen/Qwen3-1.7B --n-examples 20 --modal

# Compare two models
python eval/eval_nl_MATH.py --model Qwen/Qwen3-1.7B --model2 AI-MO/Kimina-Prover-RL-1.7B --n-examples 50
```

Outputs (accuracy, sample solutions) are under `eval/eval_nl_MATH/`.

### 4. Proof verification

Verify Lean proofs via Kimina (requires a running Kimina server, e.g. Docker):

```bash
python verification/verify_proofs_kimina.py --input results/minif2f_qwen3_8b_eval.json --output results/verified.json
python verification/verify_proofs_kimina.py --server-url http://localhost:80  # optional: override server URL
```

For single-proof or local REPL verification, see `verification/verify_single_proof.py` (requires `lean_compiler` if using local REPL).

### 5. Inference–compile–summarize pipeline

From project root (uses `src/` and `dataset/`):

```bash
bash scripts/pipeline.sh
```

Configure paths and model in the CONFIGURATION section inside `scripts/pipeline.sh`.

## References

- Algorithm and loss: [docs/core_algo_explained.md](docs/core_algo_explained.md)
- **Devlog** (change logs and specs): [devlog/](devlog/) — Goedel-8B pipeline summary, GPU config notes, generation speed and optimization options
- Kimina: [projectnumina.ai](https://projectnumina.ai)
- Modal: [modal.com](https://modal.com)
- MATH: [hendrycks/math](https://github.com/hendrycks/math)

## License

See repository license file.
