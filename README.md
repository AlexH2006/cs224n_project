# CS 224N Project: Test-Time RL for Theorem Proving

Test-time reinforcement learning for Lean 4 theorem proving using **self-distillation**, with **Qwen-based evaluation** on MiniF2F-Lean4 and discovery-time analysis.

> **Note:** As of March 2026, the self-distillation training pipelines that are maintained and working end-to-end are:
> - **`sdpo_modal_local_verify_kimina/`** — Kimina-Prover-RL-1.7B, local/Kimina verification, online RL via LoRA→vLLM weight sync
> - **`qwen_sdpo/`** — Qwen 3.5, Modal-based self-distillation
>
> Other pipelines (Kimina 2B, Goedel 8B, etc.) may be incomplete or untested.

## Overview

- **Self-distillation**: The model improves at a single problem by distilling from itself: it sees compiler feedback only when computing the teacher distribution; at test time it uses only the problem (no feedback). The main pipeline is **[qwen_sdpo/](qwen_sdpo/)** — Qwen 3.5 self-distillation on Modal. Algorithm and workflow: [docs/README_SDPO.md](docs/README_SDPO.md), [docs/SDPO_TRAINER_DEEP_DIVE.md](docs/SDPO_TRAINER_DEEP_DIVE.md). Other Modal SDPO pipelines (other models, local-verify options) live in [training/](training/).
- **Lean verification**: Proofs are checked via [Kimina](https://projectnumina.ai) (HTTP) or a **local** Lean 4 toolchain (`lake exe repl`). Local-verify packages: `sdpo_modal_local_verify_goedel` (Goedel-Prover-V2-8B, last-lean4-block parsing), `sdpo_modal_local_verify_kimina` (Kimina-Prover, QLoRA + weight sync).
- **Qwen evaluation**: [qwen_eval/](qwen_eval/) runs MiniF2F-Lean4 evaluation (pass@k, verification via Kimina), locally or on [Modal](https://modal.com). Used by baseline runs and by [dynamic_sampling](dynamic_sampling/) for budgeted multi-round evaluation.
- **Discovery time**: Scripts under `scripts/` and results under `results/Qwen3.5_4B_discovery_time/` compare how many generations (attempts) are needed to solve problems across methods: parallel sampling, self-distillation, and multi-turn correction.

## Setup

**Requirements:** Python 3.10+ (3.12 recommended), CUDA-capable GPU for local training/inference.

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
- **Local Lean 4:** Install [elan](https://github.com/leanprover/elan) and Lean 4 for local fallback.

**Optional (Modal):** For cloud GPU runs (self-distillation or qwen_eval):

```bash
pip install modal
modal token new   # one-time auth
```

## Project structure

```
.
├── README.md
├── requirements.txt
├── docs/                      # Documentation
│   ├── README_SDPO.md
│   ├── SDPO_TRAINER_DEEP_DIVE.md
│   └── SDPO_WORKFLOW.md
├── devlog/                    # Change logs and engineering notes (devlog/README.md)
├── training/                  # Self-distillation test-time RL entrypoints
│   ├── lean_sdpo_*_modal.py   # Modal pipelines (Kimina 2B, Goedel 8B, Qwen 3B, etc.)
│   ├── lean_sdpo_local_verify_modal.py
│   ├── lean_sdpo_ttt.py       # Local self-distillation
│   ├── sdpo_modal_local_verify_goedel/
│   └── sdpo_modal_local_verify_kimina/
├── qwen_sdpo/                 # Qwen 3.5 self-distillation (Modal)
├── qwen_eval/                 # MiniF2F-Lean4 evaluation (pass@k, Kimina verify)
│   ├── modal_app.py
│   ├── config.py, dataset.py, results.py
│   └── ...
├── dynamic_sampling/           # Multi-round eval with total attempt budget (uses qwen_eval)
│   ├── entrypoint.py
│   └── README.md
├── scripts/                   # Analysis and plotting
│   ├── plot_discovery_time_four_groups.py   # Discovery time: pass@k curves (parallel, self-distillation, multi-turn)
│   ├── plot_pass_at_k_*.py    # Pass@k comparison plots
│   ├── extract_discovery_time.py
│   └── ...
├── verification/               # Standalone proof verification
│   ├── verify_proofs_kimina.py
│   └── verify_single_proof.py
├── setup/
├── debug/                     # Tests and one-offs (debug/README.md)
├── SDPO/                      # SDPO/verl utilities (submodule)
├── results/                   # Run outputs (gitignored except results/misc/)
│   └── Qwen3.5_4B_discovery_time/   # Discovery-time summaries and pass@k plot
└── sdpo_results/              # Self-distillation run outputs (gitignored)
```

`sdpo_modal/` is gitignored (local/optional). Input datasets (e.g. minif2f) are loaded via Hugging Face or paths configured in each pipeline.

## Main scripts

Run from the **project root**.

### 1. Qwen self-distillation (qwen_sdpo)

Qwen 3.5 (4B/9B) with QLoRA on Modal; verification runs locally. Start the Lean verification server (see [docs/README_SDPO.md](docs/README_SDPO.md)), then:

```bash
# Single problem
python3 -m modal run qwen_sdpo/modal_app.py --model Qwen/Qwen3.5-4B --problem-idx 0

# Batch (multiple problems)
python3 -m modal run qwen_sdpo/modal_app.py::run_sdpo_batch --model Qwen/Qwen3.5-4B
```

See [docs/README_SDPO.md](docs/README_SDPO.md), [docs/QWEN_SDPO_WORKFLOW.md](docs/QWEN_SDPO_WORKFLOW.md), [docs/SDPO_TRAINER_DEEP_DIVE.md](docs/SDPO_TRAINER_DEEP_DIVE.md).

### 2. Qwen evaluation (MiniF2F-Lean4)

Pass@k evaluation with the `qwen_eval` package. Requires a running verification server (see qwen_eval docs).

```bash
python -m qwen_eval.main --model Qwen/Qwen3.5-4B --pass-k 32 --problem-idx-file qwen_eval/problem_idx.json
```

### 3. Dynamic sampling (budgeted multi-round eval)

Pass@1 then repeated rounds on unsolved problems until a total attempt budget. Uses qwen_eval.

```bash
python -m dynamic_sampling.entrypoint --budget 256
python -m dynamic_sampling.entrypoint --budget 512 --model Qwen/Qwen3.5-9B
```

See [dynamic_sampling/README.md](dynamic_sampling/README.md).

### 4. Discovery time and pass@k plots

Discovery time = number of generations until first verified solution. Compare parallel sampling, self-distillation (KL avg/sum), and multi-turn correction:

```bash
python scripts/plot_discovery_time_four_groups.py
```

Output: `results/Qwen3.5_4B_discovery_time/pass_at_k_discovery_time_four_groups.png`.

### 5. Proof verification

Batch-verify proofs from a JSON eval file (requires a verification server):

```bash
python verification/verify_proofs_kimina.py --input results/minif2f_qwen3_8b_eval.json --output results/verified.json
```

## References

- Self-distillation (SDPO): [docs/README_SDPO.md](docs/README_SDPO.md), [docs/QWEN_SDPO_WORKFLOW.md](docs/QWEN_SDPO_WORKFLOW.md), [docs/SDPO_TRAINER_DEEP_DIVE.md](docs/SDPO_TRAINER_DEEP_DIVE.md)
- Devlog (specs, bugfixes, setup): [devlog/](devlog/) — index in [devlog/README.md](devlog/README.md)
- [Modal](https://modal.com)

## License

See repository license file.
