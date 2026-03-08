# CS 224N Project: Test-Time RL for Theorem Proving

Test-time reinforcement learning for Lean 4 theorem proving using **SDPO (Self-Distilled Policy Optimization)**, plus model-agnostic evaluation on the MATH dataset.

## Overview

- **SDPO**: The model improves at a single problem by distilling from itself: it sees compiler feedback only when computing the teacher distribution; at test time it uses only the problem (no feedback). Algorithm and workflow: [docs/README_SDPO.md](docs/README_SDPO.md), [docs/SDPO_TRAINER_DEEP_DIVE.md](docs/SDPO_TRAINER_DEEP_DIVE.md). Modal pipelines: **Kimina 2B** (full fine-tune), **Kimina Distill 1.7B** (AI-MO/Kimina-Prover-Distill-1.7B), **Goedel 8B** (LoRA with Unsloth), **Qwen 3B** (optional LoRA), **DeepSeek 7B**, and **local Lean** (verify with local `lake exe repl`, no Kimina on Modal).
- **Lean verification**: Proofs are checked via [Kimina](https://projectnumina.ai) (HTTP) or a **local** Lean 4 toolchain (`lake exe repl` in a mathlib4 workspace). Three local-verify packages exist: `sdpo_modal_local_verify` (Kimina-Prover base), `sdpo_modal_local_verify_goedel` (Goedel-Prover-V2-8B, last-lean4-block parsing, truncation detection), and `sdpo_modal_local_verify_kimina` (Kimina-Prover variant).
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
├── dataset/                   # Input datasets (e.g. minif2f.jsonl, ProofNet)
├── devlog/                    # Change logs and engineering notes (see devlog/README.md)
│   ├── README.md                   # Naming convention and index
│   └── YYYYMMDD_topic_slug.md      # Dated entries
├── docs/                      # Documentation
│   ├── README_SDPO.md              # SDPO overview, test-time vs full training
│   ├── SDPO_TRAINER_DEEP_DIVE.md
│   └── SDPO_WORKFLOW.md
├── eval/                      # Evaluation scripts
│   ├── eval_nl_MATH.py        # MATH dataset evaluation (local or Modal)
│   ├── eval_nl_MATH/          # MATH eval outputs and sample solutions
│   ├── eval_minif2f_kimina.py
│   └── eval_minif2f_qwen.py
├── training/                  # SDPO test-time RL entrypoints
│   ├── lean_sdpo_kimina_2b_modal.py              # Modal: Kimina-Prover-RL-1.7B (Kimina verify)
│   ├── lean_sdpo_kimina_distill_1_7b_modal.py    # Modal: Kimina-Prover-Distill-1.7B
│   ├── lean_sdpo_goedel_8b_modal.py              # Modal: Goedel-Prover-V2-8B (LoRA/Unsloth, Kimina verify)
│   ├── lean_sdpo_goedel_local_verify_modal.py    # Modal: Goedel-Prover-V2-8B (local lake exe repl verify)
│   ├── lean_sdpo_qwen_3b_modal.py                # Modal: Qwen 3B (full or LoRA)
│   ├── lean_sdpo_qwen_3b_lora_modal.py           # Modal: Qwen 3B LoRA (Unsloth)
│   ├── lean_sdpo_deepseek_7b_modal.py            # Modal: DeepSeek 7B
│   ├── lean_sdpo_local_verify_modal.py           # Modal: Kimina-Prover; verify locally (lake exe repl)
│   ├── run_local_verify_test_set.sh              # Shell: batch run local-verify pipeline over test set
│   └── lean_sdpo_ttt.py                          # Local SDPO (no Modal)
├── sdpo_modal/                         # Kimina-based SDPO pipeline (Modal generate + Kimina verify)
├── sdpo_modal_local_verify/            # Local Lean verification pipeline — Kimina-Prover base
├── sdpo_modal_local_verify_goedel/     # Local Lean verification pipeline — Goedel-Prover-V2-8B
│                                       #   Parses last ```lean4 block only; detects truncated output;
│                                       #   always prepends default header. See parsing.py.
├── sdpo_modal_local_verify_kimina/     # Local Lean verification pipeline — Kimina-Prover variant
│                                       #   QLoRA + in-place weight sync to vLLM after each SDPO
│                                       #   step; teacher uses current iteration's feedback. See
│                                       #   sdpo_modal_local_verify_kimina/README.md
├── verification/              # Standalone proof verification utilities
│   ├── verify_proofs_kimina.py
│   └── verify_single_proof.py
├── scripts/                   # Shell scripts and helpers
│   ├── pipeline.sh            # Inference → compile → summarize pipeline
│   ├── modal_test.py
│   └── sort_kimina_2b_results.py   # Sort/inspect SDPO run results
├── setup/                     # Server and environment setup
│   └── kimina-lean-server-setup/
├── src/                       # Pipeline utilities (compile, inference, summarize)
├── baseline/                  # Minif2f-lean4 baseline eval (Modal vLLM + Kimina verify)
├── debug/                     # Tests and one-offs (see debug/README.md)
├── SDPO/                      # SDPO/verl-related training utilities (submodule)
├── Goedel-Prover-main/        # Goedel-Prover (mathlib4, REPL verifier reference)
├── sdpo_results/              # SDPO run outputs (gitignored)
│   ├── kimina_2b/
│   ├── kimina_distill_1_7b/
│   ├── goedel_8b/             # Goedel 8B runs (Kimina verify)
│   ├── deepseek_7b/
│   ├── qwen_3b/
│   ├── qwen_3b_lora/
│   └── local_verify/          # local-verify pipeline runs
│       ├── Goedel-Prover-V2-8B/
│       │   └── minif2f-lean4/
│       │       └── run_{idx}_{timestamp}/   # logs.json, metrics.json, kl/
│       └── Kimina-Prover-RL-1.7B/
│           └── minif2f-lean4/
│               └── run_{idx}_{timestamp}/   # logs.json, metrics.json, kl/, training_curves.png
└── results/                   # Other run outputs (gitignored optional)
```

## Main scripts

All commands below are run from the **project root**.

### 1. SDPO on Modal (recommended for full pipeline)

Three Modal pipelines: **Kimina 2B** (full fine-tune, A100-40GB), **Kimina Distill 1.7B** (A100-40GB), and **Goedel 8B** (LoRA with Unsloth, A100-80GB). All use GPU inference and Lean verification via Kimina; results are written to the `sdpo-output` volume and synced locally.

**Kimina 2B** (`lean_sdpo_kimina_2b_modal.py`) — Kimina-Prover-RL-1.7B, full model updates:

```bash
modal run training/lean_sdpo_kimina_2b_modal.py --problem-idx 0
modal run training/lean_sdpo_kimina_2b_modal.py --max-iterations 10 --problem-idx 0
```

Local results: `sdpo_results/kimina_2b/run_{problem_idx}_{timestamp}/`.

**Kimina Distill 1.7B** (`lean_sdpo_kimina_distill_1_7b_modal.py`) — AI-MO/Kimina-Prover-Distill-1.7B, full model updates:

```bash
modal run training/lean_sdpo_kimina_distill_1_7b_modal.py --problem-idx 0
modal run training/lean_sdpo_kimina_distill_1_7b_modal.py --max-iterations 10 --problem-idx 0
```

Local results: `sdpo_results/kimina_distill_1_7b/run_{problem_idx}_{timestamp}/`.

**Goedel 8B — Kimina verify** (`lean_sdpo_goedel_8b_modal.py`) — Goedel-Prover-V2-8B with Unsloth LoRA, gradient accumulation (default 4), proof-plan prompt format, Kimina verification. Requires A100-80GB.

```bash
modal run training/lean_sdpo_goedel_8b_modal.py --problem-idx 1 --max-iterations 4
modal run training/lean_sdpo_goedel_8b_modal.py --problem-idx 0 --lora-rank 32 --gradient-accumulation-steps 8
```

Local results: `sdpo_results/goedel_8b/run_{problem_idx}_{timestamp}/`. For pipeline details and generation-speed notes, see [devlog/](devlog/) — e.g. [20260224_sdpo_goedel_8b_modal.md](devlog/20260224_sdpo_goedel_8b_modal.md), [20260228_generation_speed_specs.md](devlog/20260228_generation_speed_specs.md).

**Goedel 8B — local verify** (`lean_sdpo_goedel_local_verify_modal.py`) — Same model; verification runs **locally** via `lake exe repl` (no Kimina on Modal). Uses `sdpo_modal_local_verify_goedel`: extracts the last `lean4` block, detects truncated output, always prepends the default Mathlib header. Requires elan + mathlib4 built (e.g. `Goedel-Prover-main/mathlib4`).

```bash
modal run training/lean_sdpo_goedel_local_verify_modal.py --problem-idx 0
modal run training/lean_sdpo_goedel_local_verify_modal.py --problem-idx 0 --max-iterations 5
```

Local results: `sdpo_results/local_verify/Goedel-Prover-V2-8B/minif2f-lean4/run_{idx}_{timestamp}/`. See [devlog/20260303_local_lean_verifier_setup.md](devlog/20260303_local_lean_verifier_setup.md), [devlog/20260304_parsing_central.md](devlog/20260304_parsing_central.md), and [devlog/20260304_kimina_server_vs_local_verify_comparison.md](devlog/20260304_kimina_server_vs_local_verify_comparison.md).

**Kimina-Prover — local verify** (`lean_sdpo_local_verify_modal.py`) — Generate and train on Modal; verification runs **locally** via `lake exe repl` (or Kimina Docker). Uses `sdpo_modal_local_verify_kimina`: QLoRA training with **in-place weight sync** to vLLM after each gradient step (via CUDA IPC), so the next generation uses the updated policy — true online RL. Teacher prompt uses the **current** iteration's compiler feedback (never stale). Requires transformers>=5.2.0. See [sdpo_modal_local_verify_kimina/README.md](sdpo_modal_local_verify_kimina/README.md).

```bash
modal run training/lean_sdpo_local_verify_modal.py --problem-idx 0
modal run training/lean_sdpo_local_verify_modal.py --problem-idx 0 --max-iterations 5
```

Local results: `sdpo_results/local_verify/Kimina-Prover-RL-1.7B/minif2f-lean4/run_{idx}_{timestamp}/`.

**Batch test-set run** (`run_local_verify_test_set.sh`) — Run the local-verify pipeline over multiple problem indices sequentially:

```bash
./training/run_local_verify_test_set.sh             # problems 0–4 (default)
./training/run_local_verify_test_set.sh 0 1 2       # specific indices
./training/run_local_verify_test_set.sh --count 10  # first 10 problems
```

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

**Kimina (HTTP):** Verify Lean proofs via Kimina (requires a running Kimina server, e.g. Docker):

```bash
python verification/verify_proofs_kimina.py --input results/minif2f_qwen3_8b_eval.json --output results/verified.json
python verification/verify_proofs_kimina.py --server-url http://localhost:80  # optional: override server URL
```

**Local Lean (lake exe repl):** Use `sdpo_modal_local_verify_goedel.local_lean_verifier.verify(lean_code)` (or the `_kimina` / base variants) from code, or run contract tests: `python debug/test_local_lean_verifier.py`. For single-proof or custom REPL scripts, see `verification/verify_single_proof.py`. Note: on recent macOS the REPL binary may fail with a `dyld __DATA_CONST` error — see [devlog/20260304_dyld_data_const_macos_repl.md](devlog/20260304_dyld_data_const_macos_repl.md); use Linux/Docker for reliable local verification.

### 5. Inference–compile–summarize pipeline

From project root (uses `src/` and `dataset/`):

```bash
bash scripts/pipeline.sh
```

Configure paths and model in the CONFIGURATION section inside `scripts/pipeline.sh`.

## References

- Algorithm and workflow: [docs/README_SDPO.md](docs/README_SDPO.md), [docs/SDPO_TRAINER_DEEP_DIVE.md](docs/SDPO_TRAINER_DEEP_DIVE.md), [docs/SDPO_WORKFLOW.md](docs/SDPO_WORKFLOW.md)
- **Devlog** (change logs and specs): [devlog/](devlog/) — index in [devlog/README.md](devlog/README.md); dated entries (YYYYMMDD_topic_slug.md) for pipeline summaries, bugfixes, GPU config, generation speed, local verification, parsing
- **Parsing notes:** [devlog/20260304_parsing_central.md](devlog/20260304_parsing_central.md) — centralized notes on model output → Lean code extraction for all pipelines
- **Local verify setup:** [devlog/20260303_local_lean_verifier_setup.md](devlog/20260303_local_lean_verifier_setup.md), [devlog/20260303_mathlib4_missing_file_not_found.md](devlog/20260303_mathlib4_missing_file_not_found.md), [devlog/20260304_dyld_data_const_macos_repl.md](devlog/20260304_dyld_data_const_macos_repl.md)
- Kimina: [projectnumina.ai](https://projectnumina.ai)
- Modal: [modal.com](https://modal.com)
- MATH: [hendrycks/math](https://github.com/hendrycks/math)

## License

See repository license file.
