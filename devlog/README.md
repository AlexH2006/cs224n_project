# Devlog

Engineering notes, bugfixes, and specs for the SDPO/Lean pipeline. Entries use a **unified naming and header format** so they sort chronologically and are easy to scan.

## Naming convention

- **Filename:** `YYYYMMDD_topic_slug.md`
  - **YYYYMMDD** — Date of the log (or primary date of the content).
  - **topic_slug** — Short lowercase slug: main topic and optional qualifier (e.g. `bugfixes_sdpo_modal`, `baseline_inspection`).
- **Header (inside each file):**
  - **Date:** `YYYY-MM-DD`
  - **Topics:** comma-separated tags (e.g. `baseline`, `bugfixes`, `sdpo`, `verification`, `vllm`).

---

## Categories

Devlogs are grouped by primary focus:

| Category | Description |
|----------|-------------|
| **Verification** | Lean verification logic, Kimina server, success/complete/sorry handling, timeouts, debug logging. |
| **Inference speed** | vLLM generation, sampling params, max_model_len, per-step timing, bottlenecks. |
| **SDPO / training** | Training loop, models (Goedel, DeepSeek, etc.), reward/loss, distillation, output analysis. |
| **Parsing** | Model output → Lean code extraction: block detection, truncation, full-block vs tactics, Goedel vs Kimina. |
| **Bugfixes** | Root-cause notes and fixes: timeouts, LoRA, verification behavior, Modal. |
| **Baseline / evaluation** | Minif2f-lean4 baselines, pass@k, failed problems, Kimina redirect, inspection. |
| **Config** | GPU (A100/H100), memory, Modal config, environment. |
| **Setup** | Local toolchain setup: elan, mathlib4, macOS REPL issues. |

---

## Index by category

### Verification

| Date       | File | Topics |
|------------|------|--------|
| 2026-03-04 | [20260304_kimina_server_vs_local_verify_comparison.md](20260304_kimina_server_vs_local_verify_comparison.md) | verification, kimina, local_lean, repl, comparison |
| 2026-03-03 | [20260303_goedel_prover_lean_verification_analysis.md](20260303_goedel_prover_lean_verification_analysis.md) | verification, goedel, modal, kimina, repl, contract |
| 2026-03-03 | [20260303_lean_verification_sdpo_integration_plan.md](20260303_lean_verification_sdpo_integration_plan.md) | verification, sdpo, modal, local_verify, integration |
| 2026-03-03 | [20260303_local_lean_verifier_setup.md](20260303_local_lean_verifier_setup.md) | verification, local_lean, lake, mathlib4, sdpo_modal |
| 2026-03-03 | [20260303_lean3_syntax_logged_as_success.md](20260303_lean3_syntax_logged_as_success.md) | verification, kimina, lean3, lean4, sdpo, logs, bugfixes |
| 2026-03-03 | [20260303_verification_logic_sdpo_modal.md](20260303_verification_logic_sdpo_modal.md) | verification, kimina, sdpo, modal, logic, success, complete, sorry |
| 2026-03-02 | [20260302_lean_verification_system_deep_dive.md](20260302_lean_verification_system_deep_dive.md) | verification, kimina, modal, timeouts, containers, debugging |
| 2026-03-02 | [20260302_verification_slowness_iter3_4.md](20260302_verification_slowness_iter3_4.md) | verification, kimina, sdpo, performance, modal |
| 2026-03-02 | [20260302_bugfixes_verification.md](20260302_bugfixes_verification.md) | bugfixes, verification, kimina, sdpo |

### Inference speed

| Date       | File | Topics |
|------------|------|--------|
| 2026-03-01 | [20260301_timing_analysis.md](20260301_timing_analysis.md) | timing, performance, vllm |
| 2026-02-28 | [20260228_generation_speed_specs.md](20260228_generation_speed_specs.md) | vllm, generation, performance |

### SDPO / training

| Date       | File | Topics |
|------------|------|--------|
| 2026-03-03 | [20260303_qwen_3b_lora_adaptation_plan.md](20260303_qwen_3b_lora_adaptation_plan.md) | sdpo, qwen, lora, unsloth, modal, oom, memory, config |
| 2026-03-03 | [20260303_sdpo_output_analysis.md](20260303_sdpo_output_analysis.md) | sdpo, reward, loss, distillation |
| 2026-03-01 | [20260301_sdpo_deepseek_7b_modal.md](20260301_sdpo_deepseek_7b_modal.md) | sdpo, deepseek, modal |
| 2026-02-24 | [20260224_sdpo_goedel_8b_modal.md](20260224_sdpo_goedel_8b_modal.md) | sdpo, goedel, modal |

### Parsing

| Date       | File | Topics |
|------------|------|--------|
| 2026-03-04 | [20260304_parsing_central.md](20260304_parsing_central.md) | parsing, goedel, kimina, sdpo, lean4, truncation, full_block |

### Bugfixes

| Date       | File | Topics |
|------------|------|--------|
| 2026-03-03 | [20260303_lean3_syntax_logged_as_success.md](20260303_lean3_syntax_logged_as_success.md) | verification, kimina, lean3, lean4, sdpo, logs, bugfixes |
| 2026-03-02 | [20260302_bugfixes_verification.md](20260302_bugfixes_verification.md) | bugfixes, verification, kimina, sdpo |
| 2026-03-01 | [20260301_bugfixes_sdpo_modal.md](20260301_bugfixes_sdpo_modal.md) | bugfixes, sdpo, modal, timeouts, lora |

### Baseline / evaluation

| Date       | File | Topics |
|------------|------|--------|
| 2026-03-02 | [20260302_baseline_inspection.md](20260302_baseline_inspection.md) | baseline, evaluation, minif2f, goedel, kimina |
| 2026-03-02 | [20260302_baseline_kimina_redirect.md](20260302_baseline_kimina_redirect.md) | baseline, kimina |

### Config

| Date       | File | Topics |
|------------|------|--------|
| 2026-03-03 | [20260303_qwen_3b_lora_adaptation_plan.md](20260303_qwen_3b_lora_adaptation_plan.md) | sdpo, qwen, lora, unsloth, modal, oom, memory, config |
| 2026-02-20 | [20260220_gpu_config_sdpo.md](20260220_gpu_config_sdpo.md) | gpu, config, sdpo |

### Setup

| Date       | File | Topics |
|------------|------|--------|
| 2026-03-04 | [20260304_dyld_data_const_macos_repl.md](20260304_dyld_data_const_macos_repl.md) | setup, macos, repl, lean4, local_lean, dyld |
| 2026-03-03 | [20260303_mathlib4_missing_file_not_found.md](20260303_mathlib4_missing_file_not_found.md) | setup, mathlib4, local_lean, submodule |
| 2026-03-03 | [20260303_local_lean_verifier_setup.md](20260303_local_lean_verifier_setup.md) | verification, local_lean, lake, mathlib4, sdpo_modal |

---

## Chronological index (newest first)

| Date       | File | Categories |
|------------|------|------------|
| 2026-03-04 | [20260304_parsing_central.md](20260304_parsing_central.md) | Parsing |
| 2026-03-04 | [20260304_kimina_server_vs_local_verify_comparison.md](20260304_kimina_server_vs_local_verify_comparison.md) | Verification |
| 2026-03-04 | [20260304_dyld_data_const_macos_repl.md](20260304_dyld_data_const_macos_repl.md) | Setup |
| 2026-03-03 | [20260303_qwen_3b_lora_adaptation_plan.md](20260303_qwen_3b_lora_adaptation_plan.md) | SDPO / training, Config |
| 2026-03-03 | [20260303_mathlib4_missing_file_not_found.md](20260303_mathlib4_missing_file_not_found.md) | Setup |
| 2026-03-03 | [20260303_goedel_prover_lean_verification_analysis.md](20260303_goedel_prover_lean_verification_analysis.md) | Verification |
| 2026-03-03 | [20260303_lean_verification_sdpo_integration_plan.md](20260303_lean_verification_sdpo_integration_plan.md) | Verification, SDPO / training |
| 2026-03-03 | [20260303_local_lean_verifier_setup.md](20260303_local_lean_verifier_setup.md) | Verification, Setup |
| 2026-03-03 | [20260303_lean3_syntax_logged_as_success.md](20260303_lean3_syntax_logged_as_success.md) | Verification, Bugfixes |
| 2026-03-03 | [20260303_verification_logic_sdpo_modal.md](20260303_verification_logic_sdpo_modal.md) | Verification |
| 2026-03-03 | [20260303_sdpo_output_analysis.md](20260303_sdpo_output_analysis.md) | SDPO / training |
| 2026-03-02 | [20260302_lean_verification_system_deep_dive.md](20260302_lean_verification_system_deep_dive.md) | Verification |
| 2026-03-02 | [20260302_verification_slowness_iter3_4.md](20260302_verification_slowness_iter3_4.md) | Verification |
| 2026-03-02 | [20260302_bugfixes_verification.md](20260302_bugfixes_verification.md) | Verification, Bugfixes |
| 2026-03-02 | [20260302_baseline_inspection.md](20260302_baseline_inspection.md) | Baseline / evaluation |
| 2026-03-02 | [20260302_baseline_kimina_redirect.md](20260302_baseline_kimina_redirect.md) | Baseline / evaluation |
| 2026-03-01 | [20260301_bugfixes_sdpo_modal.md](20260301_bugfixes_sdpo_modal.md) | Bugfixes |
| 2026-03-01 | [20260301_sdpo_deepseek_7b_modal.md](20260301_sdpo_deepseek_7b_modal.md) | SDPO / training |
| 2026-03-01 | [20260301_timing_analysis.md](20260301_timing_analysis.md) | Inference speed |
| 2026-02-28 | [20260228_generation_speed_specs.md](20260228_generation_speed_specs.md) | Inference speed |
| 2026-02-24 | [20260224_sdpo_goedel_8b_modal.md](20260224_sdpo_goedel_8b_modal.md) | SDPO / training |
| 2026-02-20 | [20260220_gpu_config_sdpo.md](20260220_gpu_config_sdpo.md) | Config |

---

## Topic tags (quick reference)

- **baseline** — Minif2f-lean4 baseline runs, pass@k, failed problems.
- **bugfixes** — Fixes and root-cause notes (timeouts, LoRA, verification, etc.).
- **config** — GPU, memory, environment configuration.
- **distillation** — SDPO loss/reward, student–teacher log-prob difference, output analysis.
- **evaluation** — Accuracy, summaries, inspection scripts.
- **generation** — vLLM sampling, speed, max_model_len.
- **goedel** — Goedel-Prover, REPL verifier, mathlib4.
- **gpu** — A100/H100, OOM, memory utilization.
- **integration** — Pipeline integration (e.g. local verify + SDPO Modal).
- **kimina** — Kimina Lean server, verification API.
- **local_lean** — Local Lean verification (lake exe repl, no Kimina).
- **modal** — Modal apps, volumes, timeouts.
- **sdpo** — SDPO training loop, LoRA, Unsloth.
- **timing** — Per-step timings, bottlenecks.
- **verification** — Lean verification, sorry, complete/error handling.
- **vllm** — vLLM engine, CUDA graphs, LoRA overlay.
- **lora** — LoRA adapters, Unsloth, 4-bit, memory-efficient training.
- **unsloth** — Unsloth 4-bit + LoRA training stack.
- **oom** — Out-of-memory, GPU memory budget, mitigation (LoRA, gradient checkpointing).
- **parsing** — Model output → Lean code extraction: block detection, truncation, full-block vs tactics, Goedel vs Kimina.
- **truncation** — Detecting cut-off generation (unclosed lean4 fence, no closing ```).
- **full_block** — Full lean4 block extraction (vs tactic-only extraction).
- **comparison** — Side-by-side comparison of approaches (e.g. Kimina server vs local verify).
- **macos** — macOS-specific issues (dyld, REPL binary compatibility).
- **submodule** — Git submodule setup (mathlib4, Goedel-Prover-main).
- **setup** — Local toolchain setup: elan, mathlib4, macOS REPL issues.
