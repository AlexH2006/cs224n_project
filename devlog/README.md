# Devlog

Engineering notes, bugfixes, and specs for the SDPO/Lean pipeline. Entries use a **unified naming and header format** so they sort chronologically and are easy to scan.

## Naming convention

- **Filename:** `YYYYMMDD_topic_slug.md`
  - **YYYYMMDD** — Date of the log (or primary date of the content).
  - **topic_slug** — Short lowercase slug: main topic and optional qualifier (e.g. `bugfixes_sdpo_modal`, `baseline_inspection`).
- **Header (inside each file):**
  - **Date:** `YYYY-MM-DD`
  - **Topics:** comma-separated tags (e.g. `baseline`, `bugfixes`, `sdpo`, `verification`, `vllm`).

## Index (newest first)

| Date       | File | Topics |
|------------|------|--------|
| 2026-03-03 | [20260303_sdpo_output_analysis.md](20260303_sdpo_output_analysis.md) | sdpo, reward, loss, distillation |
| 2026-03-02 | [20260302_lean_verification_system_deep_dive.md](20260302_lean_verification_system_deep_dive.md) | verification, kimina, modal, timeouts, containers, debugging |
| 2026-03-02 | [20260302_verification_slowness_iter3_4.md](20260302_verification_slowness_iter3_4.md) | verification, kimina, sdpo, performance, modal |
| 2026-03-02 | [20260302_bugfixes_verification.md](20260302_bugfixes_verification.md) | bugfixes, verification, kimina, sdpo |
| 2026-03-02 | [20260302_baseline_inspection.md](20260302_baseline_inspection.md) | baseline, evaluation, minif2f, goedel, kimina |
| 2026-03-02 | [20260302_baseline_kimina_redirect.md](20260302_baseline_kimina_redirect.md) | baseline, kimina |
| 2026-03-01 | [20260301_bugfixes_sdpo_modal.md](20260301_bugfixes_sdpo_modal.md) | bugfixes, sdpo, modal, timeouts, lora |
| 2026-03-01 | [20260301_sdpo_deepseek_7b_modal.md](20260301_sdpo_deepseek_7b_modal.md) | sdpo, deepseek, modal |
| 2026-03-01 | [20260301_timing_analysis.md](20260301_timing_analysis.md) | timing, performance, vllm |
| 2026-02-28 | [20260228_generation_speed_specs.md](20260228_generation_speed_specs.md) | vllm, generation, performance |
| 2026-02-24 | [20260224_sdpo_goedel_8b_modal.md](20260224_sdpo_goedel_8b_modal.md) | sdpo, goedel, modal |
| 2026-02-20 | [20260220_gpu_config_sdpo.md](20260220_gpu_config_sdpo.md) | gpu, config, sdpo |

## Topic tags (quick reference)

- **baseline** — Minif2f-lean4 baseline runs, pass@k, failed problems.
- **bugfixes** — Fixes and root-cause notes (timeouts, LoRA, verification, etc.).
- **config** — GPU, memory, environment configuration.
- **evaluation** — Accuracy, summaries, inspection scripts.
- **generation** — vLLM sampling, speed, max_model_len.
- **gpu** — A100/H100, OOM, memory utilization.
- **kimina** — Kimina Lean server, verification API.
- **modal** — Modal apps, volumes, timeouts.
- **sdpo** — SDPO training loop, LoRA, Unsloth.
- **timing** — Per-step timings, bottlenecks.
- **verification** — Lean verification, sorry, complete/error handling.
- **vllm** — vLLM engine, CUDA graphs, LoRA overlay.
