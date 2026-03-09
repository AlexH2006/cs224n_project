"""
qwen_sdpo — Self-Distillation Policy Optimization for Qwen3.5 on Lean 4 theorem proving.

TLDR: Test-time RL pipeline that iteratively fine-tunes Qwen3.5 (4B or 9B) on a single
problem using SDPO (KL divergence between student and teacher prompts). The student sees
only the problem; the teacher sees the problem plus Lean compiler error feedback from the
previous failed attempt. Runs generation + training on a Modal H100/A100; verification runs
locally via Kimina Docker.

Entry point:
    python3 -m modal run qwen_sdpo/modal_app.py \\
        --model "Qwen/Qwen3.5-9B" --problem-idx 0 --max-iterations 5

Package structure:
    config.py          SDPOConfig dataclass — single source of truth
    prompts.py         build_student_prompt() + build_teacher_prompt()
    sdpo_loss.py       KL divergence loss (model-agnostic)
    modal_trainer.py   QwenSDPOTrainer Modal class (vLLM + QLoRA on H100/A100)
    modal_app.py       Modal app, image definition, CLI entrypoint
    entrypoint.py      Local SDPO loop (generate → verify → update)
    results.py         Persistence: logs, metrics, KL artifacts, training curves

Self-contained (no qwen_eval import):
    parsing.py         extract_full_lean_block_parsed, create_full_lean_code,
                       get_answer_token_slice, get_code_token_slice (teacher modes)
    prompts.py         build_student_prompt(), build_teacher_prompt() (merged templates)
    _verifier.py       verify() (Kimina)
    _dataset.py        get_field()
"""
