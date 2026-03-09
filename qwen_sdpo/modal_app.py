"""
TLDR: Modal app entrypoint for Qwen3.5 SDPO.

Image notes (critical — do NOT change without understanding the failure chain):
  - debian_slim          → no nvcc → FlashInfer GDN JIT fails immediately
  - cuda:12.4.1-devel    → nvcc present but too old → PTX intrinsic errors
  - cuda:12.6.3-devel    → nvcc 12.6+ → FlashInfer compiles correctly ✓

Both Qwen3.5-4B and Qwen3.5-9B run on H100 (80GB). model_name is passed as a
modal.parameter so the caller picks 4B or 9B without changing any GPU config.

IMPORTANT — Modal hydration rule:
  QwenSDPOTrainer (_trainer) is instantiated at module level (below), NOT inside
  run_sdpo(). Modal only hydrates objects that exist when the app starts. Creating
  them inside a local entrypoint function causes "has not been hydrated" errors.

CLI usage:
  # 4B (default)
  python3 -m modal run qwen_sdpo/modal_app.py --model "Qwen/Qwen3.5-4B" --problem-idx 4

  # 9B
  python3 -m modal run qwen_sdpo/modal_app.py --model "Qwen/Qwen3.5-9B" --problem-idx 0

Verification runs locally via Kimina Docker — start it before running:
  docker run -d -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
"""

# Infrastructure (app, image, volumes) lives in _modal_infra to avoid circular imports:
#   modal_app.py  → imports QwenSDPOTrainer from modal_trainer.py
#   modal_trainer.py → imports app/image/volumes from _modal_infra.py
from qwen_sdpo._modal_infra import app

if app is not None:
    from qwen_sdpo.modal_trainer import QwenSDPOTrainer

    # Module-level instance — Modal hydrates objects created here, not inside local
    # entrypoint functions. model_name defaults to 4B; overridden via cfg_dict at runtime.
    _trainer = QwenSDPOTrainer()

    @app.local_entrypoint()
    def run_sdpo(
        model: str = "Qwen/Qwen3.5-4B",
        problem_idx: int = 0,
        max_iterations: int = 5,
        dataset: str = "cat-searcher/minif2f-lean4",
        dataset_split: str = "test",
        learning_rate: float = 1e-5,
        temperature: float = 0.6,
        feedback_errors_only: bool = True,
        teacher_mode: str = "full_output",
        kimina_url: str = "http://localhost:8000",
    ):
        """Local entrypoint: configure SDPOConfig and hand off to entrypoint.run_main().

        teacher_mode: One of "full_output", "answer_only", "code_only". Controls which
        part of the generation is used for the teacher in the KL loss.
        This function runs on your local machine; generate_only and run_sdpo_step
        execute on the Modal H100 GPU.
        """
        from qwen_sdpo.config import SDPOConfig
        from qwen_sdpo.entrypoint import run_main

        allowed = ("full_output", "answer_only", "code_only")
        if teacher_mode not in allowed:
            raise ValueError(f"teacher_mode must be one of {allowed}, got {teacher_mode!r}")

        cfg = SDPOConfig(
            model_name=model,
            problem_idx=problem_idx,
            max_iterations=max_iterations,
            gpu="H100",
            dataset_name=dataset,
            dataset_split=dataset_split,
            learning_rate=learning_rate,
            temperature=temperature,
            feedback_errors_only=feedback_errors_only,
            teacher_response_mode=teacher_mode,
            kimina_url=kimina_url,
        )

        # _trainer is already hydrated (module-level). model_name is forwarded
        # through cfg_dict in each .remote() call so the container loads the right weights.
        run_main(trainer=_trainer, cfg=cfg)
