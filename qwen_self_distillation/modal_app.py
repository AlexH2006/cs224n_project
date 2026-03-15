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

  # Non-thinking mode (no <think> blocks; teacher prompt includes full generation)
  python3 -m modal run qwen_sdpo/modal_app.py --model "Qwen/Qwen3.5-4B" --use-think-mode false

  # Batch: train on multiple problems; default loads qwen_sdpo/problem_idx.json
  python3 -m modal run qwen_sdpo/modal_app.py::run_sdpo_batch --model "Qwen/Qwen3.5-4B"
  # Override: --problems-json results/.../sampled_problems.json (relative = project root)

Verification runs locally via Kimina Docker — start it before running:
  docker run -d -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
"""

# Infrastructure (app, image, volumes) lives in _modal_infra to avoid circular imports:
#   modal_app.py  → imports QwenSDPOTrainer from modal_trainer.py
#   modal_trainer.py → imports app/image/volumes from _modal_infra.py
import dataclasses
from qwen_sdpo._modal_infra import app
from qwen_sdpo.config import SDPOConfig

# Single source of truth for default learning rate (used by CLI when --learning-rate omitted)
_DEFAULT_LR = next(f.default for f in dataclasses.fields(SDPOConfig) if f.name == "learning_rate")

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
        learning_rate: float = _DEFAULT_LR,
        temperature: float = 0.6,
        feedback_errors_only: bool = True,
        use_think_mode: bool = True,
        teacher_mode: str = "full_output",
        minibatch_size: int = 1,
        kl_mask_final_code_only: bool = False,
        kimina_url: str = "http://localhost:8000",
        results_base_dir: str = "sdpo_results",
    ):
        """Local entrypoint: configure SDPOConfig and hand off to entrypoint.run_main().

        minibatch_size: Number of on-policy samples per iteration (1 = single-sample; >1 = minibatch).
        kl_mask_final_code_only: When True, KL loss is summed only over the final code block; context remains full.
        use_think_mode: When True, Qwen3.5 may emit <think> reasoning (default). When False,
            non-thinking mode: student and teacher use enable_thinking=False; teacher
            prompt includes the full failed generation.
        teacher_mode: One of "full_output", "answer_only", "code_only". Controls which
            part of the generation is used for the teacher in the KL loss.
        results_base_dir: Base directory for local run output (default: sdpo_results).
            Output is written to {results_base_dir}/{model_tag}/{teacher_mode}/run_.../
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
            use_think_mode=use_think_mode,
            teacher_response_mode=teacher_mode,
            minibatch_size=minibatch_size,
            kl_mask_final_code_only=kl_mask_final_code_only,
            kimina_url=kimina_url,
            results_base_dir=results_base_dir,
        )

        # _trainer is already hydrated (module-level). model_name is forwarded
        # through cfg_dict in each .remote() call so the container loads the right weights.
        run_main(trainer=_trainer, cfg=cfg)

    @app.local_entrypoint()
    def run_sdpo_batch(
        problems_json: str = "",
        model: str = "Qwen/Qwen3.5-4B",
        max_iterations: int = 5,
        minibatch_size: int = 1,
        dataset: str = "cat-searcher/minif2f-lean4",
        dataset_split: str = "test",
        learning_rate: float = _DEFAULT_LR,
        temperature: float = 0.6,
        feedback_errors_only: bool = True,
        use_think_mode: bool = True,
        teacher_mode: str = "full_output",
        kl_mask_final_code_only: bool = False,
        kimina_url: str = "http://localhost:8000",
    ):
        """Train on multiple problems; each problem from base HF model.

        By default loads from qwen_sdpo/problem_idx.json (same folder as this app).
        Pass --problems-json <path> to use another file; relative paths are resolved
        against the project root. JSON schema: problem_indices, problems (problem_idx,
        problem_id). Each problem is trained from the base HuggingFace model
        (reset_to_base between problems). Results and manifest under sdpo_results.
        """
        import json
        from dataclasses import replace
        from datetime import datetime
        from pathlib import Path

        from qwen_sdpo.config import SDPOConfig
        from qwen_sdpo.entrypoint import run_main_batch

        _qwen_sdpo_dir = Path(__file__).resolve().parent
        _project_root = _qwen_sdpo_dir.parent
        default_problems_path = _qwen_sdpo_dir / "problem_idx.json"

        if not (problems_json and problems_json.strip()):
            path = default_problems_path
        else:
            path = Path(problems_json)
            if not path.is_absolute():
                path = (_project_root / path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"problems_json not found: {path}")

        with open(path) as f:
            data = json.load(f)
        problem_indices = data.get("problem_indices", [])
        problems = data.get("problems", [])
        if not problem_indices:
            raise ValueError("problems_json must contain a non-empty 'problem_indices' list")
        problem_id_by_idx = {p["problem_idx"]: p["problem_id"] for p in problems}

        n = len(problem_indices)
        first_few = problem_indices[: min(5, n)]
        print(f"Problems JSON: {path}")
        print(f"Problems: {n} indices, first: {first_few}")

        allowed = ("full_output", "answer_only", "code_only")
        if teacher_mode not in allowed:
            raise ValueError(f"teacher_mode must be one of {allowed}, got {teacher_mode!r}")

        cfg = SDPOConfig(
            model_name=model,
            problem_idx=problem_indices[0],  # overwritten per problem in run_main_batch
            max_iterations=max_iterations,
            minibatch_size=minibatch_size,
            gpu="H100",
            dataset_name=dataset,
            dataset_split=dataset_split,
            learning_rate=learning_rate,
            temperature=temperature,
            feedback_errors_only=feedback_errors_only,
            use_think_mode=use_think_mode,
            teacher_response_mode=teacher_mode,
            kl_mask_final_code_only=kl_mask_final_code_only,
            kimina_url=kimina_url,
        )

        # One run_Qwen3.5-4B_{timestamp} folder under sdpo_results/Qwen3.5-4B; inside it:
        #   runs/problem_7/, runs/problem_9/, ... (one folder per problem)
        #   manifest/checkpoint_manifest.json
        model_tag = model.split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_run_dir = Path(cfg.results_base_dir) / model_tag / f"run_{model_tag}_{timestamp}"
        batch_run_dir.mkdir(parents=True, exist_ok=True)
        cfg = replace(cfg, batch_run_dir=str(batch_run_dir))
        manifest_path = batch_run_dir / "manifest" / "checkpoint_manifest.json"
        summary = run_main_batch(
            trainer=_trainer,
            cfg=cfg,
            problem_indices=problem_indices,
            problem_id_by_idx=problem_id_by_idx,
            manifest_path=manifest_path,
        )
        print(f"\nBatch complete: {summary['success_count']}/{summary['total']} succeeded.")
        print(f"Manifest: {summary['manifest_path']}")
