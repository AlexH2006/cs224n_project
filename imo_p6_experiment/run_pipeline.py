"""
SDPO on Modal with Lean verification (Qwen3.5-4B).

TLDR: Uses sdpo_modal_local_verify_qwen; loop runs locally, generate and train step on Modal.
Verification uses the Kimina Docker server by default (http://localhost:8000). Start it with:
  docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
To use local Lean instead: --verify-backend local (or LEAN_VERIFY_BACKEND=local).

Run: modal run qwen_sdpo_old/run_pipeline.py --problem-idx 0
"""

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    from sdpo_modal_local_verify_qwen.modal_app import app
    from sdpo_modal_local_verify_qwen.modal_trainer import get_trainer_cls, normalize_gpu
    from sdpo_modal_local_verify_qwen.entrypoint import run_main

    @app.local_entrypoint()
    def main(
        model: str = "Qwen/Qwen3.5-4B",
        dataset: str = "cat-searcher/minif2f-lean4",
        dataset_subset: str = "",
        dataset_split: str = "test",
        problem_idx: int = 0,
        max_iterations: int = 5,
        learning_rate: float = 1e-5,
        temperature: float = 0.6,
        feedback_errors_only: bool = True,
        system_prompt: str = "",
        default_header: str = "",
        theorem_field: str = "",
        informal_field: str = "",
        header_field: str = "",
        gpu: str = "A100-40GB",
        verify_backend: str = "kimina",
        kimina_base_url: str = "http://localhost:8000",
    ):
        gpu_normalized = normalize_gpu(gpu)
        trainer_cls = get_trainer_cls(gpu)
        return run_main(
            trainer_cls=trainer_cls,
            model=model,
            dataset=dataset,
            dataset_subset=dataset_subset,
            dataset_split=dataset_split,
            problem_idx=problem_idx,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            temperature=temperature,
            feedback_errors_only=feedback_errors_only,
            system_prompt=system_prompt,
            default_header=default_header,
            theorem_field=theorem_field,
            informal_field=informal_field,
            header_field=header_field,
            gpu=gpu_normalized,
            verify_backend=verify_backend,
            kimina_base_url=kimina_base_url,
        )
else:
    app = None

    def main():
        print("This script requires Modal. Install with: pip install modal")
        print("Then run: modal run training/lean_sdpo_local_verify_modal.py --problem-idx <idx>")
        print("Ensure lake + mathlib4 are set up locally for verification (see devlog/20260303_local_lean_verifier_setup.md)")


if __name__ == "__main__":
    main()
