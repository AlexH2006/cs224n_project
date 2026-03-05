"""
Kimina 2B SDPO on Modal. TLDR: Thin entrypoint for sdpo_modal package with Kimina model defaults.

Run: modal run training/lean_sdpo_kimina_2b_modal.py --problem-idx 0
"""

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    from sdpo_modal.modal_app import app
    from sdpo_modal.modal_trainer import SDPOTrainer
    from sdpo_modal.entrypoint import run_main

    @app.local_entrypoint()
    def main(
        model: str = "AI-MO/Kimina-Prover-RL-1.7B",
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
    ):
        return run_main(
            trainer_cls=SDPOTrainer,
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
            gpu=gpu,
            output_dir_name="kimina_2b",
        )
else:
    app = None

    def main():
        print("This script requires Modal. Install with: pip install modal")
        print("Then run: modal run training/lean_sdpo_kimina_2b_modal.py --model <model> --problem-idx <idx>")


if __name__ == "__main__":
    main()
