"""
SDPO on Modal with local Lean verification for Goedel-Prover-V2-8B.

TLDR: Uses sdpo_modal_local_verify_goedel (last lean4 block only, header always prepended).
Loop runs locally; generate and train step on Modal. Verification is done locally.
Requires lake + mathlib4 built locally.

Run: modal run training/lean_sdpo_goedel_local_verify_modal.py --problem-idx 0
"""

try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    from sdpo_modal_local_verify_goedel.modal_app import app
    from sdpo_modal_local_verify_goedel.modal_trainer import get_trainer_cls, normalize_gpu
    from sdpo_modal_local_verify_goedel.entrypoint import run_main

    @app.local_entrypoint()
    def main(
        model: str = "Goedel-LM/Goedel-Prover-V2-8B",
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
        gpu: str = "A100-80GB",
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
        )
else:
    app = None

    def main():
        print("This script requires Modal. Install with: pip install modal")
        print("Then run: modal run training/lean_sdpo_goedel_local_verify_modal.py --problem-idx <idx>")
        print("Ensure lake + mathlib4 are set up locally for verification (see devlog/20260303_local_lean_verifier_setup.md)")


if __name__ == "__main__":
    main()
