"""
CLI entrypoint logic: load dataset, build config, run trainer on Modal, save local copy.

TLDR: run_main() does dataset load, config dict build, trainer.run_sdpo.remote(), and
local persistence. Used by: thin runner script that registers run_main as @app.local_entrypoint().
"""

from typing import Any, Optional, Type

from sdpo_modal.config import SDPOConfig
from sdpo_modal.utils import (
    clamp_problem_idx,
    load_dataset_with_fallback,
    plot_training_curves,
    print_gpu_unsupported_warning,
    print_problem_summary,
    print_run_banner,
    save_local_run,
)


def run_main(
    trainer_cls: Type[Any],
    config: Optional[SDPOConfig] = None,
    **kwargs: Any,
) -> dict:
    """Load dataset, build config, run trainer.run_sdpo.remote(), print results, save local copy. Returns results dict.
    All run parameters default from SDPOConfig; pass config= or any keyword to override (e.g. model=..., problem_idx=...).
    """
    cfg = config if config is not None else SDPOConfig()
    model = kwargs.get("model", cfg.model_name)
    dataset = kwargs.get("dataset", cfg.dataset_name)
    dataset_subset = kwargs.get("dataset_subset", cfg.dataset_subset) or ""
    dataset_split = kwargs.get("dataset_split", cfg.dataset_split)
    problem_idx = kwargs.get("problem_idx", cfg.problem_idx)
    max_iterations = kwargs.get("max_iterations", cfg.max_iterations)
    learning_rate = kwargs.get("learning_rate", cfg.learning_rate)
    temperature = kwargs.get("temperature", cfg.temperature)
    feedback_errors_only = kwargs.get("feedback_errors_only", cfg.feedback_errors_only)
    system_prompt = kwargs.get("system_prompt", cfg.system_prompt) or ""
    default_header = kwargs.get("default_header", cfg.default_header) or ""
    theorem_field = kwargs.get("theorem_field", cfg.theorem_field_override) or ""
    informal_field = kwargs.get("informal_field", cfg.informal_field_override) or ""
    header_field = kwargs.get("header_field", cfg.header_field_override) or ""
    gpu = kwargs.get("gpu", cfg.gpu)
    output_dir_name = kwargs.get("output_dir_name", cfg.output_dir)

    print_run_banner(
        model, gpu, dataset, dataset_subset, dataset_split,
        problem_idx, max_iterations, feedback_errors_only,
    )

    ds = load_dataset_with_fallback(dataset, dataset_subset, dataset_split)
    problem_idx = clamp_problem_idx(problem_idx, len(ds))
    problem = dict(ds[problem_idx])

    theorem_fields = ([theorem_field] if theorem_field else []) + cfg.theorem_fields
    informal_fields = ([informal_field] if informal_field else []) + cfg.informal_fields
    header_fields = ([header_field] if header_field else []) + cfg.header_fields
    id_fields = cfg.id_fields

    print_problem_summary(
        problem, problem_idx, id_fields,
        theorem_fields, informal_fields, header_fields,
    )

    config_dict = {
        "model_name": model,
        "dataset_name": dataset,
        "dataset_subset": dataset_subset or None,
        "dataset_split": dataset_split,
        "problem_idx": problem_idx,
        "max_iterations": max_iterations,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "feedback_include_failed_proof": not feedback_errors_only,
        "theorem_fields": theorem_fields,
        "informal_fields": informal_fields,
        "header_fields": header_fields,
        "id_fields": id_fields,
        "output_dir": output_dir_name,
    }
    if system_prompt:
        config_dict["system_prompt"] = system_prompt
    if default_header:
        config_dict["default_header"] = default_header

    print_gpu_unsupported_warning(gpu)
    print("Using A100-40GB GPU (for 1-2B models)")
    trainer = trainer_cls(model_name=model)
    print("Starting SDPO training on Modal...")
    results = trainer.run_sdpo.remote(config_dict, problem)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Success: {results['success']}")
    print(f"Iterations used: {len(results['iteration_logs'])}")
    if results["success"]:
        print(f"Best proof: {results['best_proof'][:200]}...")
    if results["metrics"]["losses"]:
        print(f"\nFinal metrics:")
        print(f"  Final loss: {results['metrics']['losses'][-1]:.4f}")
        print(f"  Final entropy: {results['metrics']['entropies'][-1]:.4f}")
        print(f"  Final grad norm: {results['metrics']['grad_norms'][-1]:.4f}")
    print(f"\nResults saved to Modal volume 'sdpo-output' under '{output_dir_name}/'")

    run_dir = save_local_run(results, output_dir_name, dataset, problem_idx)
    metrics = results.get("metrics", {})
    if metrics.get("iterations") and len(metrics["iterations"]) > 0:
        plot_training_curves(
            metrics,
            run_dir / "training_curves.png",
            title=f"SDPO Test-Time RL - Problem {problem_idx}",
        )
        print(f"Training curves saved to: {run_dir / 'training_curves.png'}")
    return results
