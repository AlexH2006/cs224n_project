"""
Shared utilities for SDPO: dataset access, proof/tactic extraction, Lean code assembly, and persistence.

TLDR: Single module for (1) get_field/load_problem (dataset-agnostic problem dict access),
(2) extract_tactics_from_code_block, extract_proof_tactics, theorem_code_is_commented_out
(pure parsing of model output and theorem code), (3) create_full_lean_code (assemble
full Lean file from theorem + tactics + header), and (4) plot_training_curves/save_run
(write logs, metrics, plots, and optional model to disk). Used by: trainer_core, prompts,
entrypoint, modal_trainer.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sdpo_modal.config import SDPOConfig


# -----------------------------------------------------------------------------
# Dataset helpers: field resolution and loading
# -----------------------------------------------------------------------------


def get_field(data: dict, field_names: list, default: str = "") -> str:
    """Get a field from data dict, trying multiple possible field names.

    Enables dataset-agnostic loading by supporting various naming conventions
    (e.g. lean4_code vs lean4_statement vs statement).
    """
    for field_name in field_names:
        if field_name in data and data[field_name]:
            value = data[field_name]
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)) and len(value) > 0:
                return str(value[0])
    return default


def load_problem(
    dataset_name: str,
    split: str,
    problem_idx: int,
    subset: Optional[str] = None,
) -> dict:
    """Load a single problem from a HuggingFace dataset. Returns the problem as a dict."""
    from datasets import load_dataset

    if subset:
        ds = load_dataset(dataset_name, subset, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    if problem_idx >= len(ds):
        problem_idx = 0
    return dict(ds[problem_idx])


# -----------------------------------------------------------------------------
# Proof / tactic extraction from model output and code blocks
# -----------------------------------------------------------------------------


def extract_tactics_from_code_block(block: str) -> str:
    """Extract tactics from a single code block, filtering out non-tactic lines.

    Preserves import/open/set_option statements so they can be used as header
    when the dataset doesn't provide one.
    """
    lines = []
    for line in block.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("--"):
            continue
        if " --" in stripped:
            stripped = stripped[: stripped.index(" --")].rstrip()
            if not stripped:
                continue
        if stripped.startswith("import "):
            lines.append(stripped)
            continue
        if stripped.startswith("open "):
            lines.append(stripped)
            continue
        if stripped.startswith("set_option "):
            lines.append(stripped)
            continue
        if stripped.startswith("theorem ") or stripped.startswith("lemma "):
            continue
        if stripped.startswith("(") and ":" in stripped:
            continue
        if ":= sorry" in stripped:
            continue
        if re.search(r"^\d+\s*\*.*:=\s*(by|sorry)", stripped):
            continue
        if re.search(r"≠\s*\d+\s*:=", stripped):
            continue
        if re.search(r"[≠=<>]\s*\d+\s*:=", stripped):
            continue
        if stripped.endswith(":=") or stripped.endswith(":= by"):
            continue
        if re.search(r"^\d+\s*\*.*[≠=<>]", stripped) and ":=" not in stripped and "by" not in stripped.lower():
            continue
        lines.append(stripped)
    return "\n".join(lines)


def extract_proof_tactics(output: str) -> str:
    """Extract proof tactics from model output.

    Supports <think>reasoning</think> then answer; extracts from content after </think>,
    from ```lean4/lean code blocks, and from ":= by" pattern. If reasoning
    is incomplete (<think> without </think>), returns "sorry".
    """
    output = output.strip()
    if "<think>" in output and "</think>" not in output:
        return "sorry"
    tactics = None
    code_pattern = r"```(?:lean4?|lean|tactics)?\n?(.*?)```"

    if "</think>" in output:
        after_think = output.split("</think>")[-1].strip()
        if after_think:
            matches = re.findall(code_pattern, after_think, re.DOTALL)
            if matches:
                for match in matches:
                    extracted = extract_tactics_from_code_block(match)
                    if extracted and extracted.lower() not in ["sorry", "by"]:
                        tactics = extracted
                        break
            if not tactics:
                extracted = extract_tactics_from_code_block(after_think)
                if extracted and extracted.lower() not in ["sorry", "by"]:
                    tactics = extracted

    if not tactics:
        matches = re.findall(code_pattern, output, re.DOTALL)
        if matches:
            for match in matches:
                extracted = extract_tactics_from_code_block(match)
                if extracted and extracted.lower() not in ["sorry", "by"]:
                    tactics = extracted
                    break

    if not tactics and ":= by" in output:
        by_idx = output.rfind(":= by")
        after_by = output[by_idx + 5 :].strip()
        if "```" in after_by:
            after_by = after_by.split("```")[0]
        tactic_lines = []
        for line in after_by.split("\n")[:10]:
            stripped = line.strip()
            if stripped and stripped.lower() not in ["sorry", "by", ""]:
                if not stripped.startswith("--"):
                    tactic_lines.append(stripped)
        if tactic_lines:
            tactics = "\n".join(tactic_lines)

    if tactics:
        if tactics.lower().startswith("by\n") or tactics.lower().startswith("by "):
            tactics = tactics[2:].strip()
        elif tactics.lower() == "by":
            tactics = None

    if tactics:
        if ":= sorry" in tactics or tactics.strip() == "sorry":
            tactics = None
        elif len(tactics.strip()) < 3:
            tactics = None

    return tactics if tactics else "sorry"


def theorem_code_is_commented_out(theorem_code: str) -> bool:
    """True if every non-empty, non-import line is a comment (e.g. minif2f '-- Error: Real.log')."""
    for line in theorem_code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("import "):
            continue
        if stripped.startswith("--"):
            continue
        return False
    return True


# -----------------------------------------------------------------------------
# Lean code assembly: theorem + tactics + header
# -----------------------------------------------------------------------------


def create_full_lean_code(
    theorem_code: str,
    proof_tactics: str,
    header: str,
    default_header: str,
) -> str:
    """Create full Lean 4 code by replacing sorry with proof tactics.

    If model output contains import/open/set_option, use those; else use default_header.
    Handles ":= by sorry", ":= by\n  sorry", and bare "sorry" (prepends "by\n  " when needed).
    """
    model_imports = []
    model_opens = []
    model_set_options = []
    tactics_clean_lines = []

    for line in proof_tactics.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import "):
            model_imports.append(stripped)
        elif stripped.startswith("open "):
            model_opens.append(stripped)
        elif stripped.startswith("set_option "):
            model_set_options.append(stripped)
        elif stripped.startswith("namespace ") or stripped.startswith("section "):
            continue
        else:
            tactics_clean_lines.append(line)

    tactics_clean = "\n".join(tactics_clean_lines).strip()
    if not tactics_clean:
        tactics_clean = "sorry"

    proof_lines = tactics_clean.split("\n")
    indented_proof = "\n  ".join(proof_lines)

    if ":= by sorry" in theorem_code:
        theorem_with_proof = theorem_code.replace(":= by sorry", f":= by\n  {indented_proof}")
    elif ":= by\n  sorry" in theorem_code:
        theorem_with_proof = theorem_code.replace(":= by\n  sorry", f":= by\n  {indented_proof}")
    else:
        last_sorry = theorem_code.rfind("sorry")
        if last_sorry != -1:
            before = theorem_code[:last_sorry]
            after = theorem_code[last_sorry + 5 :]
            if before.rstrip().endswith(":="):
                theorem_with_proof = before.rstrip() + f" by\n  {indented_proof}" + after
            else:
                theorem_with_proof = before + indented_proof + after
        else:
            theorem_with_proof = theorem_code

    if model_imports:
        parts = list(model_imports)
        if model_set_options:
            parts.append("")
            parts.extend(model_set_options)
        if model_opens:
            parts.append("")
            parts.extend(model_opens)
        final_header = "\n".join(parts)
    else:
        final_header = default_header

    return f"{final_header}\n\n{theorem_with_proof}"


# -----------------------------------------------------------------------------
# Persistence: save run (logs, metrics, plots, model) and plot training curves
# -----------------------------------------------------------------------------


def plot_training_curves(metrics: dict, path: Path, title: Optional[str] = None) -> None:
    """Write training_curves.png (loss, grad_norm, entropy, kl_div) to path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not metrics.get("iterations") or len(metrics["iterations"]) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if title:
        fig.suptitle(title, fontsize=12)

    ax1 = axes[0, 0]
    ax1.plot(metrics["iterations"], metrics["losses"], "b-o", linewidth=2, markersize=8)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(metrics["iterations"], metrics["grad_norms"], "r-o", linewidth=2, markersize=8)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Gradient Norm")
    ax2.set_title("Gradient Update Steps")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(metrics["iterations"], metrics["entropies"], "g-o", linewidth=2, markersize=8)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Entropy")
    ax3.set_title("Policy Entropy")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(metrics["iterations"], metrics["kl_divs"], "m-o", linewidth=2, markersize=8)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("KL Divergence")
    ax4.set_title("KL Divergence (Student vs Teacher)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_run(
    output_root: Path,
    config: SDPOConfig,
    logs: dict,
    metrics: dict,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> str:
    """Save logs, metrics, plots, and optionally model/tokenizer to output_root/run_{timestamp}. Returns model_save_dir path."""
    output_dir = output_root / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if model is not None and tokenizer is not None:
        model_save_dir = run_dir / "final_model"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_save_dir)
        tokenizer.save_pretrained(model_save_dir)

    logs_path = run_dir / "logs.json"
    with open(logs_path, "w") as f:
        json.dump(logs, f, indent=2, default=str)

    if len(metrics.get("iterations", [])) > 0:
        plot_training_curves(metrics, run_dir / "training_curves.png", f"SDPO Test-Time RL - {config.model_name}")

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    model_save_dir = run_dir / "final_model"
    return str(model_save_dir)


# -----------------------------------------------------------------------------
# CLI / entrypoint helpers: banners, dataset load, problem summary, persistence
# -----------------------------------------------------------------------------


def print_run_banner(
    model: str,
    gpu: str,
    dataset: str,
    dataset_subset: str,
    dataset_split: str,
    problem_idx: int,
    max_iterations: int,
    feedback_errors_only: bool,
) -> None:
    """Print the SDPO run configuration banner."""
    print("=" * 60)
    print("SDPO Test-Time RL on Modal")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"GPU: {gpu}")
    print(f"Dataset: {dataset}")
    if dataset_subset:
        print(f"Dataset subset: {dataset_subset}")
    print(f"Dataset split: {dataset_split}")
    print(f"Problem index: {problem_idx}")
    print(f"Max iterations: {max_iterations}")
    print(f"Feedback mode: {'errors only' if feedback_errors_only else 'errors + failed proofs'}")
    print("=" * 60)


def load_dataset_with_fallback(
    dataset: str,
    dataset_subset: str,
    dataset_split: str,
) -> Any:
    """Load HuggingFace dataset with fallbacks; print progress and errors. Returns dataset. Raises RuntimeError on failure."""
    from datasets import load_dataset

    print(f"\nLoading dataset {dataset}...")
    try:
        if dataset_subset:
            ds = load_dataset(dataset, dataset_subset, split=dataset_split)
        else:
            ds = load_dataset(dataset, split=dataset_split)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Trying alternative loading...")
        try:
            ds = load_dataset(dataset, split=dataset_split, trust_remote_code=True)
        except Exception as e2:
            print(f"Also failed with trust_remote_code: {e2}")
            try:
                full_ds = load_dataset(dataset, trust_remote_code=True)
                if dataset_split in full_ds:
                    ds = full_ds[dataset_split]
                else:
                    available_splits = list(full_ds.keys())
                    print(f"Split '{dataset_split}' not found. Available: {available_splits}")
                    ds = full_ds[available_splits[0]]
                    print(f"Using split: {available_splits[0]}")
            except Exception as e3:
                raise RuntimeError(f"Could not load dataset {dataset}: {e3}") from e3

    print(f"Dataset loaded with {len(ds)} examples")
    print(f"Available columns: {ds.column_names}")
    return ds


def clamp_problem_idx(problem_idx: int, size: int) -> int:
    """If problem_idx >= size, print warning and return 0; else return problem_idx."""
    if problem_idx >= size:
        print(f"Problem index {problem_idx} out of range (dataset has {size} examples)")
        return 0
    return problem_idx


def print_problem_summary(
    problem: dict,
    problem_idx: int,
    id_fields: list,
    theorem_fields: list,
    informal_fields: list,
    header_fields: list,
) -> None:
    """Print loaded problem ID, Lean4 code preview (or warning), informal and header previews."""
    print(f"\nLoaded problem {problem_idx}:")
    problem_id = get_field(problem, id_fields, f"problem_{problem_idx}")
    print(f"  ID: {problem_id}")
    lean4_code = get_field(problem, theorem_fields)
    if lean4_code:
        print(f"  Lean4 code: {lean4_code[:200]}...")
    else:
        print("  WARNING: No theorem code found! Check dataset field names.")
        print(f"  Available fields: {list(problem.keys())}")
    informal = get_field(problem, informal_fields)
    if informal:
        print(f"  Informal: {informal[:100]}...")
    header = get_field(problem, header_fields)
    if header:
        print(f"  Header: {header[:100]}...")


def print_gpu_unsupported_warning(gpu: str) -> None:
    """Print warning if requested GPU is not supported for SDPO (e.g. A100-80GB, H100)."""
    gpu_upper = gpu.upper()
    if gpu_upper in ["A100-80GB", "A100_80GB", "H100"]:
        print(f"\nWARNING: {gpu} GPU requested but not currently supported for SDPO training.")
        print("    8B models have memory issues (OOM during optimizer step).")
        print("    Falling back to A100-40GB with 1-2B models.\n")


def save_local_run(
    results: dict,
    output_dir_name: str,
    dataset: str,
    problem_idx: int,
) -> Path:
    """Write results to sdpo_results/{output_dir_name}/{dataset_folder}/run_{problem_idx}_{timestamp}/; print paths. Returns run_dir."""
    dataset_folder = dataset.split("/")[-1] if "/" in dataset else dataset
    dataset_folder = dataset_folder.replace("/", "_")
    local_output_dir = Path("sdpo_results") / output_dir_name / dataset_folder
    local_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = local_output_dir / f"run_{problem_idx}_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    local_log_path = run_dir / "logs.json"
    with open(local_log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Local log saved to: {local_log_path}")

    metrics = results.get("metrics", {})
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print("=" * 60)
    return run_dir
