"""
Shared utilities for SDPO: dataset access, full-block parsing, Lean code assembly, and persistence.

TLDR: Single module for (1) get_field/load_problem (dataset-agnostic problem dict access),
(2) theorem_code_is_commented_out (theorem code checks), (3) create_full_lean_code (assemble
full Lean file from theorem + extracted_block + header; uses parsing.extract_full_lean_block
at call sites), (4) KL diagnostics — collect_per_token_kl (returns structured records) and
plot_token_kl_heatmap (renders token text colored by KL value), and (5) plot_training_curves /
save_run / save_local_run (persistence). Parsing lives in sdpo_modal_local_verify_goedel.parsing.
Used by: trainer_core, entrypoint, modal_trainer.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from sdpo_modal_local_verify_goedel.config import SDPOConfig

if TYPE_CHECKING:
    import torch


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
# Theorem code checks
# -----------------------------------------------------------------------------


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
# Lean code assembly: incomplete (sorry) vs complete (full block)
# -----------------------------------------------------------------------------


def create_full_lean_code(
    theorem_code: str,
    extracted_block: str,
    header: str,
    default_header: str,
) -> str:
    """Create full Lean 4 code from parsed Goedel model output.

    Goedel never outputs import clauses, so we always prepend the header.
      - extracted_block == "sorry" (incomplete or no block): return (header or default_header) + theorem_code.
      - Otherwise: always return (default_header or header) + block (never use block as-is).
    """
    if not extracted_block or extracted_block.strip().lower() == "sorry":
        h = (header or default_header or "").strip()
        return f"{h}\n\n{theorem_code}" if h else theorem_code

    block = extracted_block.strip()
    h = (default_header or header or "").strip()
    return f"{h}\n\n{block}" if h else block


# -----------------------------------------------------------------------------
# KL diagnostics: per-token collection and heatmap visualization
# -----------------------------------------------------------------------------


def collect_per_token_kl(
    per_token_kl: "torch.Tensor",
    generated_ids: "torch.Tensor",
    tokenizer,
) -> list[dict]:
    """Decode each generated token and pair it with its KL divergence (student || teacher).

    Does NOT print to stdout — callers are responsible for logging/saving.
    Kept separate from visualization so it can be used in Modal (no display).

    Args:
        per_token_kl: 1-D tensor of shape [seq_len] from compute_sdpo_loss.
        generated_ids: 1-D or 2-D (1, seq_len) token ID tensor.
        tokenizer: HuggingFace tokenizer used to decode token IDs.

    Returns:
        List of dicts: [{"pos": int, "token_id": int, "token": str, "kl": float}, ...]
    """
    ids = generated_ids.view(-1).tolist()
    kl_vals = per_token_kl.detach().float().tolist()

    # Guard: seq lengths must match (mismatch can occur on edge cases)
    n = min(len(ids), len(kl_vals))
    ids, kl_vals = ids[:n], kl_vals[:n]

    return [
        {
            "pos": pos,
            "token_id": token_id,
            "token": tokenizer.decode([token_id], skip_special_tokens=False),
            "kl": kl,
        }
        for pos, (token_id, kl) in enumerate(zip(ids, kl_vals))
    ]


def plot_token_kl_heatmap(
    records: list[dict],
    path: Path,
    iteration: Optional[int] = None,
    max_tokens_per_line: int = 20,
) -> None:
    """Render generated text as a heatmap image: each token is colored by its KL value.

    Color scale: white (KL=0) → deep red (KL=max), using a perceptually uniform colormap.
    A colorbar is drawn on the right. Newline tokens in the generated text start a new row.
    Saved as a PNG to `path`.

    Args:
        records: Output of collect_per_token_kl — list of {pos, token_id, token, kl}.
        path: Destination PNG path.
        iteration: Optional iteration number shown in the title.
        max_tokens_per_line: Soft wrap: start a new row after this many tokens
                             (hard newlines in the token text always start a new row too).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from matplotlib.colorbar import ColorbarBase

    if not records:
        return

    kl_values = [r["kl"] for r in records]
    kl_max = max(kl_values) or 1.0  # avoid division by zero
    cmap = plt.get_cmap("Reds")
    norm = mcolors.Normalize(vmin=0.0, vmax=kl_max)

    # --- Layout: split tokens into rows ---
    # A new row begins at: (a) a hard newline inside a token, or (b) soft wrap limit.
    rows: list[list[dict]] = []
    current_row: list[dict] = []
    for r in records:
        token_text = r["token"]
        if "\n" in token_text:
            # Split on newlines: the part before goes to current row, the rest starts new rows
            parts = token_text.split("\n")
            for i, part in enumerate(parts):
                if i == 0:
                    if part:
                        current_row.append({**r, "token": part})
                    rows.append(current_row)
                    current_row = []
                else:
                    if part:
                        current_row.append({**r, "token": part})
                    if i < len(parts) - 1:
                        rows.append(current_row)
                        current_row = []
        else:
            current_row.append(r)
            if len(current_row) >= max_tokens_per_line:
                rows.append(current_row)
                current_row = []
    if current_row:
        rows.append(current_row)

    # --- Figure sizing ---
    font_size = 9
    cell_h = 0.40          # inches per row
    cell_w_per_char = 0.10 # inches per character (approximate monospace)
    max_row_chars = max(
        (sum(len(r["token"]) for r in row) for row in rows if row),
        default=10,
    )
    fig_w = max(10.0, max_row_chars * cell_w_per_char + 1.5)  # +1.5 for colorbar
    fig_h = max(2.0, len(rows) * cell_h + 1.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    # Main axes for text, narrow axes on the right for colorbar
    ax = fig.add_axes([0.01, 0.08, 0.88, 0.82])
    cax = fig.add_axes([0.91, 0.08, 0.02, 0.82])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = "Per-token KL divergence (student ∥ teacher)"
    if iteration is not None:
        title += f"  —  iteration {iteration}"
    ax.set_title(title, fontsize=10, pad=6)

    n_rows = len(rows)
    for row_idx, row in enumerate(rows):
        if not row:
            continue
        # y position: top row at y≈1, bottom row at y≈0
        y = 1.0 - (row_idx + 0.5) / n_rows

        # Compute x positions proportional to token string length
        total_chars = sum(max(len(r["token"]), 1) for r in row)
        x_cursor = 0.0
        for r in row:
            token_text = r["token"]
            char_frac = max(len(token_text), 1) / total_chars
            x_center = x_cursor + char_frac / 2
            x_cursor += char_frac

            color = cmap(norm(r["kl"]))
            # Background rectangle
            rect = mpatches.FancyBboxPatch(
                (x_center - char_frac / 2, y - 0.4 / n_rows),
                char_frac,
                0.8 / n_rows,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
                transform=ax.transData,
                clip_on=True,
            )
            ax.add_patch(rect)
            # Token text — use dark text on light backgrounds, light on dark
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "black" if brightness > 0.45 else "white"
            # Escape special chars for display
            display = token_text.replace("\t", "→").replace("\r", "↵")
            ax.text(
                x_center, y, display,
                ha="center", va="center",
                fontsize=font_size,
                color=text_color,
                fontfamily="monospace",
                clip_on=True,
            )

    ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical", label="KL divergence")

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def _save_kl_artifacts(run_dir: Path, iteration_logs: list[dict]) -> None:
    """Write per-token KL JSON and heatmap PNG for each iteration that has KL records.

    Files written:
      run_dir/kl/iter_{n}_per_token_kl.json  — raw records list
      run_dir/kl/iter_{n}_kl_heatmap.png     — token heatmap image

    Per-token KL data is kept out of logs.json to avoid bloating the main log.
    """
    kl_dir = run_dir / "kl"
    kl_dir.mkdir(exist_ok=True)
    for iter_log in iteration_logs:
        records = iter_log.get("per_token_kl")
        if not records:
            continue
        n = iter_log.get("iteration", "?")
        kl_json_path = kl_dir / f"iter_{n}_per_token_kl.json"
        with open(kl_json_path, "w") as f:
            json.dump(records, f, indent=2)
        heatmap_path = kl_dir / f"iter_{n}_kl_heatmap.png"
        try:
            plot_token_kl_heatmap(records, heatmap_path, iteration=n)
        except Exception as e:
            print(f"  Warning: could not render KL heatmap for iteration {n}: {e}")


def save_run(
    output_root: Path,
    config: SDPOConfig,
    logs: dict,
    metrics: dict,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> str:
    """Save logs, metrics, plots, KL artifacts, and optionally model/tokenizer.

    Directory layout:
      output_root/{config.output_dir}/run_{timestamp}/
        logs.json            — main log (per_token_kl stripped out to keep size small)
        metrics.json
        training_curves.png
        kl/
          iter_{n}_per_token_kl.json
          iter_{n}_kl_heatmap.png
        final_model/         — only if model+tokenizer provided

    Returns path to final_model dir (may not exist if model was None).
    """
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

    # Strip per_token_kl from iter_logs before writing main log to keep it compact
    iteration_logs = logs.get("iteration_logs", [])
    _save_kl_artifacts(run_dir, iteration_logs)
    slim_logs = {
        **logs,
        "iteration_logs": [{k: v for k, v in il.items() if k != "per_token_kl"} for il in iteration_logs],
    }

    logs_path = run_dir / "logs.json"
    with open(logs_path, "w") as f:
        json.dump(slim_logs, f, indent=2, default=str)

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
    """Write results to sdpo_results/{output_dir_name}/{dataset_folder}/run_{problem_idx}_{timestamp}/.

    Directory layout mirrors save_run:
      run_dir/
        logs.json            — main log (per_token_kl stripped out)
        metrics.json
        kl/
          iter_{n}_per_token_kl.json
          iter_{n}_kl_heatmap.png

    output_dir_name is typically 'local_verify/{model_short}'.
    Returns run_dir.
    """
    dataset_folder = dataset.split("/")[-1] if "/" in dataset else dataset
    dataset_folder = dataset_folder.replace("/", "_")
    local_output_dir = Path("sdpo_results") / Path(output_dir_name) / dataset_folder
    local_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = local_output_dir / f"run_{problem_idx}_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    iteration_logs = results.get("iteration_logs", [])
    _save_kl_artifacts(run_dir, iteration_logs)
    slim_results = {
        **results,
        "iteration_logs": [{k: v for k, v in il.items() if k != "per_token_kl"} for il in iteration_logs],
    }

    local_log_path = run_dir / "logs.json"
    with open(local_log_path, "w") as f:
        json.dump(slim_results, f, indent=2, default=str)
    print(f"Local log saved to: {local_log_path}")

    metrics = results.get("metrics", {})
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print("=" * 60)
    return run_dir
