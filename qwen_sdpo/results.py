"""
TLDR: Persistence layer for SDPO runs — logs, metrics, KL artifacts, training curves.

Output layout (local mirror):
  sdpo_results/
    Qwen3.5-9B/
      full_output/      # teacher_response_mode="full_output"
        run_Qwen3.5-9B_{problem_idx}_{timestamp}/
      answer_only/      # teacher_response_mode="answer_only"
        ...
      code_only/        # teacher_response_mode="code_only"
        run_Qwen3.5-9B_{problem_idx}_{timestamp}/
        logs.json               — per-iteration log (per_token_kl stripped to keep size small)
        metrics.json            — scalar metrics timeseries
        hyperparameters.json    — SDPOConfig used for this run
        training_curves.png     — 2x2 matplotlib figure (loss, grad_norm, entropy, kl_div)
        kl/
          iter_{n}_per_token_kl.json   — raw per-token KL records
          iter_{n}_kl_heatmap.png      — token heatmap colored by KL value
        final_model/            — saved LoRA adapter (if model+tokenizer provided)
    Qwen3.5-4B/
      ...

Public API:
  make_run_dir(cfg, problem_idx)          → (model_tag, run_dir: Path)
  save_run(run_dir, cfg, logs, metrics)  → saves to run_dir (Modal volume or local)
  save_local_run(results, cfg, problem_idx)  → mirrors to sdpo_results/{model_tag}/...
  plot_training_curves(metrics, path)    → 2x2 PNG

KL helpers (used by save_run internally):
  collect_per_token_kl(per_token_kl, generated_ids, tokenizer) → list[dict]
  plot_token_kl_heatmap(records, path)   → PNG

Used by: modal_trainer.py (save_run on volume), entrypoint.py (save_local_run).
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from qwen_sdpo.config import SDPOConfig

if TYPE_CHECKING:
    import torch


# -----------------------------------------------------------------------------
# Run directory helpers
# -----------------------------------------------------------------------------


def make_run_dir(cfg: SDPOConfig, problem_idx: int) -> tuple[str, Path]:
    """Derive model_tag and run directory path from config.

    Output is sdpo_results/{model_tag}/{full_output|answer_only|code_only}/run_... so runs
    are separated by teacher_response_mode.

    Returns:
        (model_tag, run_dir) where model_tag is e.g. "Qwen3.5-9B" and
        run_dir is sdpo_results/Qwen3.5-9B/{full_output|answer_only|code_only}/run_.../
        The directory is NOT created here — callers call run_dir.mkdir(...).
    """
    model_tag = cfg.model_name.split("/")[-1]  # "Qwen/Qwen3.5-9B" → "Qwen3.5-9B"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_dir = cfg.teacher_response_mode
    parent = Path(cfg.results_base_dir) / model_tag / mode_dir
    run_dir = parent / f"run_{model_tag}_{problem_idx}_{timestamp}"
    return model_tag, run_dir


# -----------------------------------------------------------------------------
# KL diagnostics: per-token collection and heatmap visualization
# -----------------------------------------------------------------------------


def collect_per_token_kl(
    per_token_kl: "torch.Tensor",
    generated_ids: "torch.Tensor",
    tokenizer,
) -> list[dict]:
    """Pair each generated token with its KL divergence (student || teacher).

    Args:
        per_token_kl:  1-D [seq_len] tensor from compute_sdpo_loss.
        generated_ids: 1-D or 2-D (1, seq_len) token ID tensor.
        tokenizer:     HuggingFace tokenizer used to decode token IDs.

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
    sample_index: Optional[int] = None,
    max_tokens_per_line: int = 20,
) -> None:
    """Render generated text as a heatmap: each token colored by KL value.

    Color scale: white (KL=0) → deep red (KL=max). A colorbar is drawn on
    the right. Newline tokens start a new row. Saved as PNG to `path`.

    Args:
        records:             Output of collect_per_token_kl.
        path:                Destination PNG path.
        iteration:           Optional iteration number shown in title.
        sample_index:        Optional sample index (minibatch) shown in title.
        max_tokens_per_line: Soft wrap column limit.
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
    kl_max = max(kl_values) or 1.0
    cmap = plt.get_cmap("Reds")
    norm = mcolors.Normalize(vmin=0.0, vmax=kl_max)

    rows: list[list[dict]] = []
    current_row: list[dict] = []
    for r in records:
        token_text = r["token"]
        if "\n" in token_text:
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

    font_size = 9
    cell_h = 0.40
    cell_w_per_char = 0.10
    max_row_chars = max(
        (sum(len(r["token"]) for r in row) for row in rows if row),
        default=10,
    )
    fig_w = max(10.0, max_row_chars * cell_w_per_char + 1.5)
    fig_h = max(2.0, len(rows) * cell_h + 1.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.01, 0.08, 0.88, 0.82])
    cax = fig.add_axes([0.91, 0.08, 0.02, 0.82])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = "Per-token KL divergence (student ∥ teacher)"
    if iteration is not None:
        title += f"  —  iteration {iteration}"
    if sample_index is not None:
        title += f"  sample {sample_index}"
    ax.set_title(title, fontsize=10, pad=6)

    n_rows = len(rows)
    for row_idx, row in enumerate(rows):
        if not row:
            continue
        y = 1.0 - (row_idx + 0.5) / n_rows
        total_chars = sum(max(len(r["token"]), 1) for r in row)
        x_cursor = 0.0
        for r in row:
            token_text = r["token"]
            char_frac = max(len(token_text), 1) / total_chars
            x_center = x_cursor + char_frac / 2
            x_cursor += char_frac

            color = cmap(norm(r["kl"]))
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
            brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "black" if brightness > 0.45 else "white"
            display = token_text.replace("\t", "→").replace("\r", "↵").replace("$", r"\$")
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
# Training curves plot
# -----------------------------------------------------------------------------


def plot_training_curves(metrics: dict, path: Path, title: Optional[str] = None) -> None:
    """Write training_curves.png: 2x2 grid of (loss, grad_norm, entropy, kl_div)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not metrics.get("iterations") or len(metrics["iterations"]) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if title:
        fig.suptitle(title, fontsize=12)

    axes[0, 0].plot(metrics["iterations"], metrics["losses"], "b-o", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curve")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(metrics["iterations"], metrics["grad_norms"], "r-o", linewidth=2, markersize=8)
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Gradient Norm")
    axes[0, 1].set_title("Gradient Update Steps")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(metrics["iterations"], metrics["entropies"], "g-o", linewidth=2, markersize=8)
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Entropy")
    axes[1, 0].set_title("Policy Entropy")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(metrics["iterations"], metrics["kl_divs"], "m-o", linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("KL Divergence")
    axes[1, 1].set_title("KL Divergence (Student vs Teacher)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Internal KL artifact writer
# -----------------------------------------------------------------------------


def _save_kl_artifacts(run_dir: Path, iteration_logs: list[dict]) -> None:
    """Write per-token KL JSON and heatmap PNG for each iteration (and each sample in minibatch).

    Single-sample (no "samples" key): writes iter_{n}_per_token_kl.json and iter_{n}_kl_heatmap.png.
    Minibatch (iter_log has "samples"): for each sample that has per_token_kl, writes
      iter_{n}_sample_{s}_per_token_kl.json and iter_{n}_sample_{s}_kl_heatmap.png.
    """
    kl_dir = run_dir / "kl"
    kl_dir.mkdir(exist_ok=True)
    for iter_log in iteration_logs:
        n = iter_log.get("iteration", "?")
        samples = iter_log.get("samples")
        if samples is not None:
            # Minibatch: one KL artifact per sample that has per_token_kl.
            for s_idx, sample in enumerate(samples):
                records = sample.get("per_token_kl")
                if not records:
                    continue
                kl_json_path = kl_dir / f"iter_{n}_sample_{s_idx}_per_token_kl.json"
                with open(kl_json_path, "w") as f:
                    json.dump(records, f, indent=2)
                heatmap_path = kl_dir / f"iter_{n}_sample_{s_idx}_kl_heatmap.png"
                try:
                    plot_token_kl_heatmap(
                        records, heatmap_path, iteration=n, sample_index=s_idx
                    )
                except Exception as e:
                    print(
                        f"  Warning: could not render KL heatmap for iter {n} sample {s_idx}: {e}"
                    )
            continue
        # Single-sample (legacy or minibatch_size=1 without "samples").
        records = iter_log.get("per_token_kl")
        if not records:
            continue
        kl_json_path = kl_dir / f"iter_{n}_per_token_kl.json"
        with open(kl_json_path, "w") as f:
            json.dump(records, f, indent=2)
        heatmap_path = kl_dir / f"iter_{n}_kl_heatmap.png"
        try:
            plot_token_kl_heatmap(records, heatmap_path, iteration=n)
        except Exception as e:
            print(f"  Warning: could not render KL heatmap for iteration {n}: {e}")


# -----------------------------------------------------------------------------
# save_run: write all artifacts to a directory (Modal volume or local)
# -----------------------------------------------------------------------------


def save_run(
    run_dir: Path,
    cfg: SDPOConfig,
    logs: dict,
    metrics: dict,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    save_kl: bool = True,
) -> Path:
    """Save logs, metrics, plots, KL artifacts, and optionally LoRA model to run_dir.

    The caller is responsible for creating run_dir before calling this function.
    Used both on the Modal volume (by modal_trainer.finalize_run) and locally.

    Args:
        run_dir:   Destination directory (must exist).
        cfg:       SDPOConfig (used for plot title).
        logs:      Dict with "iteration_logs" list and top-level metadata.
        metrics:   Dict with "iterations", "losses", "grad_norms", etc.
        model:     Optional PEFT model to save as final_model/. None = skip.
        tokenizer: Optional tokenizer to save alongside model.
        save_kl:   If True (default), write per-token KL JSON + heatmap PNGs to
                   kl/ subdirectory. Set False on the Modal volume side to skip
                   KL artifacts there (they are generated locally instead, using
                   the unstripped logs dict that still has per_token_kl records).

    Returns:
        run_dir (for chaining).
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    if model is not None and tokenizer is not None:
        model_save_dir = run_dir / "final_model"
        model_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_save_dir)
        tokenizer.save_pretrained(model_save_dir)

    # Strip per_token_kl from iteration_logs to keep logs.json compact.
    # Raw KL data lives in kl/iter_{n}_*.json (written above). Minibatch entries
    # have "samples"; strip per_token_kl from each sample as well.
    iteration_logs = logs.get("iteration_logs", [])
    if save_kl:
        _save_kl_artifacts(run_dir, iteration_logs)
    slim_iteration_logs = []
    for il in iteration_logs:
        slim_il = {k: v for k, v in il.items() if k != "per_token_kl"}
        if "samples" in slim_il:
            slim_il["samples"] = [
                {k: v for k, v in s.items() if k != "per_token_kl"}
                for s in slim_il["samples"]
            ]
        slim_iteration_logs.append(slim_il)
    slim_logs = {**logs, "iteration_logs": slim_iteration_logs}

    with open(run_dir / "logs.json", "w") as f:
        json.dump(slim_logs, f, indent=2, default=str)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    hyperparams = asdict(cfg)
    with open(run_dir / "hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=2, default=str)

    if metrics.get("iterations"):
        model_tag = cfg.model_name.split("/")[-1]
        plot_training_curves(
            metrics,
            run_dir / "training_curves.png",
            title=f"SDPO — {model_tag}",
        )

    return run_dir


# -----------------------------------------------------------------------------
# save_local_run: mirror results from Modal volume to local disk
# -----------------------------------------------------------------------------


def save_local_run(
    results: dict,
    cfg: SDPOConfig,
    problem_idx: int,
) -> Path:
    """Mirror SDPO run results to local sdpo_results/{model_tag}/{run_dir}/.

    Call this after trainer.finalize_run.remote() returns results.
    Saves logs.json, metrics.json, KL artifacts, and training curves locally.

    Args:
        results:     Dict returned by finalize_run (contains iteration_logs, metrics, etc.).
        cfg:         SDPOConfig (for model_name and results_base_dir).
        problem_idx: Problem index in the dataset (used for run dir naming).

    Returns:
        Local run_dir Path.
    """
    if cfg.batch_run_dir:
        run_dir = Path(cfg.batch_run_dir) / "runs" / f"problem_{problem_idx}"
        model_tag = cfg.model_name.split("/")[-1]
    else:
        model_tag, run_dir = make_run_dir(cfg, problem_idx)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = results.get("metrics", {})
    save_run(run_dir, cfg, results, metrics)

    if metrics.get("iterations"):
        plot_training_curves(
            metrics,
            run_dir / "training_curves.png",
            title=f"SDPO — {model_tag} — idx={problem_idx}",
        )

    print(f"Local run saved to: {run_dir}")
    return run_dir
