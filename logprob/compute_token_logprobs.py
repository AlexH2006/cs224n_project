#!/usr/bin/env python3
"""
Score Lean proof tokens with Qwen 3.5 4B and render heatmaps.
Computes per-token log probability only (no KL divergence).

This version is intentionally Qwen 3.5-only:
- no fallback models
- no AutoModelForCausalLM path
- uses the Qwen 3.5 multimodal model loading path

Input format: JSONL, one object per line, e.g.
{"id":"true_intro","context":"import Mathlib\n\ntheorem t : True := by\n","proof":"  trivial\n"}

Required fields: id, context, proof

Example:
  python compute_token_logprobs.py \
      --input proofs.jsonl \
      --outdir qwen35_scores \
      --model Qwen/Qwen3.5-4B

Optional:
  --device cuda
  --max-examples 20
  --fp16
  --bf16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


@dataclass
class TokenScore:
    token_index: int
    token_id: int
    token_text: str
    pretty_token: str
    logprob: float
    prob: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to JSONL file")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="HF model name or local path; intended for Qwen 3.5 4B",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu, mps, auto, or leave unset for auto",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", help="Use float16 on GPU if supported")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 on GPU if supported")
    parser.add_argument(
        "--per-row",
        type=int,
        default=24,
        help="How many proof tokens to show per heatmap row",
    )
    parser.add_argument(
        "--max-label-len",
        type=int,
        default=18,
        help="Max chars to show in each heatmap cell label",
    )
    args = parser.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Choose at most one of --fp16 or --bf16")

    return args


def choose_device(requested: Optional[str]) -> str:
    if requested and requested != "auto":
        if requested not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"Unsupported device: {requested}")
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(args: argparse.Namespace, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def read_jsonl(path: str, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i + 1}: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"Line {i + 1}: expected JSON object, got {type(obj)}")

            for key in ("id", "context", "proof"):
                if key not in obj:
                    raise ValueError(
                        f"Line {i + 1}: missing required field '{key}'. "
                        f"Required: id, context, proof"
                    )
                if not isinstance(obj[key], str):
                    raise ValueError(
                        f"Line {i + 1}: field '{key}' must be a string, got {type(obj[key])}"
                    )

            records.append(obj)

            if max_examples is not None and len(records) >= max_examples:
                break

    return records


def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:180] if s else "example"


def escape_visible_whitespace(s: str) -> str:
    return s.replace(" ", "·").replace("\n", "↵").replace("\t", "⇥")


def truncate_middle(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    if max_len <= 3:
        return s[:max_len]
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return s[:left] + "..." + s[-right:]


def pretty_token_text(token_text: str, max_len: int) -> str:
    t = escape_visible_whitespace(token_text)
    t = truncate_middle(t, max_len)
    return t if t else "∅"


def load_qwen35_model_and_processor(
    model_name: str,
    dtype: torch.dtype,
    device: str,
):
    """
    Load Qwen 3.5 4B using the multimodal image-text-to-text path.

    No fallback is used.
    """
    print(f"[info] Loading Qwen 3.5 model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=dtype,
    )
    model = model.to(device)
    model.eval()

    return model, processor


def get_text_token_ids_from_processor(processor: Any, text: str) -> List[int]:
    """
    Extract text token ids using the processor's tokenizer component.
    """
    tokenizer = processor.tokenizer
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return enc.input_ids[0].tolist()


def build_text_only_inputs(processor: Any, text: str, device: str) -> Dict[str, torch.Tensor]:
    """
    Build text-only model inputs through the processor/tokenizer stack.

    For Qwen 3.5 we avoid image inputs and just tokenize the raw text.
    """
    tokenizer = processor.tokenizer
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )

    model_inputs: Dict[str, torch.Tensor] = {}
    for k, v in enc.items():
        if isinstance(v, torch.Tensor):
            model_inputs[k] = v.to(device)

    return model_inputs


def extract_logits(outputs: Any) -> torch.Tensor:
    """
    Be tolerant to output object style.
    """
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    raise ValueError("Model output does not contain logits")


def score_proof_tokens(
    model: Any,
    processor: Any,
    context: str,
    proof: str,
    device: str,
    max_label_len: int = 18,
) -> List[TokenScore]:
    """
    Compute per-token logprob for proof tokens, conditioned on full [context + proof_prefix].

    Proof token boundaries are determined by token position, not character offsets.
    """
    full_text = context + proof

    tokenizer = processor.tokenizer

    full_token_ids = get_text_token_ids_from_processor(processor, full_text)
    context_token_ids = get_text_token_ids_from_processor(processor, context)

    proof_start_pos = len(context_token_ids)

    if len(full_token_ids) == 0:
        return []

    inputs = build_text_only_inputs(processor, full_text, device=device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = extract_logits(outputs)  # [1, seq_len, vocab]

    if logits.shape[1] < 2:
        return []

    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = inputs["input_ids"][:, 1:]
    gathered = logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)[0]

    scores: List[TokenScore] = []

    input_ids_1d = inputs["input_ids"][0]

    for pos in range(proof_start_pos, input_ids_1d.shape[0]):
        if pos == 0:
            continue

        token_id = int(input_ids_1d[pos].item())
        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

        lp = float(gathered[pos - 1].item())
        p = float(math.exp(lp))

        scores.append(
            TokenScore(
                token_index=len(scores),
                token_id=token_id,
                token_text=token_text,
                pretty_token=pretty_token_text(token_text, max_label_len),
                logprob=lp,
                prob=p,
            )
        )

    return scores


def save_csv(scores: List[TokenScore], csv_path: str) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["token_index", "token_id", "token_text", "pretty_token", "logprob", "prob"]
        )
        for s in scores:
            writer.writerow(
                [s.token_index, s.token_id, s.token_text, s.pretty_token, s.logprob, s.prob]
            )


def render_heatmap(
    scores: List[TokenScore],
    title: str,
    out_path: str,
    per_row: int = 24,
    max_label_len: int = 18,
) -> None:
    if not scores:
        return

    labels = [pretty_token_text(s.token_text, max_label_len) for s in scores]
    values = np.array([s.logprob for s in scores], dtype=np.float32)

    n = len(scores)
    nrows = math.ceil(n / per_row)
    ncols = per_row

    grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    annot = np.full((nrows, ncols), "", dtype=object)

    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx < n:
                grid[r, c] = values[idx]
                annot[r, c] = f"{labels[idx]}\n{values[idx]:.2f}"
                idx += 1

    fig_w = max(12, ncols * 0.85)
    fig_h = max(2.8, nrows * 1.25)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="white")

    im = ax.imshow(grid, aspect="auto", cmap=cmap)

    ax.set_title(title, pad=16)
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))
    ax.set_xticklabels([str(i) for i in range(ncols)])
    ax.set_yticklabels([str(i) for i in range(nrows)])
    ax.set_xlabel("Token column")
    ax.set_ylabel("Row")

    for r in range(nrows):
        for c in range(ncols):
            if not np.isnan(grid[r, c]):
                ax.text(
                    c,
                    r,
                    annot[r, c],
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Log probability")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_barcode(
    scores: List[TokenScore],
    title: str,
    out_path: str,
) -> None:
    if not scores:
        return

    values = np.array([s.logprob for s in scores], dtype=np.float32)[None, :]
    fig_w = max(12, len(scores) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, 2.3))
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn")
    ax.set_title(title, pad=12)
    ax.set_yticks([])
    ax.set_xlabel("Proof token index")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Log probability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = choose_device(args.device)
    dtype = choose_dtype(args, device)

    print(f"[info] Requested model: {args.model}")
    print(f"[info] Device: {device}, dtype: {dtype}")

    model, processor = load_qwen35_model_and_processor(
        model_name=args.model,
        dtype=dtype,
        device=device,
    )

    records = read_jsonl(args.input, args.max_examples)
    print(f"[info] Loaded {len(records)} examples")

    summary_rows: List[Dict[str, Any]] = []

    for i, rec in enumerate(records):
        ex_id = rec["id"]
        context = rec["context"]
        proof = rec["proof"]

        print(f"[info] Scoring {i + 1}/{len(records)}: {ex_id}")

        scores = score_proof_tokens(
            model=model,
            processor=processor,
            context=context,
            proof=proof,
            device=device,
            max_label_len=args.max_label_len,
        )

        safe_id = sanitize_filename(ex_id)
        ex_dir = os.path.join(args.outdir, safe_id)
        os.makedirs(ex_dir, exist_ok=True)

        csv_path = os.path.join(ex_dir, "token_scores.csv")
        heatmap_path = os.path.join(ex_dir, "heatmap.png")
        barcode_path = os.path.join(ex_dir, "barcode.png")
        text_path = os.path.join(ex_dir, "example.txt")

        save_csv(scores, csv_path)

        if scores:
            render_heatmap(
                scores=scores,
                title=f"{ex_id} — {args.model} proof token logprobs",
                out_path=heatmap_path,
                per_row=args.per_row,
                max_label_len=args.max_label_len,
            )
            render_barcode(
                scores=scores,
                title=f"{ex_id} — barcode view",
                out_path=barcode_path,
            )
        else:
            print(f"[warn] No proof tokens scored for {ex_id}; skipping plots")
            heatmap_path = ""
            barcode_path = ""

        with open(text_path, "w", encoding="utf-8") as f:
            f.write("=== CONTEXT ===\n")
            f.write(context)
            f.write("\n=== PROOF ===\n")
            f.write(proof)

        if scores:
            logprob_values = np.array([s.logprob for s in scores], dtype=np.float32)
            avg_lp = float(np.mean(logprob_values))
            min_lp = float(np.min(logprob_values))
            max_lp = float(np.max(logprob_values))
        else:
            avg_lp = float("nan")
            min_lp = float("nan")
            max_lp = float("nan")

        summary_rows.append(
            {
                "id": ex_id,
                "num_proof_tokens": len(scores),
                "avg_logprob": avg_lp,
                "min_logprob": min_lp,
                "max_logprob": max_lp,
                "csv": csv_path,
                "heatmap": heatmap_path,
                "barcode": barcode_path,
            }
        )

    summary_csv = os.path.join(args.outdir, "summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "num_proof_tokens",
                "avg_logprob",
                "min_logprob",
                "max_logprob",
                "csv",
                "heatmap",
                "barcode",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[done] Wrote outputs to: {args.outdir}")
    print(f"[done] Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()