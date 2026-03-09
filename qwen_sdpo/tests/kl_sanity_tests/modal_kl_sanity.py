"""
TLDR: Modal app for KL sanity check — runs the three-step pipeline on GPU.

Step 1: Forward pass with student context (student_prompt + generated_solution)
Step 2: Forward pass with teacher context (teacher_prompt + generated_solution)
Step 3: Calculate KL divergence + save run to volume

Supported models (shorthand): 4B, 9B (Qwen3.5), goedel8b (Goedel-Prover-V2-8B).

All code self-contained under kl_sanity_tests/. No changes to modal_app.py or
other qwen_sdpo modules.

Usage:
  modal run qwen_sdpo/tests/kl_sanity_tests/modal_kl_sanity.py \
    --input-path qwen_sdpo/tests/kl_sanity_tests/prompts.json

  Results are automatically downloaded to qwen_sdpo/tests/kl_sanity_tests/run_YYYYMMDD_HHMMSS/.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None

if modal is None:
    raise ImportError("modal is required. Install with: pip install modal")

# -----------------------------------------------------------------------------
# App, image, volumes
# -----------------------------------------------------------------------------

app = modal.App("kl-sanity")

# Lighter image: no vLLM, only HF + bitsandbytes for forward passes
kl_sanity_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "transformers>=5.2.0",
        "bitsandbytes",
        "accelerate",
        "matplotlib",
    )
    .add_local_python_source("qwen_sdpo")
)

hf_cache_volume = modal.Volume.from_name("kl-sanity-hf-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("kl-sanity-output", create_if_missing=True)

# -----------------------------------------------------------------------------
# Remote function: three-step pipeline
# -----------------------------------------------------------------------------


@app.function(
    image=kl_sanity_image,
    gpu="A100",
    volumes={
        "/hf_cache": hf_cache_volume,
        "/output": output_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=600,
)
def run_kl_sanity_remote(
    prompts: dict,
    model_name: str = "Qwen/Qwen3.5-4B",
    distillation_topk: int = 20,
) -> dict:
    """Run KL sanity pipeline on GPU: student forward, teacher forward, KL + save.

    Args:
        prompts: Dict with student_prompt, teacher_prompt, generated_solution.
        model_name: HuggingFace model name.
        distillation_topk: Top-K for KL computation.

    Returns:
        {"run_dir": "run_YYYYMMDD_HHMMSS", "volume_name": "kl-sanity-output"}
    """
    import os

    os.environ["HF_HOME"] = "/hf_cache"

    from qwen_sdpo.results import collect_per_token_kl, plot_token_kl_heatmap
    from qwen_sdpo.tests.kl_sanity_tests.run_kl_sanity import (
        _compute_kl_no_grad,
        _load_model_and_tokenizer,
    )

    student_prompt = prompts["student_prompt"]
    teacher_prompt = prompts["teacher_prompt"]
    generated_solution = prompts["generated_solution"]

    # Load model and tokenizer
    model, tokenizer = _load_model_and_tokenizer(model_name)

    # Tokenize solution (add_special_tokens=False: continuation)
    gen_ids = tokenizer(
        generated_solution,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    if gen_ids.numel() == 0:
        raise ValueError(
            "generated_solution tokenizes to empty sequence; cannot compute KL"
        )

    # Steps 1+2: Forward passes (inside _compute_kl_no_grad)
    # Step 3: KL computation
    per_token_kl = _compute_kl_no_grad(
        model=model,
        tokenizer=tokenizer,
        distillation_topk=distillation_topk,
        student_prompt=student_prompt,
        teacher_prompt=teacher_prompt,
        generated_ids=gen_ids,
    )

    records = collect_per_token_kl(per_token_kl, gen_ids, tokenizer)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"run_{timestamp}"
    run_dir = Path("/output") / run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts (include model for run provenance)
    with open(run_dir / "context.json", "w") as f:
        json.dump(
            {
                "model": model_name,
                "student_prompt": student_prompt,
                "teacher_prompt": teacher_prompt,
                "generated_solution": generated_solution,
            },
            f,
            indent=2,
        )

    with open(run_dir / "per_token_kl.json", "w") as f:
        json.dump(records, f, indent=2)

    plot_token_kl_heatmap(records, run_dir / "kl_heatmap.png")

    # Persist volume (Modal auto-commits on success, but explicit for clarity)
    output_volume.commit()

    return {"run_dir": run_dir_name, "volume_name": "kl-sanity-output"}


# -----------------------------------------------------------------------------
# Local entrypoint
# -----------------------------------------------------------------------------

_REQUIRED_KEYS = frozenset({"student_prompt", "teacher_prompt", "generated_solution"})

_MODEL_ALIASES = {
    "4B": "Qwen/Qwen3.5-4B",
    "9B": "Qwen/Qwen3.5-9B",
    "goedel8b": "Goedel-LM/Goedel-Prover-V2-8B",
}


def _resolve_model(name: str) -> str:
    """Resolve shorthand (4B, 9B, goedel8b) to full HuggingFace model path."""
    return _MODEL_ALIASES.get(name, name)


@app.local_entrypoint()
def main(
    input_path: str,
    model: str | None = None,
    distillation_topk: int = 20,
):
    """Run KL sanity check on Modal. Reads prompts from JSON.

    Args:
        input_path: Path to JSON with student_prompt, teacher_prompt, generated_solution.
        model: HuggingFace model name or shorthand (4B, 9B, goedel8b). If omitted, uses "model" from JSON or 4B.
        distillation_topk: Top-K for KL computation.

    Config: Add "model": "9B" to the JSON to use Qwen3.5-9B; --model overrides.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys in {path}: {sorted(missing)}")

    prompts = {
        "student_prompt": str(data["student_prompt"]),
        "teacher_prompt": str(data["teacher_prompt"]),
        "generated_solution": str(data["generated_solution"]),
    }

    # Model: --model CLI overrides; else use "model" from JSON; else default 4B
    model_resolved = _resolve_model(
        model if model is not None else data.get("model", "Qwen/Qwen3.5-4B")
    )

    result = run_kl_sanity_remote.remote(
        prompts=prompts,
        model_name=model_resolved,
        distillation_topk=distillation_topk,
    )

    run_dir = result["run_dir"]
    vol_name = result["volume_name"]

    # Download results via modal volume get (reload/read_file require a running function)
    local_base = Path(__file__).resolve().parent
    subprocess.run(
        ["modal", "volume", "get", vol_name, run_dir, str(local_base)],
        check=True,
    )
    local_run_dir = local_base / run_dir
    print(f"Run saved to {local_run_dir}/")
