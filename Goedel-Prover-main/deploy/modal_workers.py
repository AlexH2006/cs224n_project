import hashlib
import os
from typing import Optional

import modal

APP_NAME = "goedel-prover-modal-workers"
FUNCTION_NAME = "generate_one_attempt"
DEFAULT_MODEL_ID = "Goedel-LM/Goedel-Prover-SFT"
DEFAULT_GPU = "A10G"
HF_CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "transformers", "accelerate")
)

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name("goedel-prover-hf-cache", create_if_missing=True)

_tokenizer = None
_model = None
_loaded_model_id = None


def _load_model(model_id: str):
    global _tokenizer, _model, _loaded_model_id
    if _tokenizer is not None and _model is not None and _loaded_model_id == model_id:
        return _tokenizer, _model

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_CACHE_DIR)
    hf_token = os.environ.get("HF_TOKEN")
    _tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _model.eval()
    _loaded_model_id = model_id
    return _tokenizer, _model


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 20,
    scaledown_window=60 * 10,
    startup_timeout=60 * 10,
    volumes={HF_CACHE_DIR: hf_cache_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate_one_attempt(job: dict):
    import torch

    prompt = job["prompt"]
    prompt_idx = int(job["prompt_idx"])
    attempt_idx = int(job["attempt_idx"])

    model_name: Optional[str] = job.get("model_name") or os.environ.get("MODEL_ID") or DEFAULT_MODEL_ID
    temperature: float = float(job.get("temperature", 1.0))
    top_p: float = float(job.get("top_p", 0.95))
    max_tokens: int = int(job.get("max_tokens", 2048))

    seed_input = f"{prompt_idx}:{attempt_idx}:{model_name}"
    seed = int.from_bytes(hashlib.sha256(seed_input.encode("utf-8")).digest()[:4], "big")

    tokenizer, model = _load_model(model_name)
    device = model.device

    with torch.inference_mode():
        torch.manual_seed(seed)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = encoded["input_ids"].shape[-1]
    completion_ids = generated[0][prompt_len:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "prompt_idx": prompt_idx,
        "attempt_idx": attempt_idx,
        "text": completion_text,
    }
