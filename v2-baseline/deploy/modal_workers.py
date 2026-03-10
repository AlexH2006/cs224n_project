import os
import time

import modal

APP_NAME = "goedel-prover-modal-workers"

app = modal.App(APP_NAME)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
    )
)

MODEL_CACHE = {}


def _load_model(model_name: str):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="/cache/huggingface",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/cache/huggingface",
    )
    model.eval()
    MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    volumes={"/cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate_one_attempt(job: dict) -> dict:
    """Generate a single completion for a single prompt.

    job keys:
        prompt (str): the prompt text
        prompt_idx (int): index of the prompt in the batch
        attempt_idx (int): which attempt this is for the prompt
        model_name (str, optional): HF model name (default Goedel-LM/Goedel-Prover-SFT)
        temperature (float): sampling temperature
        top_p (float): nucleus sampling p
        max_tokens (int): max new tokens to generate

    Returns dict with: prompt_idx, attempt_idx, text, seconds_used
    """
    import torch

    model_name = job.get("model_name", "Goedel-LM/Goedel-Prover-SFT")
    tokenizer, model = _load_model(model_name)

    t0 = time.time()

    inputs = tokenizer(job["prompt"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=job.get("max_tokens", 2048),
            temperature=job.get("temperature", 1.0),
            top_p=job.get("top_p", 0.95),
            do_sample=True,
        )
    prompt_len = inputs["input_ids"].shape[1]
    generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

    return {
        "prompt_idx": job["prompt_idx"],
        "attempt_idx": job["attempt_idx"],
        "text": generated,
        "seconds_used": time.time() - t0,
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={"/cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
def generate_n_for_prompt(job: dict) -> dict:
    """Generate N completions for a single prompt.

    job keys:
        prompt (str): the prompt text
        prompt_idx (int): index of the prompt in the batch
        n (int): number of completions to generate
        model_name (str, optional): HF model name
        temperature (float): sampling temperature
        top_p (float): nucleus sampling p
        max_tokens (int): max new tokens to generate

    Returns dict with: prompt_idx, texts (list[str]), seconds_used
    """
    import torch

    model_name = job.get("model_name", "Goedel-LM/Goedel-Prover-SFT")
    tokenizer, model = _load_model(model_name)

    n = job.get("n", 1)
    t0 = time.time()
    texts = []

    inputs = tokenizer(job["prompt"], return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    for _ in range(n):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=job.get("max_tokens", 2048),
                temperature=job.get("temperature", 1.0),
                top_p=job.get("top_p", 0.95),
                do_sample=True,
            )
        generated = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        texts.append(generated)

    return {
        "prompt_idx": job["prompt_idx"],
        "texts": texts,
        "seconds_used": time.time() - t0,
    }
