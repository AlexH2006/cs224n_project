import os
import time

import modal

APP_NAME = "goedel-prover-modal-server"

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
        "fastapi[standard]",
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
    timeout=3600,
    volumes={"/cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface")],
)
@modal.fastapi_endpoint(method="POST")
def generate(body: dict) -> dict:
    """HTTP endpoint: generate completions for a batch of prompts.

    Request body:
        prompts (list[str]): list of prompt strings
        n (int): number of completions per prompt (default 1)
        model_name (str, optional): HF model name
        temperature (float): default 1.0
        top_p (float): default 0.95
        max_tokens (int): default 2048

    Response:
        outputs (list[list[str]]): one list of n completion strings per prompt
    """
    import torch

    prompts = body["prompts"]
    n = body.get("n", 1)
    model_name = body.get("model_name", "Goedel-LM/Goedel-Prover-SFT")
    temperature = body.get("temperature", 1.0)
    top_p = body.get("top_p", 0.95)
    max_tokens = body.get("max_tokens", 2048)

    tokenizer, model = _load_model(model_name)

    all_outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        texts = []
        for _ in range(n):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
            generated = tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True
            )
            texts.append(generated)
        all_outputs.append(texts)

    return {"outputs": all_outputs}
