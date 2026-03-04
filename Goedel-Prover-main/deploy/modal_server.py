import os
from typing import List, Optional

import modal

APP_NAME = "goedel-prover-modal-hf"
DEFAULT_MODEL_ID = "Goedel-LM/Goedel-Prover-SFT"
DEFAULT_GPU = "A10G"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "transformers", "accelerate", "fastapi>=0.110,<1.0")
)

app = modal.App(APP_NAME)

_tokenizer = None
_model = None
_loaded_model_id = None


def _load_model(model_id: str):
    global _tokenizer, _model, _loaded_model_id
    if _tokenizer is not None and _model is not None and _loaded_model_id == model_id:
        return _tokenizer, _model

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
)
@modal.fastapi_endpoint(method="POST")
def generate(payload: dict):
    import torch

    prompts: List[str] = payload.get("prompts", [])
    if not isinstance(prompts, list) or any(not isinstance(p, str) for p in prompts):
        return {"error": "payload.prompts must be a list of strings"}

    n: int = int(payload.get("n", 1))
    temperature: float = float(payload.get("temperature", 1.0))
    top_p: float = float(payload.get("top_p", 0.95))
    max_tokens: int = int(payload.get("max_tokens", 2048))
    model_name: Optional[str] = payload.get("model_name") or os.environ.get("MODEL_ID") or DEFAULT_MODEL_ID

    tokenizer, model = _load_model(model_name)
    device = model.device
    outputs: List[List[str]] = []

    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = encoded["input_ids"].shape[-1]
            completions = []
            for sequence in generated:
                completion_ids = sequence[prompt_len:]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                completions.append(completion_text)
            outputs.append(completions)

    return {"outputs": outputs}
