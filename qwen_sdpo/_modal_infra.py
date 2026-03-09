"""
TLDR: Modal infrastructure (app, image, volumes) shared by modal_trainer.py and modal_app.py.

Kept in a separate file to break the circular import:
  modal_app.py  → imports QwenSDPOTrainer from modal_trainer.py
  modal_trainer.py → imports app/image/volumes from here (not from modal_app.py)

Volumes:
  hf_cache_volume      → /hf_cache    — model weights, downloaded once and reused
  output_volume        → /output      — training results (logs, metrics, saved model)
  compile_cache_volume → /root/.cache/vllm/torch_compile_cache
                         Persists vLLM's torch.compile artifacts across container restarts.
                         First cold start: ~220s (compile + CUDA graph capture).
                         Subsequent cold starts: ~11s (CUDA graph capture only — compile
                         artifacts are read from this volume, skipping recompilation).
"""

try:
    import modal
except ImportError:
    modal = None

if modal is None:
    app = None
    inference_image = None
    hf_cache_volume = None
    output_volume = None
    compile_cache_volume = None

else:
    app = modal.App("qwen-sdpo")

    # Shared with qwen_eval so model weights are only downloaded once.
    hf_cache_volume = modal.Volume.from_name("qwen-eval-hf-cache", create_if_missing=True)
    output_volume = modal.Volume.from_name("qwen-sdpo-output", create_if_missing=True)

    # Persists torch.compile kernel cache across cold starts.
    # Reduces first-inference latency from ~220s → ~11s after the first run ever.
    compile_cache_volume = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)

    inference_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.6.3-devel-ubuntu22.04",
            add_python="3.11",
        )
        # vLLM nightly: required for Qwen3.5's GatedDeltaNet architecture.
        .pip_install(
            "vllm",
            extra_index_url="https://wheels.vllm.ai/nightly",
            pre=True,
        )
        # Supporting libs after vLLM to avoid version conflicts.
        .pip_install(
            "accelerate",
            "datasets",
            "sentencepiece",
            "protobuf",
            "matplotlib",
            "peft",
            "bitsandbytes",
        )
        # transformers>=5.2.0 is required: qwen3_5 model type was added in v5.2.0 (Feb 2026).
        # Install AFTER vLLM so this pip_install step overrides whatever vLLM pinned.
        # Changing the version pin here (from >=4.51.0) also busts Modal's layer cache.
        .pip_install("transformers>=5.2.0")
        .add_local_python_source("qwen_sdpo")
        .add_local_python_source("qwen_eval")
    )
