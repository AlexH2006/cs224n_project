"""
vLLM Throughput Benchmark — Modal H100

TLDR: Measures raw vLLM generation throughput (tokens/s) for Qwen3.5-4B on a
single MiniF2F problem (default: index 4, test split, pass@1). Records wall-clock
time, output token count, and tokens/s. Saves results to devlog/throughput_results/.

Design goals:
  - Isolate *generation* latency only — no verification, no Kimina.
  - Measure: TTFT (time-to-first-token via prefill proxy), total generation time,
    output tokens generated, and effective tokens/s.
  - Two warmup rounds before the timed round so CUDA graph capture cost is excluded.
  - Configurable pass_k so you can measure batch-of-1 vs batch-of-N.

Usage:
    # Single problem, pass@1 (batch size 1), measure throughput:
    python3 -m modal run qwen_multiturn/tests/test_vllm_throughput.py

    # Batch of 4 (measures throughput of the normal eval mode):
    python3 -m modal run qwen_multiturn/tests/test_vllm_throughput.py --pass-k 4

    # Different problem index:
    python3 -m modal run qwen_multiturn/tests/test_vllm_throughput.py --problem-idx 0 --pass-k 1
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import modal

from qwen_multiturn.config import EvalConfig
from qwen_multiturn.dataset import load_problems

# ---------------------------------------------------------------------------
# Modal app + image (reuse the same inference image as the eval pipeline)
# ---------------------------------------------------------------------------

app = modal.App("qwen-eval-throughput-bench")

hf_cache_volume = modal.Volume.from_name("qwen-eval-hf-cache", create_if_missing=True)

inference_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm",
        extra_index_url="https://wheels.vllm.ai/nightly",
        pre=True,
    )
    # transformers>=5.2.0: qwen3_5 model type added in v5.2.0 (Feb 2026).
    # Install AFTER vLLM to override whatever vLLM pinned.
    .pip_install(
        "transformers>=5.2.0",
        "accelerate",
        "datasets",
        "sentencepiece",
        "protobuf",
    )
    .add_local_python_source("qwen_multiturn")
)


# ---------------------------------------------------------------------------
# Modal class: runs vLLM on H100
# ---------------------------------------------------------------------------

@app.cls(
    image=inference_image,
    gpu="H100",
    timeout=1800,
    volumes={"/hf_cache": hf_cache_volume},
)
class ThroughputBenchmark:
    """
    Loads Qwen3.5-4B via vLLM (identical config to ProofGenerator in modal_app.py)
    and runs timed generation passes.
    """

    @modal.enter()
    def setup(self):
        self._llm = None
        self._tokenizer = None
        self._sampling_params = None

    def _load(self, model_name: str, sampling_cfg: dict) -> None:
        """Load model + build SamplingParams. Idempotent."""
        import os
        from vllm import LLM, SamplingParams

        os.environ["HF_HOME"] = "/hf_cache"

        if self._llm is not None:
            return

        print(f"[bench] Loading model: {model_name}")
        load_start = time.perf_counter()
        self._llm = LLM(
            model=model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            download_dir="/hf_cache",
            gpu_memory_utilization=0.90,
            max_model_len=16384,
            language_model_only=True,
            enforce_eager=False,           # CUDA graphs ON
            enable_prefix_caching=True,
        )
        load_elapsed = time.perf_counter() - load_start
        print(f"[bench] Model loaded in {load_elapsed:.1f}s")

        self._sampling_params = SamplingParams(
            temperature=sampling_cfg["temperature"],
            top_p=sampling_cfg["top_p"],
            top_k=sampling_cfg["top_k"],
            min_p=sampling_cfg["min_p"],
            presence_penalty=sampling_cfg["presence_penalty"],
            repetition_penalty=sampling_cfg["repetition_penalty"],
            max_tokens=sampling_cfg["max_new_tokens"],
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._model_load_s = load_elapsed
        print("[bench] Model ready.")

    @modal.method()
    def run_benchmark(
        self,
        problem: dict,
        pass_k: int,
        model_name: str,
        sampling_cfg: dict,
        n_warmup: int = 2,
    ) -> dict:
        """
        Run one timed generation benchmark for a single problem.

        Args:
            problem:      Problem dict (formal_statement, informal_stmt, header).
            pass_k:       Batch size — number of parallel proof attempts.
            model_name:   HuggingFace model ID.
            sampling_cfg: Sampling hyperparameters dict.
            n_warmup:     Number of warmup rounds before the timed round.
                          Warmup excludes CUDA graph capture from the benchmark time.

        Returns:
            Dict with timing and throughput metrics.
        """
        from qwen_multiturn.prompts import build_prompt

        self._load(model_name, sampling_cfg)

        prompt = build_prompt(
            theorem_code=problem["formal_statement"],
            informal=problem["informal_stmt"],
            header=problem["header"],
            tokenizer=self._tokenizer,
            cfg=EvalConfig(),
        )
        prompts = [prompt] * pass_k

        prompt_tokens = len(self._tokenizer.encode(prompt))
        print(f"[bench] Prompt tokens: {prompt_tokens}")
        print(f"[bench] Batch size (pass_k): {pass_k}")
        print(f"[bench] Max new tokens: {sampling_cfg['max_new_tokens']}")

        # ------------------------------------------------------------------
        # Warmup rounds — lets vLLM finalize CUDA graph capture and scheduler
        # state. We use a small max_tokens to keep warmup fast.
        # ------------------------------------------------------------------
        from vllm import SamplingParams as SP

        warmup_params = SP(
            temperature=sampling_cfg["temperature"],
            top_p=sampling_cfg["top_p"],
            top_k=sampling_cfg["top_k"],
            min_p=sampling_cfg["min_p"],
            max_tokens=64,  # short warmup
        )
        for i in range(n_warmup):
            print(f"[bench] Warmup round {i + 1}/{n_warmup}...")
            warmup_start = time.perf_counter()
            self._llm.generate(prompts, warmup_params)
            warmup_elapsed = time.perf_counter() - warmup_start
            print(f"[bench]   Warmup {i + 1} done in {warmup_elapsed:.2f}s")

        # ------------------------------------------------------------------
        # Timed round — full max_tokens generation
        # ------------------------------------------------------------------
        print(f"[bench] Starting timed generation (pass_k={pass_k})...")
        gen_start = time.perf_counter()
        outputs = self._llm.generate(prompts, self._sampling_params)
        gen_elapsed = time.perf_counter() - gen_start

        # ------------------------------------------------------------------
        # Collect per-sequence metrics
        # ------------------------------------------------------------------
        seq_metrics = []
        total_output_tokens = 0
        for i, out in enumerate(outputs):
            seq_out = out.outputs[0]
            # vLLM reports token count directly in the completion object.
            output_tokens = len(seq_out.token_ids)
            total_output_tokens += output_tokens
            seq_metrics.append({
                "seq_idx": i,
                "output_tokens": output_tokens,
                "finish_reason": seq_out.finish_reason,
                "truncated": seq_out.finish_reason == "length",
                "output_preview": seq_out.text[:200].replace("\n", " "),
            })
            print(
                f"[bench]   seq {i}: {output_tokens} tokens, "
                f"finish={seq_out.finish_reason!r}"
            )

        tokens_per_sec_total = total_output_tokens / gen_elapsed if gen_elapsed > 0 else 0
        tokens_per_sec_per_seq = tokens_per_sec_total / pass_k if pass_k > 0 else 0

        result = {
            "model_name": model_name,
            "problem_idx": problem["problem_idx"],
            "problem_id": problem["problem_id"],
            "pass_k": pass_k,
            "prompt_tokens": prompt_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_output_tokens_per_seq": total_output_tokens / pass_k,
            "gen_wall_clock_s": round(gen_elapsed, 3),
            "tokens_per_sec_total": round(tokens_per_sec_total, 2),
            "tokens_per_sec_per_seq": round(tokens_per_sec_per_seq, 2),
            "n_warmup_rounds": n_warmup,
            "max_new_tokens": sampling_cfg["max_new_tokens"],
            "gpu": "H100",
            "vllm_config": {
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.90,
                "max_model_len": 16384,
                "enforce_eager": False,
                "enable_prefix_caching": True,
            },
            "sequences": seq_metrics,
        }

        print(f"\n[bench] === RESULT ===")
        print(f"[bench]   Wall clock:       {gen_elapsed:.2f}s")
        print(f"[bench]   Total out tokens: {total_output_tokens}")
        print(f"[bench]   Throughput total: {tokens_per_sec_total:.1f} tok/s")
        print(f"[bench]   Throughput/seq:   {tokens_per_sec_per_seq:.1f} tok/s")
        print(f"[bench]===================")

        return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_benchmark(
    problem_idx: int = 4,
    pass_k: int = 1,
    model: str = "Qwen/Qwen3.5-4B",
    max_new_tokens: int = 8192,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    n_warmup: int = 2,
    dataset_split: str = "test",
    output_dir: str = "devlog/throughput_results",
):
    """
    Benchmark vLLM throughput for Qwen3.5-4B on a single MiniF2F problem.

    Results are printed to stdout and saved as JSON in devlog/throughput_results/.

    Default: problem_idx=4, pass@1, test split — matching user's reported observation
    of ~4 minutes for 8000 tokens (which would be ~33 tok/s if true).
    """
    cfg = EvalConfig(
        model_name=model,
        n_problems=1,
        pass_k=pass_k,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        problem_indices=[problem_idx],
        dataset_split=dataset_split,
    )

    sampling_cfg = {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k,
        "min_p": cfg.min_p,
        "presence_penalty": cfg.presence_penalty,
        "repetition_penalty": cfg.repetition_penalty,
        "max_new_tokens": cfg.max_new_tokens,
    }

    print("=" * 60)
    print("vLLM Throughput Benchmark")
    print(f"  model:         {model}")
    print(f"  problem_idx:   {problem_idx}  (split={dataset_split})")
    print(f"  pass_k:        {pass_k}")
    print(f"  max_new_tokens:{max_new_tokens}")
    print(f"  n_warmup:      {n_warmup}")
    print("=" * 60)

    # Load problem locally — dataset is tiny, no GPU needed.
    problems = load_problems(cfg)
    problem = problems[0]
    print(f"\nProblem: [{problem['problem_idx']}] {problem['problem_id']}")
    print(f"  formal_statement: {problem['formal_statement'][:100]}...")

    # Dispatch to Modal H100.
    bench = ThroughputBenchmark()
    result = bench.run_benchmark.remote(
        problem=problem,
        pass_k=pass_k,
        model_name=model,
        sampling_cfg=sampling_cfg,
        n_warmup=n_warmup,
    )

    # Add metadata not available inside the Modal container.
    result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    result["dataset_split"] = dataset_split

    # ------------------------------------------------------------------
    # Save results to devlog/throughput_results/
    # ------------------------------------------------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_safe = model.replace("/", "_").replace("-", "_")
    fname = f"bench_{model_safe}_prob{problem_idx}_passk{pass_k}_{result['timestamp']}.json"
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print("\n" + "=" * 60)
    print("THROUGHPUT SUMMARY")
    print("=" * 60)
    print(f"  Model:              {result['model_name']}")
    print(f"  Problem:            [{result['problem_idx']}] {result['problem_id']}")
    print(f"  Pass@k (batch):     {result['pass_k']}")
    print(f"  Prompt tokens:      {result['prompt_tokens']}")
    print(f"  Total output tokens:{result['total_output_tokens']}")
    print(f"  Avg tokens/seq:     {result['avg_output_tokens_per_seq']:.0f}")
    print(f"  Wall clock:         {result['gen_wall_clock_s']:.2f}s")
    print(f"  Throughput (total): {result['tokens_per_sec_total']:.1f} tok/s")
    print(f"  Throughput (per seq):{result['tokens_per_sec_per_seq']:.1f} tok/s")
    print(f"  Truncated seqs:     {sum(1 for s in result['sequences'] if s['truncated'])}/{result['pass_k']}")
    print("=" * 60)
