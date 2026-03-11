"""
Qwen MiniF2F Eval Pipeline (Modal)

TLDR: Two-phase eval pipeline. Phase 1 runs on a single Modal H100: one worker
loads vLLM once and processes all problems in batched generate() calls.
Phase 2 (parse + verify) runs locally; results are saved incrementally per problem.

Phase 1 — Generation (Modal H100, single GPU):
    ProofGenerator loads the model via vLLM (bfloat16) once.
    generate_all() builds prompts for all problems (each × pass_k), runs vLLM
    generate() in one or more batches on that GPU, returns one result list.

Phase 2 — Verification (local driver):
    extract_full_lean_block() + create_full_lean_code() parse each raw output.
    verify() calls the Kimina Lean Server on localhost.
    Results written to baseline/run_{model}_{timestamp}/; saved after each problem.

Usage:
    # Start Kimina first:
    #   docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0

    python3 -m modal run qwen_eval/modal_app.py
    python3 -m modal run qwen_eval/modal_app.py --model "Qwen/Qwen3.5-9B" --problem-idx 0 --pass-k 4
    python3 -m modal run qwen_eval/modal_app.py --n-problems 20 --pass-k 4
    python3 -m modal run qwen_eval/modal_app.py --generation-save-batch-size 50   # save after each 50 problems
    python3 -m modal run qwen_eval/modal_app.py --generate-only
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Optional

import modal

from qwen_eval.batch_generation import build_flat_prompts_and_meta, unflatten_results
from qwen_eval.config import EvalConfig
from qwen_eval.dataset import load_problems
from qwen_eval.local_lean_verifier import verify
from qwen_eval.parsing import create_full_lean_code, extract_full_lean_block_parsed
from qwen_eval.results import build_problem_log, make_run_dir, save_results

# ---------------------------------------------------------------------------
# Modal app + image
# ---------------------------------------------------------------------------

app = modal.App("qwen-eval-minif2f")

# Persistent volume — avoids re-downloading the model on every run.
hf_cache_volume = modal.Volume.from_name("qwen-eval-hf-cache", create_if_missing=True)

inference_image = (
    # FlashInfer's GDN SM90 kernel (used by Qwen3.5's Gated DeltaNet layers) requires
    # JIT compilation at first inference. This JIT uses PTX intrinsics introduced in
    # CUDA 12.6 (fence_proxy_tensormap_generic, n32_t). CUDA 12.4 nvcc fails with
    # "namespace cuda::ptx has no member fence_proxy_tensormap_generic".
    # Solution: use CUDA 12.6+ devel image which ships a compatible nvcc.
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.11",
    )
    # vLLM nightly: required for Qwen3.5 GDN architecture support.
    # Recipe: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
    .pip_install(
        "vllm",
        extra_index_url="https://wheels.vllm.ai/nightly",
        pre=True,
    )
    # transformers>=5.2.0: qwen3_5 model type was added in v5.2.0 (Feb 2026).
    # Install AFTER vLLM so this overrides whatever vLLM pinned.
    .pip_install(
        "transformers>=5.2.0",
        "accelerate",
        "datasets",
        "sentencepiece",
        "protobuf",
    )
    # Ship the local qwen_eval package into the container so all imports work.
    .add_local_python_source("qwen_eval")
)


# ---------------------------------------------------------------------------
# Phase 1: Generation
# ---------------------------------------------------------------------------

@app.cls(
    image=inference_image,
    gpu="H100",
    timeout=7200,  # 2 hours
    volumes={"/hf_cache": hf_cache_volume},
)
class ProofGenerator:
    """
    Runs vLLM on a single Modal H100. One worker for the whole run: model is loaded
    once; generate_all() processes all problems in one or more batched generate() calls.
    """

    @modal.enter()
    def setup(self):
        """Pre-warm: nothing to do — model loaded on first call."""
        self._model_name: Optional[str] = None
        self._llm = None
        self._tokenizer = None

    def _load(self, model_name: str, sampling_cfg: dict) -> None:
        """Load vLLM model + tokenizer if not already loaded."""
        import os
        from vllm import LLM, SamplingParams

        os.environ["HF_HOME"] = "/hf_cache"

        if self._llm is not None and self._model_name == model_name:
            return  # already loaded, reuse

        print(f"Loading model: {model_name}")
        self._llm = LLM(
            model=model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            download_dir="/hf_cache",
            gpu_memory_utilization=0.90,
            max_model_len=16384,
            # Skip vision encoder: Qwen3.5 is multimodal but we only need text.
            # Equivalent to CLI --language-model-only: frees GPU memory for KV cache.
            # Recipe: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
            language_model_only=True,
            # CUDA graphs enabled (default). The nightly vLLM wheel ships pre-compiled
            # FlashInfer GDN kernels so no JIT/nvcc is needed at inference time.
            enforce_eager=False,
            enable_prefix_caching=True,
        )
        from vllm import SamplingParams
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
        self._model_name = model_name
        print("Model ready.")

    @modal.method()
    def generate_all(
        self,
        problems: list[dict],
        pass_k: int,
        model_name: str,
        sampling_cfg: dict,
        inference_batch_size: Optional[int] = 128,
    ) -> dict:
        """
        Build prompts for all problems (each × pass_k), run vLLM generate on one GPU
        in one or more batches. Returns a dict with raw_results and generation_metrics
        (throughput, wall time, token counts) for debugging.

        Returned dict:
            raw_results: list[list] — result[i] = [prompt_i, (raw_0, reason_0), ...].
            generation_metrics: dict with generation_wall_s, total_output_tokens,
                tokens_per_second, n_requests, avg_generation_s_per_problem.
        """
        import time
        from qwen_eval.config import EvalConfig
        from qwen_eval.prompts import build_prompt

        if not problems:
            return {"raw_results": [], "generation_metrics": {}}

        self._load(model_name, sampling_cfg)

        def prompt_builder(p: dict) -> str:
            return build_prompt(
                theorem_code=p["formal_statement"],
                informal=p["informal_stmt"],
                header=p["header"],
                tokenizer=self._tokenizer,
                cfg=EvalConfig(),
            )

        flat_prompts, prompt_meta = build_flat_prompts_and_meta(
            problems, pass_k, prompt_builder
        )

        batch_size = inference_batch_size or 0
        if batch_size <= 0:
            batch_size = len(flat_prompts)

        all_outputs: list[tuple[str, str]] = []
        total_output_tokens = 0
        gen_start = time.perf_counter()

        for start in range(0, len(flat_prompts), batch_size):
            chunk = flat_prompts[start : start + batch_size]
            chunk_out = self._llm.generate(chunk, self._sampling_params)
            for o in chunk_out:
                out = o.outputs[0]
                all_outputs.append((out.text, out.finish_reason))
                # vLLM CompletionOutput has token_ids; fallback to 0 if missing.
                token_ids = getattr(out, "token_ids", None)
                total_output_tokens += len(token_ids) if token_ids is not None else 0

        generation_wall_s = time.perf_counter() - gen_start
        n_requests = len(all_outputs)
        tokens_per_second = total_output_tokens / generation_wall_s if generation_wall_s > 0 else 0.0
        n_problems = len(problems)
        avg_generation_s_per_problem = generation_wall_s / n_problems if n_problems > 0 else 0.0

        generation_metrics = {
            "generation_wall_s": round(generation_wall_s, 4),
            "total_output_tokens": total_output_tokens,
            "tokens_per_second": round(tokens_per_second, 2),
            "n_requests": n_requests,
            "avg_generation_s_per_problem": round(avg_generation_s_per_problem, 4),
        }

        raw_results = unflatten_results(prompt_meta, all_outputs, problems)
        return {"raw_results": raw_results, "generation_metrics": generation_metrics}


# ---------------------------------------------------------------------------
# Phase 2: Verification helpers (run locally on the driver)
# ---------------------------------------------------------------------------

@dataclass
class AttemptResult:
    """Parsed outputs for one (problem, attempt) pair, before verification."""
    problem_idx: int
    attempt: int
    prompt: str
    raw_output: str
    extracted_block: str
    full_code: str
    num_tokens: int
    truncated: bool = False        # True iff reasoning was cut off before the code block
    finish_reason: str = "stop"   # vLLM finish_reason: "stop" (EOS) or "length" (limit hit)


def _verify_problem(
    problem: dict,
    attempts: list[AttemptResult],
    cfg: EvalConfig,
) -> list[dict]:
    """
    Verify all attempts for one problem serially.

    Every attempt is verified — no early stopping — so the logs capture
    the full pass@k picture (how many of k attempts succeeded).
    Retries on Kimina server errors.

    Returns list of attempt log dicts ready for build_problem_log().
    """
    attempt_logs = []

    for att in attempts:
        verification = None
        for retry in range(cfg.verify_retries + 1):
            verification = verify(
                att.full_code,
                kimina_url=cfg.kimina_url,
                timeout=cfg.verify_timeout_s,
            )
            if not verification.get("is_server_error"):
                break
            if retry < cfg.verify_retries:
                print(
                    f"    server error on attempt {att.attempt}, "
                    f"retrying ({retry + 1}/{cfg.verify_retries})..."
                )
                time.sleep(cfg.verify_retry_wait_s)

        attempt_logs.append({
            "attempt": att.attempt,
            "prompt": att.prompt,
            "raw_output": att.raw_output,
            "extracted_block": att.extracted_block,
            "full_code": att.full_code,
            "verification": verification,
            "num_tokens": att.num_tokens,
            "truncated": att.truncated,
            "finish_reason": att.finish_reason,
        })

    return attempt_logs


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_eval(
    n_problems: int = 20,
    problem_idx: int = -1,         # -1 = first n_problems; >=0 = single problem at that index
    pass_k: int = 4,
    model: str = "Qwen/Qwen3.5-4B",
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 8192,
    seed: int = 42,
    kimina_url: str = "http://localhost:8000",
    generate_only: bool = False,
    inference_batch_size: Optional[int] = None,
    generation_save_batch_size: Optional[int] = None,
):
    """
    Orchestrate the full eval: generate → parse → verify → save.

    --problem-idx N  Run on a single problem at dataset index N (overrides --n-problems).
    --generate-only  Skip verification (useful for testing generation + parsing).
    --inference-batch-size N  Max prompts per vLLM generate() call (default: 256).
    --generation-save-batch-size N  Save results after each batch of N problems (default: all at once).
    """
    problem_indices = [problem_idx] if problem_idx >= 0 else None
    cfg = EvalConfig(
        model_name=model,
        n_problems=1 if problem_idx >= 0 else n_problems,
        pass_k=pass_k,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        seed=seed,
        kimina_url=kimina_url,
        problem_indices=problem_indices,
    )
    if inference_batch_size is not None:
        cfg.inference_batch_size = inference_batch_size
    if generation_save_batch_size is not None:
        cfg.generation_save_batch_size = generation_save_batch_size

    # Sampling config dict passed explicitly to Modal workers (avoids env var hack).
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
    print(f"Qwen MiniF2F Eval  |  model={cfg.model_name}")
    print(f"  dataset={cfg.dataset_name}, split={cfg.dataset_split}")
    print(f"  n_problems={cfg.n_problems}, pass@{cfg.pass_k}")
    print(f"  kimina={cfg.kimina_url}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load dataset locally (small, no GPU needed)
    # ------------------------------------------------------------------
    problems = load_problems(cfg)
    print(f"\n[1/3] Loaded {len(problems)} problems")
    for p in problems:
        print(f"  [{p['problem_idx']}] {p['problem_id']}: {p['formal_statement'][:60]}...")

    run_dir = make_run_dir(cfg)

    # Problem batches: save results after each batch of N problems (None = one batch = all).
    batch_size = cfg.generation_save_batch_size or 0
    if batch_size <= 0:
        batch_size = len(problems)
    chunks = [
        problems[i : i + batch_size]
        for i in range(0, len(problems), batch_size)
    ]
    n_chunks = len(chunks)

    generator = ProofGenerator()
    aggregated_metrics: dict[str, float | int] = {
        "generation_wall_s": 0.0,
        "total_output_tokens": 0,
        "n_requests": 0,
    }
    problem_logs: list[dict] = []

    if not generate_only:
        print(f"\n[3/3] Verifying with Kimina at {cfg.kimina_url}...")

    for chunk_idx, problem_chunk in enumerate(chunks):
        # ------------------------------------------------------------------
        # Phase 1 (this chunk): Generate on Modal
        # ------------------------------------------------------------------
        print(f"\n[2/3] Generating chunk {chunk_idx + 1}/{n_chunks} ({len(problem_chunk)} problems, {len(problem_chunk) * cfg.pass_k} outputs)...")
        gen_response = generator.generate_all.remote(
            problem_chunk,
            cfg.pass_k,
            cfg.model_name,
            sampling_cfg,
            cfg.inference_batch_size,
        )
        raw_results_chunk: list[list] = gen_response["raw_results"]
        metrics = gen_response.get("generation_metrics") or {}
        aggregated_metrics["generation_wall_s"] += metrics.get("generation_wall_s", 0)
        aggregated_metrics["total_output_tokens"] += metrics.get("total_output_tokens", 0)
        aggregated_metrics["n_requests"] += metrics.get("n_requests", 0)
        if metrics:
            tps = metrics.get("tokens_per_second", 0)
            wall = metrics.get("generation_wall_s", 0)
            print(f"  Chunk done: {wall:.1f}s, {tps:.1f} tok/s")

        # ------------------------------------------------------------------
        # Parse: extract lean4 blocks from raw outputs (this chunk)
        # ------------------------------------------------------------------
        all_attempts_chunk: list[list[AttemptResult]] = []
        for problem, result in zip(problem_chunk, raw_results_chunk):
            prompt = result[0]
            raw_pairs = result[1:]
            problem_attempts = []
            for attempt_idx, (raw_output, finish_reason) in enumerate(raw_pairs):
                parse_result = extract_full_lean_block_parsed(
                    raw_output,
                    finish_reason=finish_reason,
                )
                extracted_block = parse_result.block
                if parse_result.truncated:
                    print(f"    [attempt {attempt_idx}] TRUNCATED (finish_reason={finish_reason!r}) — no code block reached")
                elif parse_result.no_block:
                    print(f"    [attempt {attempt_idx}] PARSE FAILED — no lean4 block found in output")
                full_code = create_full_lean_code(
                    theorem_code=problem["formal_statement"],
                    extracted_block=extracted_block,
                    default_header=cfg.default_header,
                )
                problem_attempts.append(AttemptResult(
                    problem_idx=problem["problem_idx"],
                    attempt=attempt_idx,
                    prompt=prompt,
                    raw_output=raw_output,
                    extracted_block=extracted_block,
                    full_code=full_code,
                    num_tokens=len(raw_output.split()),
                    truncated=parse_result.truncated,
                    finish_reason=finish_reason,
                ))
            all_attempts_chunk.append(problem_attempts)

        # ------------------------------------------------------------------
        # Verify (or build logs without verification) and append to problem_logs
        # ------------------------------------------------------------------
        if generate_only:
            print("  --generate-only: skipping verification.")
            total_wall = aggregated_metrics["generation_wall_s"]
            n_after = len(problem_logs) + len(problem_chunk)
            avg_gen_s = (total_wall / n_after) if n_after > 0 else 0.0
            for problem, attempts in zip(problem_chunk, all_attempts_chunk):
                problem_log = build_problem_log(
                    problem,
                    [
                        {
                            "attempt": a.attempt,
                            "prompt": a.prompt,
                            "raw_output": a.raw_output,
                            "extracted_block": a.extracted_block,
                            "full_code": a.full_code,
                            "verification": None,
                            "num_tokens": a.num_tokens,
                        }
                        for a in attempts
                    ],
                    cfg,
                    generation_time_s=avg_gen_s,
                )
                problem_logs.append(problem_log)
        else:
            for prob_num, (problem, attempts) in enumerate(zip(problem_chunk, all_attempts_chunk), len(problem_logs) + 1):
                print(f"  [{prob_num}/{len(problems)}] {problem['problem_id']}")
                verify_start = time.perf_counter()
                attempt_logs = _verify_problem(problem, attempts, cfg)
                verification_time_s = time.perf_counter() - verify_start
                total_wall = aggregated_metrics["generation_wall_s"]
                n_so_far = len(problem_logs) + 1
                avg_gen_s = (total_wall / n_so_far) if n_so_far > 0 else 0.0
                problem_log = build_problem_log(
                    problem,
                    attempt_logs,
                    cfg,
                    verification_time_s=verification_time_s,
                    generation_time_s=avg_gen_s,
                )
                problem_logs.append(problem_log)
                status = "SOLVED" if problem_log["success"] else "failed"
                best = problem_log.get("best_attempt")
                print(f"    → {status}" + (f" (attempt {best})" if best is not None else "") + f"  [{verification_time_s:.1f}s]")

        # Derived metrics for summary (so far)
        total_wall = aggregated_metrics["generation_wall_s"]
        n_done = len(problem_logs)
        save_metrics = {
            "generation_wall_s": round(total_wall, 4),
            "total_output_tokens": aggregated_metrics["total_output_tokens"],
            "tokens_per_second": round(
                aggregated_metrics["total_output_tokens"] / total_wall, 2
            ) if total_wall > 0 else 0.0,
            "n_requests": aggregated_metrics["n_requests"],
            "avg_generation_s_per_problem": round(total_wall / n_done, 4) if n_done > 0 else 0.0,
        }
        save_results(run_dir, cfg, problem_logs, generation_metrics=save_metrics)
        print(f"  Saved results ({n_done}/{len(problems)} problems).")

    if problem_logs:
        total_wall = aggregated_metrics["generation_wall_s"]
        n_done = len(problem_logs)
        final_metrics = {
            "generation_wall_s": round(total_wall, 4),
            "total_output_tokens": aggregated_metrics["total_output_tokens"],
            "tokens_per_second": round(
                aggregated_metrics["total_output_tokens"] / total_wall, 2
            ) if total_wall > 0 else 0.0,
            "n_requests": aggregated_metrics["n_requests"],
            "avg_generation_s_per_problem": round(total_wall / n_done, 4),
        }
        save_results(run_dir, cfg, problem_logs, generation_metrics=final_metrics)
    print(f"\nDone. Results in: {run_dir}")
