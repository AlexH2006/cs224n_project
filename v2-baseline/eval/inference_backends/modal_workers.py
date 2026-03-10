import time
import math

import modal


class ModalWorkersBackend:
    """Inference backend that dispatches generation to a deployed Modal app."""

    def __init__(
        self,
        app_name: str = "goedel-prover-modal-workers",
        function_name: str = "generate_n_for_prompt",
        model_name: str | None = None,
    ):
        self.app_name = app_name
        self.function_name = function_name
        self.model_name = model_name

    def generate(
        self,
        prompts: list[str],
        n: int = 32,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        max_batch_size: int = 50,
        **kwargs,
    ) -> list[list[str]]:
        """Return list-of-lists: one list of n completion strings per prompt.

        Dispatches work to Modal using fn.map() in chunks of max_batch_size.
        """
        fn = modal.Function.from_name(self.app_name, self.function_name)
        use_batched = self.function_name == "generate_n_for_prompt"

        if use_batched:
            return self._generate_batched(
                fn, prompts, n, temperature, top_p, max_tokens, max_batch_size
            )
        else:
            return self._generate_per_attempt(
                fn, prompts, n, temperature, top_p, max_tokens, max_batch_size
            )

    def _generate_batched(
        self, fn, prompts, n, temperature, top_p, max_tokens, max_batch_size
    ):
        """One Modal call per prompt, each producing n completions."""
        jobs = []
        for idx, prompt in enumerate(prompts):
            job = {
                "prompt": prompt,
                "prompt_idx": idx,
                "n": n,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if self.model_name:
                job["model_name"] = self.model_name
            jobs.append(job)

        results_by_idx: dict[int, list[str]] = {}
        num_chunks = math.ceil(len(jobs) / max_batch_size)

        for chunk_i in range(num_chunks):
            chunk = jobs[chunk_i * max_batch_size : (chunk_i + 1) * max_batch_size]
            print(
                f"[Modal] Dispatching chunk {chunk_i + 1}/{num_chunks} "
                f"({len(chunk)} prompts)"
            )
            t0 = time.time()
            retries = 0
            while retries < 3:
                try:
                    for result in fn.map(chunk, order_outputs=False):
                        results_by_idx[result["prompt_idx"]] = result["texts"]
                    break
                except Exception as e:
                    retries += 1
                    print(f"[Modal] Chunk {chunk_i + 1} failed (attempt {retries}): {e}")
                    if retries >= 3:
                        self._fallback_remote(fn, chunk, results_by_idx)
                    else:
                        time.sleep(2 ** retries)

            elapsed = time.time() - t0
            print(f"[Modal] Chunk {chunk_i + 1} done in {elapsed:.1f}s")

        return [results_by_idx[i] for i in range(len(prompts))]

    def _generate_per_attempt(
        self, fn, prompts, n, temperature, top_p, max_tokens, max_batch_size
    ):
        """One Modal call per (prompt, attempt) pair."""
        jobs = []
        for idx, prompt in enumerate(prompts):
            for attempt in range(n):
                job = {
                    "prompt": prompt,
                    "prompt_idx": idx,
                    "attempt_idx": attempt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                if self.model_name:
                    job["model_name"] = self.model_name
                jobs.append(job)

        results_by_idx: dict[int, list[str | None]] = {
            i: [None] * n for i in range(len(prompts))
        }
        num_chunks = math.ceil(len(jobs) / max_batch_size)

        for chunk_i in range(num_chunks):
            chunk = jobs[chunk_i * max_batch_size : (chunk_i + 1) * max_batch_size]
            print(
                f"[Modal] Dispatching chunk {chunk_i + 1}/{num_chunks} "
                f"({len(chunk)} attempts)"
            )
            t0 = time.time()
            retries = 0
            while retries < 3:
                try:
                    for result in fn.map(chunk, order_outputs=False):
                        pidx = result["prompt_idx"]
                        aidx = result["attempt_idx"]
                        results_by_idx[pidx][aidx] = result["text"]
                    break
                except Exception as e:
                    retries += 1
                    print(f"[Modal] Chunk {chunk_i + 1} failed (attempt {retries}): {e}")
                    if retries >= 3:
                        self._fallback_remote(fn, chunk, results_by_idx, per_attempt=True)
                    else:
                        time.sleep(2 ** retries)

            elapsed = time.time() - t0
            print(f"[Modal] Chunk {chunk_i + 1} done in {elapsed:.1f}s")

        return [results_by_idx[i] for i in range(len(prompts))]

    @staticmethod
    def _fallback_remote(fn, chunk, results_by_idx, per_attempt=False):
        """Last-resort: call fn.remote() one job at a time."""
        print("[Modal] Falling back to fn.remote() for remaining jobs")
        for job in chunk:
            pidx = job["prompt_idx"]
            if per_attempt:
                aidx = job["attempt_idx"]
                if results_by_idx[pidx][aidx] is not None:
                    continue
            else:
                if pidx in results_by_idx:
                    continue
            try:
                result = fn.remote(job)
                if per_attempt:
                    results_by_idx[pidx][result["attempt_idx"]] = result["text"]
                else:
                    results_by_idx[pidx] = result["texts"]
            except Exception as e:
                print(f"[Modal] fn.remote() failed for prompt_idx={pidx}: {e}")
                if per_attempt:
                    results_by_idx[pidx][job["attempt_idx"]] = ""
                else:
                    results_by_idx[pidx] = [""] * job.get("n", 1)
