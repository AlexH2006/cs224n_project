import time

from eval.inference_backends.base import InferenceBackend


class ModalWorkersBackend(InferenceBackend):
    def __init__(
        self,
        app_name="goedel-prover-modal-workers",
        function_name="generate_one_attempt",
    ):
        self.app_name = app_name
        self.function_name = function_name

    def generate(self, prompts, n, temperature, top_p, max_tokens, max_batch_size, model_name=None):
        try:
            import modal
        except Exception as exc:
            raise RuntimeError("Modal SDK not available. Install with `pip install modal`.") from exc

        fn = modal.Function.from_name(self.app_name, self.function_name)
        chunk_size = max(1, int(max_batch_size))
        max_retries = 2
        jobs = []
        for prompt_idx, prompt in enumerate(prompts):
            for attempt_idx in range(n):
                jobs.append(
                    {
                        "prompt_idx": prompt_idx,
                        "attempt_idx": attempt_idx,
                        "prompt": prompt,
                        "model_name": model_name,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                    }
                )

        results_by_key = {}
        try:
            from tqdm import tqdm

            progress = tqdm(total=len(jobs), desc="Modal worker attempts", unit="attempt")
            use_tqdm = True
        except Exception:
            progress = None
            use_tqdm = False
            print(f"Modal worker attempts: {len(jobs)} total")

        def _store_result(result):
            key = (int(result["prompt_idx"]), int(result["attempt_idx"]))
            results_by_key[key] = result["text"]
            if use_tqdm and progress is not None:
                progress.update(1)

        for start in range(0, len(jobs), chunk_size):
            chunk_jobs = jobs[start:start + chunk_size]
            chunk_done = False
            for retry in range(max_retries + 1):
                try:
                    for result in fn.map(chunk_jobs, order_outputs=False):
                        _store_result(result)
                    chunk_done = True
                    break
                except Exception as exc:
                    if retry < max_retries:
                        wait_s = min(10, 2 ** retry)
                        print(
                            f"[modal_workers] chunk {start}:{start + len(chunk_jobs)} failed "
                            f"(retry {retry + 1}/{max_retries + 1}): {exc}. Waiting {wait_s}s..."
                        )
                        time.sleep(wait_s)
                    else:
                        print(
                            f"[modal_workers] chunk {start}:{start + len(chunk_jobs)} failed after retries. "
                            "Falling back to per-attempt calls."
                        )
            if chunk_done:
                continue

            for job in chunk_jobs:
                job_done = False
                for retry in range(max_retries + 1):
                    try:
                        result = fn.remote(job)
                        _store_result(result)
                        job_done = True
                        break
                    except Exception as exc:
                        if retry < max_retries:
                            wait_s = min(10, 2 ** retry)
                            print(
                                f"[modal_workers] single attempt failed for prompt={job['prompt_idx']} "
                                f"attempt={job['attempt_idx']} (retry {retry + 1}/{max_retries + 1}): {exc}. "
                                f"Waiting {wait_s}s..."
                            )
                            time.sleep(wait_s)
                        else:
                            raise RuntimeError(
                                f"Modal worker failed for prompt={job['prompt_idx']} attempt={job['attempt_idx']}"
                            ) from exc
                if not job_done:
                    raise RuntimeError(
                        f"Unable to complete prompt={job['prompt_idx']} attempt={job['attempt_idx']}"
                    )

        if use_tqdm and progress is not None:
            progress.close()

        outputs = []
        for prompt_idx in range(len(prompts)):
            group = []
            for attempt_idx in range(n):
                key = (prompt_idx, attempt_idx)
                if key not in results_by_key:
                    raise ValueError(f"Missing worker output for prompt={prompt_idx}, attempt={attempt_idx}")
                group.append(results_by_key[key])
            outputs.append(group)
        return outputs
