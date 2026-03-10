from vllm import LLM, SamplingParams


class VllmBackend:
    """Local vLLM inference backend (Linux + NVIDIA GPU only)."""

    def __init__(self, model_path: str, gpu: int = 1):
        self.model = LLM(
            model=model_path,
            seed=1,
            trust_remote_code=True,
            swap_space=8,
            tensor_parallel_size=gpu,
            max_model_len=4096,
        )

    def generate(
        self,
        prompts: list[str],
        n: int = 32,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_tokens: int = 2048,
        **kwargs,
    ) -> list[list[str]]:
        """Return list-of-lists: one list of n completion strings per prompt."""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
        )
        model_outputs = self.model.generate(prompts, sampling_params, use_tqdm=True)
        return [
            [output.text for output in request_output.outputs]
            for request_output in model_outputs
        ]
