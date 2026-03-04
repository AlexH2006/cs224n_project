from vllm import LLM, SamplingParams

from eval.inference_backends.base import InferenceBackend


class LocalVllmBackend(InferenceBackend):
    def __init__(self, model_name, gpu_count=1):
        self.model = LLM(
            model=model_name,
            seed=1,
            trust_remote_code=True,
            swap_space=8,
            tensor_parallel_size=gpu_count,
            max_model_len=4096,
        )

    def generate(self, prompts, n, temperature, top_p, max_tokens, max_batch_size, model_name=None):
        del max_batch_size
        del model_name
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
        )
        outputs = self.model.generate(
            prompts,
            sampling_params,
            use_tqdm=True,
        )
        return [[sample.text for sample in output.outputs] for output in outputs]
