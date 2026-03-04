class InferenceBackend:
    def generate(self, prompts, n, temperature, top_p, max_tokens, max_batch_size, model_name=None):
        raise NotImplementedError()
