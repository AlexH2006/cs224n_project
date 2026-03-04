import json
import os
import ssl
from urllib import request
from urllib.error import HTTPError, URLError

from eval.inference_backends.base import InferenceBackend


class ModalHttpBackend(InferenceBackend):
    def __init__(self, base_url, token_env='MODAL_API_TOKEN', timeout=300):
        if not base_url:
            raise ValueError("--modal_url is required when --provider modal is used")
        self.base_url = base_url.rstrip('/')
        self.token = os.environ.get(token_env, '')
        self.timeout = timeout

    def _build_candidate_urls(self, path):
        trimmed = self.base_url.rstrip('/')
        path_part = path if path.startswith('/') else f'/{path}'
        if trimmed.endswith(path_part):
            return [trimmed]
        return [f"{trimmed}{path_part}", trimmed]

    def _post_json(self, path, payload):
        body = json.dumps(payload).encode('utf-8')
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = f"Bearer {self.token}"
        context = None
        # On some macOS Python installs, default trust store is missing/intermittent.
        # Prefer certifi if present to avoid SSL certificate verification failures.
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            context = ssl.create_default_context()

        last_error = None
        for endpoint in self._build_candidate_urls(path):
            req = request.Request(endpoint, data=body, headers=headers, method='POST')
            try:
                with request.urlopen(req, timeout=self.timeout, context=context) as response:
                    content = response.read().decode('utf-8')
                return json.loads(content)
            except HTTPError as exc:
                # Try alternate URL shape on 404s.
                if exc.code == 404:
                    last_error = exc
                    continue
                raise
            except URLError as exc:
                msg = str(exc)
                if "CERTIFICATE_VERIFY_FAILED" in msg:
                    raise RuntimeError(
                        "SSL verification failed when calling Modal. "
                        "Install a CA bundle with `python -m pip install certifi` and retry."
                    ) from exc
                raise
        if last_error is not None:
            raise RuntimeError(
                f"Modal endpoint returned 404 for URL forms derived from base URL '{self.base_url}'. "
                "Use the full web endpoint URL from `modal deploy` output."
            ) from last_error
        raise RuntimeError("Modal request failed for an unknown reason.")

    def generate(self, prompts, n, temperature, top_p, max_tokens, max_batch_size, model_name=None):
        all_outputs = []
        batch_starts = list(range(0, len(prompts), max_batch_size))
        try:
            from tqdm import tqdm

            iterator = tqdm(batch_starts, desc="Modal inference batches", unit="batch")
        except Exception:
            iterator = batch_starts
            print(f"Modal inference batches: {len(batch_starts)} total")

        for start in iterator:
            prompt_batch = prompts[start:start + max_batch_size]
            payload = {
                "prompts": prompt_batch,
                "n": n,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if model_name:
                payload["model_name"] = model_name
            response = self._post_json('/generate', payload)
            outputs = response.get("outputs", [])
            if len(outputs) != len(prompt_batch):
                raise ValueError(
                    f"Modal response size mismatch: expected {len(prompt_batch)} groups, got {len(outputs)}"
                )
            all_outputs.extend(outputs)
        return all_outputs
