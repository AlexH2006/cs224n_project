"""
Thin transport layer: call Kimina Lean Server (HTTP) and return a Goedel-style result dict.

TLDR: Single responsibility — POST full Lean code to Kimina /verify, then convert the
response to the same Goedel-style dict that _goedel_to_verify_result() consumes. No
verification logic lives here; this is only "run via HTTP and normalize shape."
Used by: local_lean_verifier.verify(..., backend="kimina").
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional


# Default base URL when using Kimina backend (Docker on local device).
_DEFAULT_KIMINA_BASE_URL = "http://localhost:8000"


def _kimina_result_to_goedel_format(result_item: dict, wall_s: float) -> dict:
    """
    Convert one Kimina /verify result entry to Goedel-style dict.

    Kimina result: { "custom_id", "error": str|null, "response": { "messages", "sorries", ... } }.
    Goedel-style:  { "pass", "complete", "errors", "sorries", "warnings", "system_errors", "verify_time" }.
    Output shape must match what _goedel_to_verify_result() expects (errors as list of dicts with "data").
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    sorries: list[Any] = []
    system_errors: Optional[str] = None

    if result_item.get("error"):
        system_errors = result_item["error"]
        errors = [{"data": system_errors}]
        return {
            "pass": False,
            "complete": False,
            "errors": errors,
            "sorries": sorries,
            "warnings": warnings,
            "system_messages": "",
            "system_errors": system_errors,
            "verify_time": wall_s,
        }

    resp = result_item.get("response") or {}
    messages = resp.get("messages") or []
    sorries = list(resp.get("sorries") or [])

    for m in messages:
        if not isinstance(m, dict):
            continue
        severity = m.get("severity")
        data = m.get("data", str(m))
        if severity == "error":
            errors.append({"data": data})
        elif severity == "warning":
            warnings.append({"data": data})

    pass_ = not errors
    complete = (
        pass_
        and not sorries
        and not any(
            "declaration uses 'sorry'" in (w.get("data") or "")
            or "failed" in (w.get("data") or "")
            for w in warnings
        )
    )

    return {
        "pass": pass_,
        "complete": complete,
        "errors": errors,
        "sorries": sorries,
        "warnings": warnings,
        "system_messages": "",
        "system_errors": system_errors,
        "verify_time": wall_s,
    }


def run_kimina_verify(
    lean_code: str,
    base_url: str = _DEFAULT_KIMINA_BASE_URL,
    timeout: int = 300,
    api_key: Optional[str] = None,
) -> dict:
    """
    POST full Lean code to Kimina /verify and return a Goedel-style result dict.

    Does not interpret verification semantics; only performs HTTP and shape conversion
    so the caller can pass the result to _goedel_to_verify_result() unchanged.
    """
    url = base_url.rstrip("/") + "/verify"
    payload = {
        "codes": [{"custom_id": "1", "proof": lean_code}],
        "timeout": timeout,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.time()
    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout + 5) as f:
            raw = f.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        wall_s = time.time() - start
        try:
            body_err = e.read().decode("utf-8")
        except Exception:
            body_err = str(e)
        return _kimina_result_to_goedel_format(
            {"error": f"HTTP {e.code}: {body_err}"},
            wall_s,
        )
    except urllib.error.URLError as e:
        wall_s = time.time() - start
        # Connection refused (Errno 61/111) = nothing listening at base_url: Kimina Lean Server
        # (e.g. Docker) not running or wrong port. Start with e.g.:
        #   docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
        # or set LEAN_VERIFY_BACKEND=local to use lake exe repl instead.
        return _kimina_result_to_goedel_format(
            {"error": f"Request failed: {e.reason}"},
            wall_s,
        )
    except (TimeoutError, OSError) as e:
        wall_s = time.time() - start
        return _kimina_result_to_goedel_format(
            {"error": f"Verification timed out or connection error: {e}"},
            wall_s,
        )

    wall_s = time.time() - start
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return _kimina_result_to_goedel_format(
            {"error": f"Invalid JSON from server: {e}"},
            wall_s,
        )

    results = data.get("results") or []
    if not results:
        return _kimina_result_to_goedel_format(
            {"error": "Kimina returned no results"},
            wall_s,
        )
    return _kimina_result_to_goedel_format(results[0], wall_s)
