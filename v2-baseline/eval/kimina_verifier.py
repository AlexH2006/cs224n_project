"""
Verify Lean 4 code via Kimina Lean Server (HTTP).

POSTs each code to KIMINA_URL/verify and returns a result dict in the same
shape as the local Lean verifier (pass, complete, errors, sorries, verify_time)
so step2_compile can use it interchangeably.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Optional


DEFAULT_KIMINA_URL = os.environ.get("KIMINA_VERIFY_URL", "http://localhost:8000")


def _kimina_result_to_compilation(result_item: dict, wall_s: float) -> dict:
    """Convert one Kimina /verify result entry to step2 compilation_result shape."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    sorries: list[Any] = []

    if result_item.get("error"):
        errors = [{"data": result_item["error"]}]
        return {
            "pass": False,
            "complete": False,
            "errors": errors,
            "sorries": sorries,
            "warnings": warnings,
            "system_errors": result_item["error"],
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
        "system_errors": None,
        "verify_time": wall_s,
    }


def verify_one(
    lean_code: str,
    base_url: str = DEFAULT_KIMINA_URL,
    timeout: int = 300,
    api_key: Optional[str] = None,
) -> dict:
    """
    POST one Lean 4 code to Kimina /verify; return compilation_result-shaped dict.
    """
    url = base_url.rstrip("/") + "/verify"
    payload = {
        "codes": [{"custom_id": "1", "proof": lean_code}],
        "infotree_type": "original",
        "timeout": timeout,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.time()
    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout + 10) as f:
            raw = f.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        wall_s = time.time() - start
        try:
            body_err = e.read().decode("utf-8")
        except Exception:
            body_err = str(e)
        return _kimina_result_to_compilation({"error": f"HTTP {e.code}: {body_err}"}, wall_s)
    except urllib.error.URLError as e:
        wall_s = time.time() - start
        return _kimina_result_to_compilation({"error": f"Request failed: {e.reason}"}, wall_s)
    except (TimeoutError, OSError) as e:
        wall_s = time.time() - start
        return _kimina_result_to_compilation({"error": str(e)}, wall_s)

    wall_s = time.time() - start
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return _kimina_result_to_compilation({"error": f"Invalid JSON: {e}"}, wall_s)

    results = data.get("results") or []
    if not results:
        return _kimina_result_to_compilation({"error": "Kimina returned no results"}, wall_s)
    return _kimina_result_to_compilation(results[0], wall_s)
