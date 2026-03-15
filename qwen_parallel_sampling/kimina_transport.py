"""
TLDR: HTTP transport layer — POST Lean code to the Kimina Lean Server and return
a normalized result dict.

Single responsibility: network I/O + JSON shape normalization. No verification
semantics live here. All error cases (HTTP error, connection refused, timeout,
bad JSON) are converted to the same dict shape so callers always get a dict back.

Start the Kimina server locally with Docker before running eval:
    docker run -p 8000:8000 projectnumina/kimina-lean-server:2.0.0

Used by: local_lean_verifier.py.
Adapted from: sdpo_modal_local_verify_kimina/kimina_transport.py (self-contained copy).
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional


def _normalize_result_item(result_item: dict, wall_s: float) -> dict:
    """
    Convert one Kimina /verify result entry into a normalized result dict.

    Kimina shape: {"custom_id", "error": str|null, "response": {"messages", "sorries", ...}}
    Output shape: {"success", "complete", "errors", "sorries", "warnings", "system_errors", "wall_s"}
    """
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    sorries: list[Any] = []
    system_errors: Optional[str] = None

    if result_item.get("error"):
        system_errors = result_item["error"]
        return {
            "success": False,
            "complete": False,
            "errors": [{"data": system_errors}],
            "sorries": [],
            "warnings": [],
            "system_errors": system_errors,
            "wall_s": wall_s,
        }

    resp = result_item.get("response") or {}
    messages = list(resp.get("messages") or [])
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

    success = not errors
    # complete: no errors, no sorries, and no "declaration uses 'sorry'" in warnings
    complete = success and not sorries and not any(
        "declaration uses 'sorry'" in (w.get("data") or "")
        for w in warnings
    )

    return {
        "success": success,
        "complete": complete,
        "errors": errors,
        "sorries": sorries,
        "warnings": warnings,
        "system_errors": None,
        "wall_s": wall_s,
    }


def _error_result(message: str, wall_s: float) -> dict:
    """Return a normalized failure dict for network/parse errors."""
    return _normalize_result_item({"error": message}, wall_s)


def verify_via_kimina(
    lean_code: str,
    base_url: str = "http://localhost:8000",
    timeout: int = 300,
    api_key: Optional[str] = None,
) -> dict:
    """
    POST full Lean code to the Kimina /verify endpoint.

    Returns a normalized result dict (always a dict, never raises).
    Caller is responsible for interpreting success/complete/errors.
    """
    url = base_url.rstrip("/") + "/verify"
    payload = json.dumps(
        {"codes": [{"custom_id": "1", "proof": lean_code}], "timeout": timeout},
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.time()
    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        wall_s = time.time() - start
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        return _error_result(f"HTTP {e.code}: {body}", wall_s)
    except urllib.error.URLError as e:
        # Connection refused → Kimina Docker not running.
        return _error_result(f"Connection error: {e.reason}", time.time() - start)
    except (TimeoutError, OSError) as e:
        return _error_result(f"Timeout or OS error: {e}", time.time() - start)

    wall_s = time.time() - start
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return _error_result(f"Invalid JSON from server: {e}", wall_s)

    results = data.get("results") or []
    if not results:
        return _error_result("Kimina returned no results", wall_s)

    return _normalize_result_item(results[0], wall_s)
