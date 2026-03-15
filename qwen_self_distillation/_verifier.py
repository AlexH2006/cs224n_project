"""
TLDR: Self-contained Kimina Lean Server verification for SDPO.

Provides verify() — POST Lean code to Kimina and return a VerifyResult-shaped dict.
Inlines transport + error-handling logic from qwen_eval. No dependency on qwen_eval.

Used by: entrypoint._verify_with_retries.
"""

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Optional

_LOG = logging.getLogger(__name__)


def _normalize_result_item(result_item: dict, wall_s: float) -> dict:
    """
    Convert one Kimina /verify result entry into a normalized result dict.
    """
    errors: list[dict[str, Any]] = []
    sorries: list[Any] = []
    system_errors: Optional[str] = None

    if result_item.get("error"):
        system_errors = result_item["error"]
        return {
            "success": False,
            "complete": False,
            "errors": [{"data": system_errors}],
            "sorries": [],
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

    success = not errors
    complete = success and not sorries

    return {
        "success": success,
        "complete": complete,
        "errors": errors,
        "sorries": sorries,
        "system_errors": None,
        "wall_s": wall_s,
    }


def _verify_via_kimina(
    lean_code: str,
    base_url: str = "http://localhost:8000",
    timeout: int = 300,
    api_key: Optional[str] = None,
) -> dict:
    """POST Lean code to Kimina /verify endpoint. Returns normalized result dict."""
    url = base_url.rstrip("/") + "/verify"
    payload = json.dumps(
        {"codes": [{"custom_id": "1", "proof": lean_code}], "timeout": timeout},
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    start = time.time()

    def _err(msg: str) -> dict:
        return _normalize_result_item({"error": msg}, time.time() - start)

    try:
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        return _err(f"HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        return _err(f"Connection error: {e.reason}")
    except (TimeoutError, OSError) as e:
        return _err(f"Timeout or OS error: {e}")

    wall_s = time.time() - start
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return _err(f"Invalid JSON from server: {e}")

    results = data.get("results") or []
    if not results:
        return _err("Kimina returned no results")

    return _normalize_result_item(results[0], wall_s)


def verify(
    lean_code: str,
    kimina_url: str = "http://localhost:8000",
    timeout: int = 60,
    api_key: Optional[str] = None,
) -> dict:
    """
    Verify a Lean 4 source string using the Kimina Lean Server.

    Returns a VerifyResult-shaped dict with keys: success, complete, has_sorry,
    feedback, errors, messages, sorries, source, is_server_error, debug.
    """
    start = time.time()
    try:
        raw = _verify_via_kimina(
            lean_code, base_url=kimina_url, timeout=timeout, api_key=api_key
        )
    except Exception as e:
        wall_s = time.time() - start
        _LOG.warning("Unexpected error calling Kimina: %s", e)
        return {
            "success": False,
            "complete": False,
            "has_sorry": "sorry" in lean_code.lower(),
            "feedback": f"Unexpected error calling Kimina: {e}",
            "errors": [str(e)],
            "messages": [],
            "sorries": [],
            "source": "kimina",
            "is_server_error": True,
            "debug": {"verifier_wall_s": round(wall_s, 3), "error": str(e)[:500]},
        }

    wall_s = time.time() - start
    code_has_sorry = "sorry" in lean_code.lower()
    system_errors = raw.get("system_errors")
    errors_raw = raw.get("errors") or []
    sorries = raw.get("sorries") or []

    error_strings = []
    for e in errors_raw:
        if isinstance(e, dict):
            error_strings.append(e.get("data", str(e)))
        else:
            error_strings.append(str(e))

    if not error_strings and system_errors:
        error_strings = [system_errors]

    has_sorry = bool(sorries) or code_has_sorry
    success = raw.get("success", not error_strings)
    complete = raw.get("complete", success and not has_sorry)
    is_server_error = bool(system_errors) and not error_strings

    return {
        "success": success,
        "complete": complete,
        "has_sorry": has_sorry,
        "feedback": "\n".join(error_strings) if error_strings else "",
        "errors": error_strings,
        "messages": [],
        "sorries": [str(s) for s in sorries],
        "source": "kimina",
        "is_server_error": is_server_error,
        "debug": {"verifier_wall_s": round(wall_s, 3)},
    }
