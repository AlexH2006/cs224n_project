"""
TLDR: Single verify() entry point that calls the Kimina Lean Server over HTTP
and returns a VerifyResult-shaped dict.

This module is the only verification interface for the qwen_eval pipeline.
It wraps kimina_transport.verify_via_kimina() and converts its output to the
VerifyResult shape defined in lean_verification.py.

Retry logic lives in the caller (modal_app.py) to keep this module stateless.

Used by: modal_app.py (verification phase, local driver).
"""

import time
from typing import Optional

from qwen_eval.kimina_transport import verify_via_kimina
from qwen_eval.lean_verification import verification_error_result


def verify(
    lean_code: str,
    kimina_url: str = "http://localhost:8000",
    timeout: int = 60,
    api_key: Optional[str] = None,
) -> dict:
    """
    Verify a Lean 4 source string using the Kimina Lean Server.

    Args:
        lean_code:   Complete Lean 4 source (imports + theorem + proof).
        kimina_url:  Base URL of the Kimina server (e.g. "http://localhost:8000").
        timeout:     HTTP timeout in seconds.
        api_key:     Optional Bearer token for authenticated Kimina endpoints.

    Returns:
        VerifyResult-shaped dict with keys: success, complete, has_sorry, feedback,
        errors, messages, sorries, source, is_server_error, debug.
    """
    start = time.time()
    try:
        raw = verify_via_kimina(lean_code, base_url=kimina_url, timeout=timeout, api_key=api_key)
    except Exception as e:
        return verification_error_result(
            lean_code,
            f"Unexpected error calling Kimina: {e}",
            is_server_error=True,
            verifier_wall_s=time.time() - start,
        )

    wall_s = time.time() - start
    code_has_sorry = "sorry" in lean_code.lower()
    system_errors = raw.get("system_errors")
    errors_raw = raw.get("errors") or []
    sorries = raw.get("sorries") or []

    # Flatten errors from Kimina's {"data": "..."} dicts to plain strings.
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
