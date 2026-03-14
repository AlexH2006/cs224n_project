"""
Lean 4 verification: result contract and Kimina response parsing.

TLDR: Single module for (1) VerifyResult type, (2) parsing Kimina server JSON
into that shape. The main flow calls a verify_fn(lean_code) -> dict; Modal's
LeanVerifier uses this module to produce that dict. Used by: modal_app (LeanVerifier),
trainer_core (via verify_fn), __init__.
"""

from typing import Any, TypedDict


class VerifyResult(TypedDict, total=False):
    """Result of verifying a Lean 4 code string. All keys optional for partial results."""

    success: bool
    complete: bool
    has_sorry: bool
    feedback: str
    errors: list
    messages: list
    sorries: list
    source: str
    is_server_error: bool
    is_truncated: bool
    debug: dict[str, Any]


def parse_kimina_response(
    raw_result: dict,
    lean_code: str,
    verifier_wall_s: float = 0.0,
) -> dict:
    """Turn raw Kimina /verify response into a VerifyResult-shaped dict.

    Handles: server error dict, results[0] with messages/sorries/status,
    and unexpected format. Call this from any verifier that talks to Kimina HTTP API.
    """
    debug = {"verifier_wall_s": round(verifier_wall_s, 3)}
    code_has_sorry = "sorry" in lean_code.lower()

    if "error" in raw_result:
        return {
            "success": False,
            "complete": False,
            "has_sorry": code_has_sorry,
            "feedback": f"Kimina server error: {raw_result['error']}",
            "errors": [raw_result["error"]],
            "messages": [],
            "sorries": [],
            "source": "kimina",
            "is_server_error": raw_result.get("is_server_error", False),
            "debug": debug,
        }

    if "results" not in raw_result or len(raw_result["results"]) == 0:
        return {
            "success": False,
            "complete": False,
            "has_sorry": code_has_sorry,
            "feedback": "Unexpected response format from Kimina server",
            "errors": ["Unexpected response format"],
            "messages": [],
            "sorries": [],
            "source": "kimina",
            "is_server_error": True,
            "debug": debug,
        }

    r = raw_result["results"][0]
    resp = r.get("response") or {}
    messages = resp.get("messages", []) or r.get("messages", []) or []
    sorries = resp.get("sorries", []) or r.get("sorries", []) or []
    status = r.get("status", "")

    errors = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        msg_text = msg.get("data", str(msg))
        if msg.get("severity") == "error":
            errors.append(msg_text)
        elif "unsolved goals" in (msg_text or "").lower():
            errors.append(msg_text or "unsolved goals")

    has_error = len(errors) > 0 or status == "error"
    has_sorry = len(sorries) > 0 or code_has_sorry

    return {
        "success": not has_error,
        "complete": not has_error and not has_sorry,
        "has_sorry": has_sorry,
        "feedback": "\n".join(errors) if errors else "",
        "errors": errors,
        "messages": [str(m) for m in messages],
        "sorries": [str(s) for s in sorries],
        "source": "kimina",
        "is_server_error": False,
        "debug": debug,
    }


def verification_error_result(
    lean_code: str,
    message: str,
    is_server_error: bool = True,
    verifier_wall_s: float = 0.0,
) -> dict:
    """Build a VerifyResult-shaped dict for a verification exception or generic error."""
    return {
        "success": False,
        "complete": False,
        "has_sorry": "sorry" in lean_code.lower(),
        "feedback": message,
        "errors": [message],
        "messages": [],
        "sorries": [],
        "source": "kimina",
        "is_server_error": is_server_error,
        "debug": {"verifier_wall_s": round(verifier_wall_s, 3), "error": message[:500]},
    }
