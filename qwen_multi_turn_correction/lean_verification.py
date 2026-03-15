"""
TLDR: VerifyResult contract and Kimina HTTP response parsing.

VerifyResult is the shared dict shape returned by every verification path.
parse_kimina_response() converts the raw Kimina /verify JSON into that shape.
verification_error_result() builds a failure VerifyResult for exceptions.

Used by: local_lean_verifier.py.
Adapted from: sdpo_modal_local_verify_kimina/lean_verification.py (self-contained copy).
"""

from typing import Any, TypedDict


class VerifyResult(TypedDict, total=False):
    """Result of verifying a Lean 4 source string. All keys optional for partial results."""

    success: bool        # True iff no compilation errors
    complete: bool       # True iff success AND no sorry
    has_sorry: bool      # sorry in code or in sorries list
    feedback: str        # Newline-joined error strings (empty on success)
    errors: list         # List of error strings
    messages: list       # Raw message strings from verifier
    sorries: list        # Sorry positions from verifier
    source: str          # "kimina"
    is_server_error: bool
    debug: dict[str, Any]


def parse_kimina_response(
    raw_result: dict,
    lean_code: str,
    verifier_wall_s: float = 0.0,
) -> dict:
    """
    Convert a raw Kimina /verify HTTP response dict into a VerifyResult-shaped dict.

    Handles: server error dict, empty results list, and normal results[0] payload.
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

    if not raw_result.get("results"):
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

    has_error = bool(errors) or status == "error"
    has_sorry = bool(sorries) or code_has_sorry

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
    """Build a VerifyResult-shaped failure dict for an exception or connection error."""
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
        "debug": {
            "verifier_wall_s": round(verifier_wall_s, 3),
            "error": message[:500],
        },
    }
