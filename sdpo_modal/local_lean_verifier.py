"""
Local Lean 4 verification using Goedel-Prover's lake exe repl. Produces VerifyResult-shaped dict for sdpo_modal.

TLDR: verify(lean_code) runs `lake exe repl` in a mathlib4 workspace (no Kimina, no HTTP).
Maps the REPL JSON output to the same VerifyResult contract used by trainer_core.
Used by: entrypoint_local_verify (local-verify pipeline). Requires: elan/lake, mathlib4 built.
"""

import json
import os
import subprocess
import tempfile
import time
import traceback
from typing import Optional

from sdpo_modal.lean_verification import VerifyResult, verification_error_result


# Default paths: assume repo layout is 224n_project/sdpo_modal/ and 224n_project/Goedel-Prover-main/mathlib4/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_DEFAULT_LEAN_WORKSPACE = os.path.join(_REPO_ROOT, "Goedel-Prover-main", "mathlib4")
_HOME = os.path.expanduser("~")
_DEFAULT_LAKE_PATH = os.path.join(_HOME, ".elan", "bin", "lake")


def _run_goedel_style_verify(
    code: str,
    lake_path: str,
    lean_workspace: str,
    timeout: int,
) -> dict:
    """
    Run Lean 4 REPL (lake exe repl) and return Goedel-style result dict.
    Logic adapted from Goedel-Prover-main/prover/lean/verifier.py verify_lean4_file().
    Does not use ast_parser; only pass/complete/errors/sorries/warnings.
    """
    command = {"cmd": code, "allTactics": False, "ast": False, "tactics": False, "premises": False}
    message_str = json.dumps(command, ensure_ascii=False)
    start_time = time.time()
    system_messages = ""

    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as temp_file:
            temp_file.write(message_str + "\r\n\r\n")
            temp_file.seek(0)
            proc = subprocess.run(
                [lake_path, "exe", "repl"],
                stdin=temp_file,
                capture_output=True,
                text=True,
                cwd=lean_workspace,
                timeout=timeout,
            )
        raw = proc.stdout
        if not raw or not raw.strip():
            return {
                "pass": False,
                "complete": False,
                "errors": [proc.stderr or "Empty REPL output"],
                "sorries": [],
                "warnings": [],
                "system_errors": proc.stderr,
                "system_messages": system_messages,
                "verify_time": time.time() - start_time,
            }
        result = json.loads(raw)
    except subprocess.TimeoutExpired:
        return {
            "pass": False,
            "complete": False,
            "errors": ["Verification timed out"],
            "sorries": [],
            "warnings": [],
            "system_errors": "Timeout",
            "system_messages": system_messages,
            "verify_time": time.time() - start_time,
        }
    except Exception:
        return {
            "pass": False,
            "complete": False,
            "errors": [traceback.format_exc()],
            "sorries": [],
            "warnings": [],
            "system_errors": traceback.format_exc(),
            "system_messages": system_messages,
            "verify_time": time.time() - start_time,
        }

    # Normalize to Goedel-style keys
    messages = result.get("messages", [])
    errors = [m for m in messages if isinstance(m, dict) and m.get("severity") == "error"]
    warnings = [m for m in messages if isinstance(m, dict) and m.get("severity") == "warning"]
    sorries = result.get("sorries", [])

    out = {
        "sorries": sorries,
        "errors": errors,
        "warnings": warnings,
        "system_messages": system_messages,
        "system_errors": None,
        "verify_time": time.time() - start_time,
    }
    out["pass"] = not out["errors"]
    out["complete"] = (
        out["pass"]
        and not out["sorries"]
        and not any(
            "declaration uses 'sorry'" in (w.get("data") or "")
            or "failed" in (w.get("data") or "")
            for w in out["warnings"]
        )
    )
    return out


def _goedel_to_verify_result(goedel: dict, lean_code: str, verifier_wall_s: float) -> dict:
    """Map Goedel-style result to VerifyResult (success, complete, has_sorry, feedback, errors, source)."""
    errors_raw = goedel.get("errors") or []
    if not errors_raw and goedel.get("system_errors"):
        errors_raw = [{"data": goedel.get("system_errors", "Unknown error")}]
    error_strings = []
    for e in errors_raw:
        if isinstance(e, dict):
            error_strings.append(e.get("data", str(e)))
        else:
            error_strings.append(str(e))

    sorries = goedel.get("sorries") or []
    warnings = goedel.get("warnings") or []
    has_sorry = len(sorries) > 0 or "sorry" in lean_code.lower()
    if not has_sorry:
        for w in warnings:
            if isinstance(w, dict) and ("sorry" in (w.get("data") or "") or "failed" in (w.get("data") or "")):
                has_sorry = True
                break

    return {
        "success": bool(goedel.get("pass", False)),
        "complete": bool(goedel.get("complete", False)),
        "has_sorry": has_sorry,
        "feedback": "\n".join(error_strings) if error_strings else "",
        "errors": error_strings,
        "messages": [],
        "sorries": [str(s) for s in sorries],
        "source": "local_lean",
        "is_server_error": False,
        "debug": {"verifier_wall_s": round(verifier_wall_s, 3)},
    }


def verify(
    lean_code: str,
    timeout: int = 300,
    lake_path: Optional[str] = None,
    lean_workspace: Optional[str] = None,
) -> dict:
    """
    Verify a single Lean 4 source string using local lake exe repl. Returns VerifyResult-shaped dict.

    Requires: lake on PATH or at lake_path; lean_workspace is the mathlib4 directory (with Lakefile).
    """
    lake_path = lake_path or os.environ.get("LAKE_PATH", _DEFAULT_LAKE_PATH)
    lean_workspace = lean_workspace or os.environ.get("LEAN_WORKSPACE", _DEFAULT_LEAN_WORKSPACE)
    lean_workspace = os.path.abspath(lean_workspace)

    start = time.time()
    try:
        goedel = _run_goedel_style_verify(lean_code, lake_path, lean_workspace, timeout)
        wall_s = time.time() - start
        return _goedel_to_verify_result(goedel, lean_code, wall_s)
    except FileNotFoundError as e:
        out = verification_error_result(
            lean_code,
            f"Lean/lake not found: {e}. Set LAKE_PATH and ensure LEAN_WORKSPACE points to mathlib4.",
            is_server_error=True,
            verifier_wall_s=time.time() - start,
        )
        out["source"] = "local_lean"
        return out
    except Exception as e:
        out = verification_error_result(
            lean_code,
            str(e),
            is_server_error=True,
            verifier_wall_s=time.time() - start,
        )
        out["source"] = "local_lean"
        return out


def get_default_workspace() -> str:
    """Return default mathlib4 workspace path (for docs and tests)."""
    return os.environ.get("LEAN_WORKSPACE", _DEFAULT_LEAN_WORKSPACE)
