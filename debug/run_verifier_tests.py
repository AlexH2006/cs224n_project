#!/usr/bin/env python3
"""
Run all verifier-related tests in this folder.

Runs test_local_verify_pipeline.py and test_local_lean_verifier.py in sequence.
Must be run from repo root: python debug/run_verifier_tests.py
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEBUG_DIR = Path(__file__).resolve().parent


def main() -> int:
    scripts = [
        ("sdpo_modal_local_verify_kimina pipeline (smoke)", [sys.executable, str(DEBUG_DIR / "test_local_verify_pipeline.py")]),
        ("sdpo_modal local_lean_verifier (contract)", [sys.executable, str(DEBUG_DIR / "test_local_lean_verifier.py")]),
    ]
    for name, cmd in scripts:
        print(f"\n--- {name} ---")
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"FAILED: {name} (exit {result.returncode})", file=sys.stderr)
            return result.returncode
    print("\nAll verifier tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
