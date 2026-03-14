#!/usr/bin/env python3
"""One-off: verify a Lean file with local Kimina Lean Server (localhost:8000)."""
import json
import sys
from pathlib import Path

# Project root for imports
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
from qwen_sdpo._verifier import verify

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else _root / "debug/lean_samples/mathd_numbertheory_435_check.lean"
    with open(path, "r") as f:
        lean_code = f.read()

    print("Verifying with Kimina (http://localhost:8000)...")
    result = verify(lean_code, kimina_url="http://localhost:8000", timeout=120)
    print(json.dumps(result, indent=2))
    if result.get("success"):
        print("\nResult: VALID (proof checks).")
    else:
        print("\nResult: INVALID or ERROR.")
        if result.get("feedback"):
            print("Feedback:", result["feedback"])

if __name__ == "__main__":
    main()
