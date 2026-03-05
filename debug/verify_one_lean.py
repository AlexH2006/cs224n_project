#!/usr/bin/env python3
"""
Verify a single Lean 4 snippet against Kimina Lean Server (HTTP API).
Sample .lean files live in debug/lean_samples/ (use -f to point at them).

Usage:
  python verify_one_lean.py [--server-url URL] [--file FILE]
  echo '<lean code>' | python verify_one_lean.py
  python verify_one_lean.py --file proof.lean
Server must be running, e.g.:
  docker run --rm -p 8000:8000 projectnumina/kimina-lean-server:2.0.0
"""
import argparse
import json
import sys
import httpx


def verify(lean_code: str, server_url: str = "http://localhost:8000", timeout: float = 120.0) -> dict:
    """POST to /verify and return raw JSON response."""
    url = f"{server_url.rstrip('/')}/verify"
    payload = {
        "codes": [{"custom_id": "1", "proof": lean_code}],
        "infotree_type": "original",
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(description="Verify one Lean snippet with Kimina server")
    parser.add_argument("--server-url", default="http://localhost:8000", help="Kimina server URL")
    parser.add_argument("--file", "-f", help="Read Lean code from file (default: stdin)")
    parser.add_argument("--output", "-o", help="Save raw JSON response to file")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r") as f:
            lean_code = f.read()
    else:
        lean_code = sys.stdin.read()

    if not lean_code.strip():
        print("No Lean code provided.", file=sys.stderr)
        sys.exit(1)

    print("Verifying with Kimina Lean Server...", flush=True)
    try:
        result = verify(lean_code, server_url=args.server_url, timeout=args.timeout)
    except httpx.ConnectError as e:
        print(f"Connection error: {e}", file=sys.stderr)
        print("Is the server running? e.g. docker run --rm -p 8000:8000 projectnumina/kimina-lean-server:2.0.0", file=sys.stderr)
        sys.exit(2)
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(4)

    # Save raw response to file if requested
    if args.output:
        with open(args.output, "w") as out:
            json.dump(result, out, indent=2)
        print(f"Raw response saved to {args.output}", file=sys.stderr)

    # Pretty-print and interpret
    print(json.dumps(result, indent=2))

    if "error" in result:
        print("\nResult: FAIL (server error)", file=sys.stderr)
        sys.exit(5)

    if "results" in result and len(result["results"]) > 0:
        r = result["results"][0]
        resp = r.get("response") or {}
        status = r.get("status", "")
        messages = resp.get("messages", []) or r.get("messages", []) or []
        sorries = resp.get("sorries", []) or r.get("sorries", []) or []
        errors = []
        for msg in messages:
            if isinstance(msg, dict) and msg.get("severity") == "error":
                errors.append(msg.get("data", str(msg)))
            if isinstance(msg, dict) and "unsolved goals" in (msg.get("data") or "").lower():
                errors.append(msg.get("data") or "unsolved goals")

        if errors:
            print("\nResult: FAIL", file=sys.stderr)
            for e in errors[:5]:
                print(f"  {e}", file=sys.stderr)
            sys.exit(6)
        if sorries or "sorry" in lean_code:
            print("\nResult: OK (compiles, but contains sorry)", file=sys.stderr)
            sys.exit(0)
        print("\nResult: OK (complete, no errors)", file=sys.stderr)
        sys.exit(0)

    print("\nResult: Unknown response format", file=sys.stderr)
    sys.exit(7)


if __name__ == "__main__":
    main()
