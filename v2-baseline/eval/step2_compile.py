import json
import os
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="", type=str)
parser.add_argument('--output_path', default="", type=str)
parser.add_argument('--cpu', default=64, type=int)
parser.add_argument(
    '--verifier',
    default="local",
    choices=["local", "kimina"],
    help="Use local Lean (lake exe repl) or Kimina Docker server (HTTP)",
)
parser.add_argument(
    '--kimina_url',
    default=os.environ.get("KIMINA_VERIFY_URL", "http://localhost:8000"),
    type=str,
    help="Kimina server URL when --verifier kimina",
)
parser.add_argument('--timeout', default=300, type=int, help="Verification timeout per proof")
args = parser.parse_args()

input_file_path = args.input_path

with open(input_file_path, 'r') as json_file:
    codes = json.load(json_file)

if args.verifier == "kimina":
    from eval.kimina_verifier import verify_one
    print(f"[step2] Verifying {len(codes)} proofs via Kimina at {args.kimina_url}")
    outputs_list = []
    for i, item in enumerate(codes):
        result = verify_one(
            item["code"],
            base_url=args.kimina_url,
            timeout=args.timeout,
        )
        outputs_list.append(result)
        if (i + 1) % 10 == 0 or i == len(codes) - 1:
            print(f"  Verified {i + 1}/{len(codes)}")
else:
    from prover.lean.verifier import Lean4ServerScheduler
    lean4_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=args.cpu, timeout=args.timeout, memory_limit=10, name='verifier'
    )
    request_id_list = lean4_scheduler.submit_all_request([code["code"] for code in codes])
    outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
    lean4_scheduler.close()

assert len(outputs_list) == len(codes)
verifier_source = "kimina" if args.verifier == "kimina" else "local_lean"
ana_result = []
for i in range(len(codes)):
    codes[i]["compilation_result"] = outputs_list[i]
    ana_result.append(
        {"name": codes[i]["name"],
         "compilation_result": outputs_list[i]["complete"]}
    )
with open(args.output_path, 'w') as json_file:
    json.dump(codes, json_file, indent=4)

# Update logs.json with verification results (logs.json created at end of step1)
logs_path = os.path.join(os.path.dirname(args.output_path), "logs.json")
if os.path.exists(logs_path):
    with open(logs_path, 'r') as f:
        logs = json.load(f)
    comp_idx = 0
    for entry in logs:
        for att in entry["attempts"]:
            if comp_idx < len(codes):
                comp = codes[comp_idx]["compilation_result"]
                comp_idx += 1
                errors_raw = comp.get("errors", [])
                error_strings = [e.get("data", str(e)) if isinstance(e, dict) else str(e) for e in errors_raw]
                sorries_raw = comp.get("sorries", [])
                passed = comp.get("pass", False)
                complete = comp.get("complete", False)
                has_sorry = len(sorries_raw) > 0
                att["verification"] = {
                    "success": passed,
                    "complete": complete,
                    "has_sorry": has_sorry,
                    "feedback": "\n".join(error_strings) if error_strings else "",
                    "errors": error_strings,
                    "messages": [],
                    "sorries": [str(s) for s in sorries_raw],
                    "source": verifier_source,
                    "is_server_error": False,
                    "debug": {"verifier_wall_s": comp.get("verify_time", 0.0)},
                }
                att["success"] = complete
        best = None
        best_proof = None
        for i, att in enumerate(entry["attempts"]):
            if att.get("verification", {}).get("complete"):
                best = i
                best_proof = att.get("full_code")
                break
        entry["success"] = best is not None
        entry["best_attempt"] = best
        entry["best_proof"] = best_proof
    with open(logs_path, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"Updated {logs_path} with verification results")
