import re
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--split', default="none", type=str)
parser.add_argument('--n', default=32, type=int)
parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--provider', default="vllm", choices=["vllm", "modal_workers"], type=str)
parser.add_argument('--app_name', default="goedel-prover-modal-workers", type=str)
parser.add_argument('--function_name', default="generate_n_for_prompt", type=str)
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--top_p', default=0.95, type=float)
parser.add_argument('--max_tokens', default=2048, type=int)
parser.add_argument('--max_batch_size', default=50, type=int)

args = parser.parse_args()

# --- Load dataset ---

data_path = args.input_path
data_list = []

with open(data_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        if args.split == "none":
            data_list.append(data)
        else:
            try:
                int_split = int(args.split)
            except:
                int_split = None
                pass
            if isinstance(int_split, int):
                if (int(data["split"]) == int(args.split)):
                    data_list.append(data)
            else:
                if ((data["split"]) == (args.split)):
                    data_list.append(data)

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"

model_inputs = []
for data in data_list:
        model_inputs.append("Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
                header=data.get('header', LEAN4_DEFAULT_HEADER),
                informal_prefix=data.get('informal_prefix', str()),
                formal_statement=data['formal_statement'],
            )
        )

# --- Run inference via backend ---

from eval.inference_backends import build_backend

backend = build_backend(args)
completions = backend.generate(
    prompts=model_inputs,
    n=args.n,
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.max_tokens,
    max_batch_size=args.max_batch_size,
)

assert len(completions) == len(model_inputs)

# --- Post-process ---

def extrac_code(inputs):
    try:
        return re.search(r'```lean4\n(.*?)\n```', inputs, re.DOTALL).group(1)
    except:
        return "None"

to_inference_codes = []
for i in range(len(data_list)):
    data_list[i]["model_input"] = model_inputs[i]
    data_list[i]["model_outputs"] = completions[i]
    data_list[i]["full_code"] = [extrac_code(model_inputs[i] + output) for output in completions[i]]
    if "problem_id" in data_list[i]:
        to_inference_codes += [{"name": data_list[i]["problem_id"], "code": code} for code in data_list[i]["full_code"]]
    else:
        to_inference_codes += [{"name": data_list[i]["name"], "code": code} for code in data_list[i]["full_code"]]

os.makedirs(args.output_dir, exist_ok=True)

output_file_path = f'{args.output_dir}/full_records.json'
print(f"Outputting to {output_file_path}")
with open(output_file_path, 'w') as json_file:
    json.dump(data_list, json_file, indent=4)

toinfer_file_path = f'{args.output_dir}/to_inference_codes.json'
print(f"Outputting to {toinfer_file_path}")
with open(toinfer_file_path, 'w') as json_file:
    json.dump(to_inference_codes, json_file, indent=4)

# Baseline-style logs.json (verification filled in after step2)
VERIFICATION_PLACEHOLDER = {
    "success": False,
    "complete": False,
    "has_sorry": False,
    "feedback": "(pending verification)",
    "errors": [],
    "messages": [],
    "sorries": [],
    "source": "pending",
    "is_server_error": False,
    "debug": {},
}
logs = []
for problem_idx, record in enumerate(data_list):
    problem_id = record.get("problem_id") or record.get("name", f"problem_{problem_idx}")
    prompt = record.get("model_input", "")
    raw_outputs = record.get("model_outputs", [])
    full_codes = record.get("full_code", [])
    problem_meta = {
        "id": problem_id,
        "problem_idx": problem_idx,
        "dataset": args.input_path or "",
        "split": args.split,
        "formal_statement": record.get("formal_statement", ""),
        "header": record.get("header", ""),
        "informal_stmt": record.get("informal_prefix", ""),
    }
    attempts = []
    for attempt_idx in range(len(raw_outputs)):
        raw_output = raw_outputs[attempt_idx] if attempt_idx < len(raw_outputs) else ""
        full_code = full_codes[attempt_idx] if attempt_idx < len(full_codes) else ""
        extracted = extrac_code(prompt + raw_output) if raw_output else (full_code or "")
        attempts.append({
            "attempt": attempt_idx,
            "prompt": prompt,
            "raw_output": raw_output,
            "extracted_block": extracted,
            "full_code": full_code,
            "num_tokens": 0,
            "verification": dict(VERIFICATION_PLACEHOLDER),
            "success": False,
        })
    logs.append({
        "problem": problem_meta,
        "attempts": attempts,
        "success": False,
        "best_attempt": None,
        "best_proof": None,
        "config": {
            "model_name": args.model_path,
            "dataset": args.input_path,
            "split": args.split,
            "pass_k": args.n,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "provider": args.provider,
        },
    })
logs_path = f'{args.output_dir}/logs.json'
print(f"Outputting to {logs_path}")
with open(logs_path, 'w') as f:
    json.dump(logs, f, indent=2)
