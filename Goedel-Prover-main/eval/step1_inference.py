import argparse
import json
import os
import re

LEAN4_DEFAULT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


def parse_args():
    parser = argparse.ArgumentParser()
    # datasets/minif2f.jsonl
    parser.add_argument('--input_path', type=str)
    # Goedel-LM/Goedel-Prover-SFT
    parser.add_argument('--model_path', type=str)
    # results/test
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--split', default="none", type=str)
    parser.add_argument('--n', default=32, type=int)
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--provider', default='local_vllm', choices=['local_vllm', 'modal', 'modal_workers'], type=str)
    parser.add_argument('--max_batch_size', default=64, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--max_tokens', default=2048, type=int)
    parser.add_argument('--modal_url', default='', type=str)
    parser.add_argument('--modal_token_env', default='MODAL_API_TOKEN', type=str)
    parser.add_argument('--modal_timeout', default=300, type=int)
    parser.add_argument('--modal_app_name', default='goedel-prover-modal-workers', type=str)
    parser.add_argument('--modal_function_name', default='generate_one_attempt', type=str)
    return parser.parse_args()


def load_data(input_path, split):
    data_list = []
    with open(input_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if split == "none":
                data_list.append(data)
                continue
            try:
                int_split = int(split)
            except Exception:
                int_split = None
            if isinstance(int_split, int):
                if int(data["split"]) == int_split:
                    data_list.append(data)
            elif data["split"] == split:
                data_list.append(data)
    return data_list


def build_model_inputs(data_list):
    model_inputs = []
    for data in data_list:
        model_inputs.append(
            "Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n```lean4\n{header}{informal_prefix}{formal_statement}".format(
                header=data.get('header', LEAN4_DEFAULT_HEADER),
                informal_prefix=data.get('informal_prefix', str()),
                formal_statement=data['formal_statement'],
            )
        )
    return model_inputs


def extract_code(inputs):
    try:
        return re.search(r'```lean4\n(.*?)\n```', inputs, re.DOTALL).group(1)
    except Exception:
        return "None"


def build_backend(args):
    if args.provider == 'modal_workers':
        from eval.inference_backends.modal_workers import ModalWorkersBackend
        return ModalWorkersBackend(
            app_name=args.modal_app_name,
            function_name=args.modal_function_name,
        )
    if args.provider == 'modal':
        from eval.inference_backends.modal_http import ModalHttpBackend
        return ModalHttpBackend(
            base_url=args.modal_url,
            token_env=args.modal_token_env,
            timeout=args.modal_timeout,
        )
    from eval.inference_backends.local_vllm import LocalVllmBackend
    return LocalVllmBackend(model_name=args.model_path, gpu_count=args.gpu)


def validate_outputs(outputs, expected_prompt_count, expected_n):
    if len(outputs) != expected_prompt_count:
        raise ValueError(f"Expected {expected_prompt_count} prompt outputs, got {len(outputs)}")
    for output_group in outputs:
        if len(output_group) != expected_n:
            raise ValueError(f"Expected {expected_n} completions per prompt, got {len(output_group)}")


def main():
    args = parse_args()
    print(f"[step1] Loading dataset from {args.input_path} (split={args.split})")
    data_list = load_data(args.input_path, args.split)
    print(f"[step1] Loaded {len(data_list)} problems")
    model_inputs = build_model_inputs(data_list)
    print(f"[step1] Built prompts. Provider={args.provider}, pass@k={args.n}")
    backend = build_backend(args)
    model_outputs = backend.generate(
        prompts=model_inputs,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_batch_size=args.max_batch_size,
        model_name=args.model_path,
    )
    validate_outputs(model_outputs, expected_prompt_count=len(model_inputs), expected_n=args.n)
    print("[step1] Inference complete. Formatting outputs...")

    to_inference_codes = []
    for i in range(len(data_list)):
        data_list[i]["model_input"] = model_inputs[i]
        data_list[i]["model_outputs"] = model_outputs[i]
        data_list[i]["full_code"] = [extract_code(model_inputs[i] + output) for output in model_outputs[i]]
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
    print(f"[step1] Done. Generated {len(to_inference_codes)} candidate proofs.")


if __name__ == '__main__':
    main()
