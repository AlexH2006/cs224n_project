import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

DEFAULT_MODAL_URL = os.environ.get(
    "MODAL_URL",
    "https://banksaj27--goedel-prover-modal-hf-generate.modal.run",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple wrapper for running Goedel-Prover with N problems and Pass@K."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="",
        help="Hugging Face model id/path. You can also pass --model.",
    )
    parser.add_argument("--dataset", default="datasets/minif2f.jsonl", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_problems", default=3, type=int)
    parser.add_argument("--pass_k", default=10, type=int)
    parser.add_argument("--model", dest="model_override", default="", type=str)
    parser.add_argument("--provider", choices=["local_vllm", "modal", "modal_workers"], default="modal_workers", type=str)
    parser.add_argument("--output_dir", default="results/simple_run", type=str)
    parser.add_argument("--cpu", default=4, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--field", choices=["complete", "pass"], default="complete", type=str)

    parser.add_argument("--modal_url", default=DEFAULT_MODAL_URL, type=str)
    parser.add_argument("--modal_token_env", default="MODAL_API_TOKEN", type=str)
    parser.add_argument("--max_batch_size", default=16, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--modal_timeout", default=300, type=int)
    parser.add_argument("--modal_app_name", default="goedel-prover-modal-workers", type=str)
    parser.add_argument("--modal_function_name", default="generate_one_attempt", type=str)
    return parser.parse_args()


def filter_by_split(record, split):
    if split == "none":
        return True
    try:
        return int(record.get("split")) == int(split)
    except Exception:
        return str(record.get("split")) == str(split)


def build_subset_dataset(input_path, split, num_problems):
    selected = []
    with open(input_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if filter_by_split(item, split):
                selected.append(item)
            if len(selected) >= num_problems:
                break
    if not selected:
        raise ValueError(
            f"No records found for split={split} in dataset={input_path}. "
            "Try --split none or a different split value."
        )
    if len(selected) < num_problems:
        print(
            f"Warning: requested {num_problems} problems, but only found {len(selected)}.",
            file=sys.stderr,
        )
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    with temp_file:
        for item in selected:
            temp_file.write(json.dumps(item) + "\n")
    return temp_file.name, len(selected)


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    start = time.time()
    subprocess.run(cmd, check=True)
    print(f"Finished in {time.time() - start:.1f}s")


def main():
    args = parse_args()
    model_name = args.model_override or args.model
    if not model_name:
        raise ValueError("Please pass a model id as a positional arg or with --model.")
    os.makedirs(args.output_dir, exist_ok=True)

    subset_path, actual_count = build_subset_dataset(
        input_path=args.dataset,
        split=args.split,
        num_problems=args.num_problems,
    )
    print(f"Subset dataset: {subset_path} ({actual_count} problems)")

    try:
        step1_cmd = [
            sys.executable,
            "-m",
            "eval.step1_inference",
            "--input_path",
            subset_path,
            "--model_path",
            model_name,
            "--output_dir",
            args.output_dir,
            "--split",
            "none",
            "--n",
            str(args.pass_k),
            "--gpu",
            str(args.gpu),
            "--provider",
            args.provider,
            "--max_batch_size",
            str(args.max_batch_size),
            "--temperature",
            str(args.temperature),
            "--top_p",
            str(args.top_p),
            "--max_tokens",
            str(args.max_tokens),
        ]
        if args.provider == "modal":
            if not args.modal_url:
                raise ValueError("--modal_url is required when --provider modal is used (or set MODAL_URL env var)")
            step1_cmd.extend(
                [
                    "--modal_url",
                    args.modal_url,
                    "--modal_token_env",
                    args.modal_token_env,
                    "--modal_timeout",
                    str(args.modal_timeout),
                ]
            )
        elif args.provider == "modal_workers":
            step1_cmd.extend(
                [
                    "--modal_app_name",
                    args.modal_app_name,
                    "--modal_function_name",
                    args.modal_function_name,
                ]
            )
        print("\n=== Stage 1/3: Inference ===")
        run_cmd(step1_cmd)

        input_file = os.path.join(args.output_dir, "to_inference_codes.json")
        compile_output_path = os.path.join(args.output_dir, "code_compilation.json")
        summarize_output_path = os.path.join(args.output_dir, "compilation_summarize.json")

        print("\n=== Stage 2/3: Lean Compilation ===")
        run_cmd(
            [
                sys.executable,
                "-m",
                "eval.step2_compile",
                "--input_path",
                input_file,
                "--output_path",
                compile_output_path,
                "--cpu",
                str(args.cpu),
            ]
        )
        print("\n=== Stage 3/3: Summarize ===")
        run_cmd(
            [
                sys.executable,
                "-m",
                "eval.step3_summarize_compile",
                "--input_path",
                compile_output_path,
                "--output_path",
                summarize_output_path,
                "--field",
                args.field,
            ]
        )
    finally:
        if os.path.exists(subset_path):
            os.remove(subset_path)

    print("\nDone.")
    print(f"Summary: {os.path.join(args.output_dir, 'compilation_summarize.json')}")


if __name__ == "__main__":
    main()
