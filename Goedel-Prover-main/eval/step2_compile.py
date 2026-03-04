import json
import sys
import multiprocessing as mp

import argparse

def main():
    # On macOS + newer Python versions, the default "spawn" start method can
    # fail for this scheduler graph due to pickling manager/process state.
    # Force "fork" for compatibility with this multiprocessing design.
    current_method = mp.get_start_method(allow_none=True)
    if current_method != "fork":
        mp.set_start_method("fork", force=True)

    from prover.lean.verifier import Lean4ServerScheduler

    parser = argparse.ArgumentParser()
    # 'results/test/to_inference_codes.json'
    parser.add_argument('--input_path', default="", type=str)
    # 'results/test/code_compilation.json'
    parser.add_argument('--output_path', default="", type=str)
    parser.add_argument('--cpu', default=64, type=int)
    args = parser.parse_args()

    input_file_path = args.input_path

    with open(input_file_path, 'r') as json_file:
        codes = json.load(json_file)

    print(f"[step2] Compiling {len(codes)} generated proofs with cpu={args.cpu}")
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=args.cpu, timeout=300, memory_limit=10, name='verifier')

    request_id_list = lean4_scheduler.submit_all_request([code["code"] for code in codes])
    outputs_list = []
    try:
        from tqdm import tqdm

        iterator = tqdm(request_id_list, desc="Lean compilation", unit="proof")
    except Exception:
        iterator = request_id_list
    for request_id in iterator:
        outputs_list.append(lean4_scheduler.get_request_outputs(request_id))
    lean4_scheduler.close()
    print("[step2] Compilation complete")

    assert len(outputs_list) == len(codes)
    ana_result = []
    for i in range(len(codes)):
        codes[i]["compilation_result"] = outputs_list[i]
        ana_result.append(
            {"name": codes[i]["name"],
             "compilation_result": outputs_list[i]["complete"]}
        )
    with open(args.output_path, 'w') as json_file:
        json.dump(codes, json_file, indent=4)


if __name__ == '__main__':
    main()
