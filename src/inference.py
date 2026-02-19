import re
import random
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
from utils import *

# if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default="", type=str)
parser.add_argument('--model_path', default="/scratch/gpfs/yl7690/models/Translator_Qwen2.5-Coder-32B_numina_sonnet_130K_translator_Epoch2_LR1e-4", type=str)
parser.add_argument('--output_dir', default="/scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5/results/translator", type=str)
parser.add_argument('--split', default="none", type=str)
parser.add_argument('--n', default=32, type=int)
parser.add_argument("--max_model_len", default=131072, type=int)#16384
parser.add_argument('--inference_handler', type=str, choices=["dpskcot", "dpsknoncot", "kiminacot"])
parser.add_argument('--trunck', default=1, type=int)
parser.add_argument('--gpu', default=4, type=int)
parser.add_argument("--base_output_template", default="qwen", type=str)
parser.add_argument('--node', default=1, type=int)
parser.add_argument('--error_thres', default=True)
parser.add_argument('--temp',  default=1.0, type=float)

parser.add_argument('--correction_round', type=int, default=0,
                    help="0 for initial inference, >0 for correction round N.")
parser.add_argument('--previous_run_output_dir', type=str,
                    help="Path to output dir of previous run (for correction_round > 0).")

args = parser.parse_args()
seed = random.randint(1, 99999)

if args.correction_round == 0:
    assert args.input_path != "" # if not doing revision, should have input jsonl

actual_previous_run_output_dir = args.previous_run_output_dir # output dir, new time string or not
if args.correction_round > 0 and not actual_previous_run_output_dir:
    print(f"Info: --previous_run_output_dir not specified for correction round {args.correction_round}. "
            f"Assuming previous round's files are in current --output_dir: {args.output_dir}")
    actual_previous_run_output_dir = args.output_dir

model_name = args.model_path
hf_tokenizer_for_chat_template = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

if args.inference_handler == "dpskcot":
    handler=DeepSeekCoTHandler()
elif args.inference_handler == "dpsknoncot":
    handler=DeepSeekNonCoTHandler()
elif args.inference_handler == "kiminacot":
    handler=KiminaCoTHandler()
else:
    raise Exception

records_file_suffix = f"_corr{args.correction_round}" if args.correction_round > 0 else ""
output_file_path_records = os.path.join(args.output_dir, f'full_records{records_file_suffix}.json')
output_file_path_inference_codes = os.path.join(args.output_dir, f'to_inference_codes{records_file_suffix}.json')

items_for_llm_processing = []
if args.correction_round > 0:
    if not actual_previous_run_output_dir:  # Should have been set or exited if R0
        print("Error: previous_run_output_dir logic failed for correction round.")
        exit(1)
    items_for_llm_processing = load_data_for_correction(actual_previous_run_output_dir, args.correction_round, args.n, args.base_output_template)
else:  # Initial inference (Round 0)
    if not args.input_path:
        print("Error: --input_path is required for initial inference (correction_round == 0).")
        exit(1)
    initial_data_list = handler.load_split(args.input_path, args.split)

    for idata_orig in initial_data_list:
        origin_id = idata_orig.get("origin_problem_id", idata_orig.get('problem_id', idata_orig.get('name')))
        if not idata_orig.get("lean4_code"): continue
        for ij in range(args.n):
            item_for_attempt = idata_orig.copy() 
            item_for_attempt["origin_problem_id"] = origin_id
            item_for_attempt["problem_id"] = f"{origin_id}_g{ij}"  # Suffix for this specific attempt
            item_for_attempt["id_maps"] = [{"origin_problem_id": origin_id},
                                           {"generation_id": item_for_attempt["problem_id"]}]
            items_for_llm_processing.append(item_for_attempt)

if not items_for_llm_processing:
    print("No data available for LLM processing. Exiting.")
    exit(0)

input_chunks = handler.split_list_into_chunks(items_for_llm_processing, num_chunks=args.trunck)
print(
    f"Total items for LLM: {len(items_for_llm_processing)}, split into {len(input_chunks)} chunks for round {args.correction_round}.")

all_processed_records = []
all_inference_code_outputs = []

if args.node > 1:
    model = LLM(model=model_name, seed=seed, trust_remote_code=True, max_model_len=args.max_model_len, tensor_parallel_size=args.gpu, pipeline_parallel_size=args.node, distributed_executor_backend="ray")
else:
    model = LLM(model=model_name, seed=seed, trust_remote_code=True, max_model_len=args.max_model_len, tensor_parallel_size=args.gpu)

sampling_params = SamplingParams(
    temperature=args.temp,
    max_tokens=args.max_model_len,
    top_p=0.95,
    n=1,
)

for chunk_idx, current_chunk_input_items in enumerate(
        tqdm(input_chunks, desc=f"Processing Chunks (Round {args.correction_round})")):
    records = []
    for i, item_data in enumerate(tqdm(current_chunk_input_items,
        desc=f"Preparing data...")):
        item_data["lean4_code"] = item_data["lean4_code"].split(":= by")[0] + ":= by sorry"
        if args.correction_round > 0:
            error_str = get_error_str(
                item_data.get('compiled_code_that_failed_in_prev_round', ''),
                item_data.get('errors_for_compiled_code_from_prev_round', {}).get('errors', []),
                args.error_thres
            )
            prompt_str, messages_for_this = handler.generate_correction_prompt(
                lean4_code_original_stmt=item_data["lean4_code"],
                history_messages_from_prev_round=item_data.get("history_messages_from_prev_round_for_new_prompt",
                                                                []),
                prev_round_llm_raw_output=item_data.get("prev_round_llm_raw_output_for_new_prompt", ""),
                error_message_for_prev_round=error_str,
                tokenizer=hf_tokenizer_for_chat_template,
                current_correction_round_num=args.correction_round
            )
        else:  # Initial inference
            prompt_str, messages_for_this = handler.prover_inference(
                item_data["lean4_code"], hf_tokenizer_for_chat_template
            )
        num_tokens = len(hf_tokenizer_for_chat_template.tokenize(prompt_str))  
        # num_cot_tokens = len(hf_tokenizer_for_chat_template.tokenize(messages_for_this[1]["content"]))
        records.append({
            # "cot_token_nums": num_cot_tokens,
            "token_nums": num_tokens,
            "prompts_for_vllm": prompt_str,
            "messages_lists_for_current_prompts": messages_for_this,
            "current_chunk_input_items": item_data
        })

    df_med = pd.DataFrame(records)
    max_length = args.max_model_len * 3 / 4 # fixed to be Qwen

    to_process_df = df_med[df_med.token_nums <= max_length].reset_index(drop=True)
    print(F"In total {len(df_med)}, selected {len(to_process_df)} whose length is smaller than {max_length}")
    # import pdb; pdb.set_trace()
    vllm_outputs = model.generate(to_process_df.prompts_for_vllm, sampling_params)

    for i in range(len(to_process_df)):
        input_item = to_process_df.current_chunk_input_items[i].copy()  # Process a copy
        input_item["model_input"] = to_process_df.prompts_for_vllm[i]
        input_item["messages_history_for_this_attempt"] = to_process_df.messages_lists_for_current_prompts[i]

        llm_response_text = vllm_outputs[i].outputs[0].text
        input_item["model_output"] = llm_response_text
        extracted_code = handler.extrac_code(llm_response_text)
        if extracted_code == "None" or extracted_code is None:
            input_item["full_code"] = "None"
        else:
            input_item["full_code"] = handler.problem_check(input_item["lean4_code"], extracted_code)
        
        all_processed_records.append(input_item)
        all_inference_code_outputs.append({
            "problem_id": input_item["problem_id"],
            "origin_problem_id": input_item.get("origin_problem_id"),
            "id_maps": input_item.get("id_maps"),
            "lean4_code": input_item["lean4_code"],
            "model_input": input_item["model_input"],
            "messages_history_list": input_item["messages_history_for_this_attempt"],
            "model_output": input_item["model_output"],
            "full_code": input_item["full_code"]
        })
    
    print(F"Saving {chunk_idx}th trunk of round {args.correction_round} to {args.output_dir}")
    jsave(all_processed_records, output_file_path_records)
    jsave(all_inference_code_outputs, output_file_path_inference_codes)

print(f"Outputs saved: \n  Records: {output_file_path_records}\n  Inference Codes (JSONL): {output_file_path_inference_codes}")
print("Script finished.")
