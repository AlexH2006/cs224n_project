import json
import sys

import os

import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir) 

from lean_compiler.repl_scheduler import scheduler



import argparse

import random

import numpy as np

def split_list_randomly(lst, k):
    random.shuffle(lst)  # Shuffle the list randomly
    return list(map(list, np.array_split(lst, k)))  # Split into k approximately equal parts

def handle(text):
    lines = text.split('\n')

    filtered_lines = [line for line in lines if not (
            line.strip().startswith('import') or
            line.strip().startswith('set_option') or
            line.strip().startswith('open')
    )]

    return '\n'.join(filtered_lines)


parser = argparse.ArgumentParser()
# 'results/test/to_inference_codes.json'
parser.add_argument('--input_path', default="", type=str)
# 'results/test/code_compilation.json'
parser.add_argument('--output_path', default="", type=str)
# parser.add_argument('--output_path', default="example_data/o1_sorried_output.json", type=str)
parser.add_argument('--cpu', default=64, type=int)
args = parser.parse_args()

input_file_path = args.input_path

with open(input_file_path, 'r') as json_file:
    codes = json.load(json_file)


code_df = pd.DataFrame(codes)
# sub_df = code_df[code_df.full_code.apply(lambda x: "theorem" in x)].reset_index(drop=True).copy()
sub_df = code_df

if "problem_id" in sub_df.columns:
    sub_df["name"] = sub_df["problem_id"]
else:
    sub_df["problem_id"] = sub_df["name"]
if "full_code" in sub_df.columns:
    sub_df["code"] = sub_df["full_code"].apply(handle)
codes = sub_df[["name", "code", "problem_id"]].to_dict(orient='records')

random.shuffle(codes)

outputs_list = scheduler(codes, num_workers = args.cpu)

with open(args.output_path, 'w') as json_file:
    json.dump(outputs_list, json_file, indent=4)