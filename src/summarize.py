import pandas as pd
import numpy as np
import argparse
import re
parser = argparse.ArgumentParser()
#/scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5/results/new_pipe_minif2f/code_compilation.json 
parser.add_argument('--input_path',  type=str)
#/scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5/results/new_pipe_minif2f/compilation_summarize.json
parser.add_argument('--full_record_path',  type=str)
parser.add_argument('--output_dir',  type=str)
# parser.add_argument('--group',  type=bool, default=False)

# parser.add_argument('--round',  type=int, default=0)
parser.add_argument('--field', default="complete",choices=["complete", "pass"], type=str)
args = parser.parse_args()


input_file= args.input_path
df = pd.read_json(input_file)
df_full = pd.read_json(args.full_record_path)
ids_lookup = dict(zip(df_full.problem_id, df_full.id_maps))

import numpy as np

ids_num_ = np.unique(df_full.id_maps.apply(lambda x: len(x)))
assert len(ids_num_) == 1
ids_num = ids_num_[0]
first_element = df_full.id_maps[0]
# import pdb; pdb.set_trace()
# df["correct"] = df.apply(lambda row: int(  ((row["compilation_result"][args.field])) ), axis=1) #  and ("apply?" not in row["code"]) and
df["correct"] = df.apply(lambda row: int( ((row["compilation_result"][args.field])) and ("apply?" not in row["code"]) and ("exact?" not in row["code"])), axis=1) #   and

import os
os.makedirs(args.output_dir, exist_ok=True)

meta_result = []
name_list = []
for i in range(ids_num):
  names = [k for k, _ in first_element[i].items()]
  assert len(names) == 1
  name = names[0]
  name_list.append(name)
  df[name] =  df["name"].apply(lambda x: ids_lookup[x][i][name])
  df_grp = df[[name, "correct"]].groupby(name)["correct"].aggregate(["sum", "count"]).reset_index()
  df_grp.to_csv(F"{args.output_dir}/{name}_summarize.csv", index=False, header=True, sep='\t', quoting=1, na_rep='Missing')
  meta_result.append({
    "level": F"{name}", 
    "value": {
        "problem_num": len(df_grp),
        "solved_num": sum(df_grp["sum"]>0),
        "solved_ratio": F"{sum(df_grp['sum']>0) / len(df_grp) * 100: 2f}"
      }
  })
  
pd.DataFrame(meta_result).to_json(F"{args.output_dir}/meta_summarize.json", indent=4, orient="records")


# import pdb; pdb.set_trace()


# import pdb; pdb.set_trace()

# def extract_grp(x_str):
#   pattern = r'(.*?)(?=_g\d+$)'
#   return re.match(pattern, x_str).group(1) 


# if args.group:
#   df["name_group"] = df.name.apply(lambda x: extract_grp(x))
#   grp_field = "name_group" 
# else:
#   grp_field = "name"

# df_grp = df.groupby(grp_field)["correct"].sum()

# result = {
#   "total": len(df_grp),
#   "correct": sum(df_grp > 0),
#   "accuracy": F"{sum(df_grp > 0) / len(df_grp)  * 100:.2f}",
#   "field": args.field
# }
# import json
# with open(args.output_path, "w") as f:
#     json.dump(result, f)

# #
# df_grp.reset_index()[[grp_field, "correct"]].to_csv(args.output_path.replace(".json", ".csv"), index=False, header=True, sep='\t', quoting=1, na_rep='Missing')