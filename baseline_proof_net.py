from datasets import load_dataset
from tqdm import tqdm
import json

data = load_dataset("hoskinson-center/proofnet")

print(data)

print("example valid instance")
print(data["validation"][0])

print("example test instance:")
print(data["test"][0])

print("looping through validation data...")
for x in tqdm(data["validation"]):
    pass

print("looping through test data...")
for x in tqdm(data["test"]): 
    pass

print("finished!")

print(data["validation"][0])

print("saving validation data...")
with open("validation_data.jsonl", "w") as f:
    for x in data["validation"]:
        f.write(json.dumps(x) + "\n")

print("saving test data...")
with open("test_data.jsonl", "w") as f:
    for x in data["test"]:
        f.write(json.dumps(x) + "\n")