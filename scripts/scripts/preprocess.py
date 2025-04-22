import os
import json
from datasets import load_dataset
from tqdm import tqdm

def normalize_code(code: str) -> str:
    code = code.strip()
    code = code.replace('System.Out1', 'System.Console')  # Fix common typo
    code = code.replace(';', ';\n')  # Optional: line break after statements
    return code

def save_to_json(dataset_split, filename):
    os.makedirs("data", exist_ok=True)
    data = []
    for example in tqdm(dataset_split):
        java_code = normalize_code(example["java"])
        cs_code = normalize_code(example["cs"])
        data.append({"java": java_code, "cs": cs_code})
    with open(os.path.join("data", filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Load dataset
dataset = load_dataset("code_x_glue_cc_code_to_code_trans", split={
    "train": "train",
    "validation": "validation",
    "test": "test"
})

# Save cleaned JSONs
save_to_json(dataset["train"], "train.json")
save_to_json(dataset["validation"], "validation.json")
save_to_json(dataset["test"], "test.json")
