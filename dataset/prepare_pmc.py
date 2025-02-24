import json
from datasets import load_dataset

dataset = load_dataset("axiong/pmc_llama_instructions")

def extract_fields(example):
    return {
        "instruction": example.get("instruction", ""),
        "input": example.get("input", ""),
        "output": example.get("output", "")
    }

filtered_dataset = dataset["train"].map(extract_fields)

# Save to a JSONL file
output_file = f"pmc_llama_instructions.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in filtered_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
