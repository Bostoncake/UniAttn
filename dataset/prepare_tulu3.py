import json
import random
from datasets import load_dataset
import os


os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"
content = load_dataset("allenai/tulu-3-sft-mixture")["train"]

dict_flan = []
for i in range(len(content)):
    if len(content[i]['messages']) == 2:
        now_data_dict = {
            'id': content[i]['id'],
            'input': content[i]['messages'][0]['content'],
            'output': content[i]['messages'][1]['content'],
            'label': "", 
            'langdetect': "",
            'source': content[i]['source'],
        }
        if now_data_dict.get("input", "") != "":
            dict_flan.append(now_data_dict)
    else:
        print("Warning! Multi-round conversation!")

random.shuffle(dict_flan)
output_file = f"tulu3_sft_filtered.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in dict_flan:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
