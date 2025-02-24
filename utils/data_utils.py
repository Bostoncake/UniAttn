import json


def jload(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        # import pdb;pdb.set_trace()
        # lines = file.read
        # Read each line in the JSONL file
        data = [json.loads(line) for line in file]
    return data
