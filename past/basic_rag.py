from datasets import load_dataset
dataset = load_dataset('McAuley-Lab/Amazon-C4')['test']

import json
from huggingface_hub import hf_hub_download

# Download the 1M item metadata pool file from the Amazon-C4 dataset repository
filepath = hf_hub_download(repo_id='McAuley-Lab/Amazon-C4',
                           filename='sampled_item_metadata_1M.jsonl', 
                           repo_type='dataset')

print(filepath)

# Load all items into a list
item_pool = []
with open(filepath, 'r') as file:
    for line in file:
        item_pool.append(json.loads(line.strip()))

print(len(item_pool))
print(item_pool[0])

unique_items = {}
for item in item_pool:
    unique_items[item['item_id']] = item
item_pool = list(unique_items.values())
print(f"Unique items in pool: {len(item_pool)}")