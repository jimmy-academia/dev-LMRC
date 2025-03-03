# https://github.com/hyp1231/AmazonReviews2023/blob/main/amazon-c4/README.md
import json
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from debug import check


queries = load_dataset('McAuley-Lab/Amazon-C4')['test']

filepath = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-C4',
        filename='sampled_item_metadata_1M.jsonl',
        repo_type='dataset'
    )
item_pool = []
with open(filepath, 'r') as f:
    for line in f:
        item_pool.append(json.loads(line.strip()))


reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                               "raw_review_All_Beauty", 
                               split="full", 
                               trust_remote_code=True, 
                               streaming=True)

metadata_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                                "raw_meta_All_Beauty", 
                                split="full", 
                                trust_remote_code=True)


categories_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023", 
    filename="all_categories.txt",
    repo_type='dataset'
)

# Download asin2category.json
asin2category_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023", 
    filename="asin2category.json",
    repo_type='dataset'
)

with open(categories_path, "r") as f:
    all_categories = [line.strip() for line in f if line.strip()]
print(all_categories)

check()

with open(asin2category_path, "r") as f:
    asin_to_category = json.load(f)
print(list(asin_to_category.items())[:5])

# check()