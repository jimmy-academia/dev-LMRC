from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json
import logging
from collections import defaultdict
from structure import ItemData
from tqdm import tqdm

from debug import check

def get_task_loader(args):
    logging.info("Loading Amazon-C4 test split...")
    queries = load_dataset('McAuley-Lab/Amazon-C4')['test']

    if args.dev:
        queries = queries.select(range(10))  # Subsample for dev

    logging.info(f"Loaded {len(queries)} test queries.")

    logging.info("Loading ~1M item pool from BLaIR / Amazon-C4 repository...")
    filepath = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-C4',
        filename='sampled_item_metadata_1M.jsonl',
        repo_type='dataset'
    )
    item_pool = []
    with open(filepath, 'r') as f:
        for line in f:
            item_pool.append(json.loads(line.strip()))

    item_dict = { item['item_id']: ItemData(item['item_id'], item) for item in item_pool }

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

    for category in all_categories:
        logging.info(f'== working on {category} ==')
        metadata_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{category}",split="full", trust_remote_code=True)

        for meta in tqdm(metadata_dataset, ncols=88, desc='meta', leave=False):
            parent_asin = meta.get('parent_asin')
            if parent_asin in item_dict:
                item_dict[parent_asin].add_metadata(meta)

        reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}", split="full", trust_remote_code=True)

        for review in tqdm(reviews_dataset, ncols=88, desc='review', leave=False):
            asin = review.get('asin')
            if asin in item_dict:
                item_dict[asin].add_review(review)

    check()

    return {
        'queries': queries,
        'item_pool': item_pool
    }
