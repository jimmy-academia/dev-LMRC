from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json
import logging
from collections import defaultdict

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

    return {
        'queries': queries,
        'item_pool': item_pool
    }
