from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json
import logging

from debug import check

def get_task_loader(args):
    logging.info("Loading Amazon-C4 test split...")
    queries = load_dataset('McAuley-Lab/Amazon-C4')['test']

    if args.dev:
        queries = queries.select(range(200))  # Subsample for dev

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

    # Remove duplicates by item_id
    # unique_items_dict = {}
    # for item in item_pool:
    #     unique_items_dict[item['item_id']] = item
    # unique_item_pool = list(unique_items_dict.values())
    # logging.info(f"Unique items after dedup: {len(item_pool)}.")

    id_pool = [x['item_id'] for x in item_pool]

    assert all([q['item_id'] in id_pool for q in queries])

    return {
        'queries': queries,
        'item_pool': item_pool
    }
