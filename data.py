import json
import pickle
import logging

from datasets import load_dataset
from huggingface_hub import hf_hub_download

def prepare_item_requests(data_pkl):
    """Load or prepare the item pool and requests."""
    if not data_pkl.exists():
        logging.info('no pickle, loading item_pool and requests from hugging face...')
        requests = load_dataset('McAuley-Lab/Amazon-C4')['test']
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/Amazon-C4',
            filename='sampled_item_metadata_1M.jsonl',
            repo_type='dataset'
        )
        item_pool = []
        with open(filepath, 'r') as f:
            for line in f:
                item_pool.append(json.loads(line.strip()))
        with open(data_pkl, 'wb') as f:
            pickle.dump((item_pool, requests), f)
        logging.info(f'      .... load and save to {data_pkl} pickle complete.')
    else:
        with open(data_pkl, 'rb') as f:
            item_pool, requests = pickle.load(f)

        logging.info(f'loaded item_pool and requests from {data_pkl}')

    return item_pool, requests

def prepare_file_system(fs_pkl, item_pool):
    """Load or prepare the file system."""
    from file_system import FileSystem  # Import here to avoid circular imports
    
    if fs_pkl.exists():
        logging.info(f"Loading file system from {fs_pkl}")
        with open(fs_pkl, 'rb') as f:
            fs = pickle.load(f)
        return fs
    
    logging.info("Creating new file system")
    fs = FileSystem(item_pool)
    fs.save(fs_pkl)
    return fs