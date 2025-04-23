import json
import os
import gc
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
        
        # Process the JSONL file in chunks to save memory
        item_pool = []
        chunk_size = 10000
        counter = 0
        
        logging.info(f"Processing JSONL file: {filepath}")
        with open(filepath, 'r') as f:
            chunk = []
            for line in f:
                counter += 1
                chunk.append(json.loads(line.strip()))
                
                if len(chunk) >= chunk_size:
                    item_pool.extend(chunk)
                    chunk = []
                    gc.collect()  # Force garbage collection
                    logging.info(f"Processed {counter} items")
            
            # Add any remaining items
            if chunk:
                item_pool.extend(chunk)
        
        logging.info(f"Total items processed: {len(item_pool)}")
        
        # Save in chunks to avoid memory issues
        temp_dir = data_pkl.parent / "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save requests
        requests_file = temp_dir / "requests.pkl"
        with open(requests_file, 'wb') as f:
            pickle.dump(requests, f)
        
        # Save item_pool in chunks
        chunk_size = 100000
        for i in range(0, len(item_pool), chunk_size):
            end = min(i + chunk_size, len(item_pool))
            chunk_file = temp_dir / f"item_pool_{i}_{end}.pkl"
            with open(chunk_file, 'wb') as f:
                pickle.dump(item_pool[i:end], f)
            logging.info(f"Saved chunk {i} to {end}")
        
        # Save a manifest file
        manifest = {
            "requests_file": str(requests_file),
            "item_pool_chunks": [
                str(temp_dir / f"item_pool_{i}_{min(i + chunk_size, len(item_pool))}.pkl")
                for i in range(0, len(item_pool), chunk_size)
            ],
            "total_items": len(item_pool)
        }
        
        with open(data_pkl, 'wb') as f:
            pickle.dump(manifest, f)
            
        logging.info(f"Data saved to {data_pkl} and chunk files")
    else:
        # Load the manifest
        with open(data_pkl, 'rb') as f:
            manifest = pickle.load(f)
        
        # Load requests
        with open(manifest["requests_file"], 'rb') as f:
            requests = pickle.load(f)
        
        # Load item_pool chunks
        item_pool = []
        for chunk_file in manifest["item_pool_chunks"]:
            with open(chunk_file, 'rb') as f:
                chunk = pickle.load(f)
                item_pool.extend(chunk)
                gc.collect()  # Force garbage collection
        
        logging.info(f"Loaded {len(item_pool)} items and {len(requests)} requests from {data_pkl}")

    return item_pool, requests

def prepare_file_system(fs_pkl, item_pool):
    """Load or prepare the file system."""
    # Import here to avoid circular imports
    from file_system import FileSystem
    
    if fs_pkl.exists():
        try:
            fs = FileSystem.load(fs_pkl, item_pool)
            return fs
        except Exception as e:
            logging.error(f"Error loading file system: {e}")
            logging.info("Creating new file system")
    else:
        logging.info("Creating new file system")
    
    # Create a new file system with chunked processing
    fs = FileSystem(item_pool)
    fs.save(fs_pkl)
    return fs