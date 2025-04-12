import pickle
import logging
import random
import os
from pathlib import Path
import gc

from debug import *

def create_small_sample(data_pkl='cache/queries_item_pool.pkl', output_pkl='cache/sample_query_item.pkl', sample_ratio=0.001):
    """
    Read data from the previously created pickle files and create a smaller subset.
    
    Args:
        data_pkl (Path): Path to the original manifest pickle file
        output_pkl (Path): Path to save the new sampled data manifest
        sample_ratio (float): The fraction of data to sample (default: 0.01 = 1%)
    """
    data_pkl = Path(data_pkl)
    output_pkl = Path(output_pkl)


    if output_pkl.exists():
        logging.info(f"Sample data already exists at {output_pkl}. Skipping processing.")
        # Load and return the existing manifest
        with open(output_pkl, 'rb') as f:
            return pickle.load(f)
            
    logging.info(f"Creating new sample data at {output_pkl}")
    
    # Ensure output directory exists
    os.makedirs(output_pkl.parent, exist_ok=True)
    
    # Create a new temp directory for the sampled data
    temp_dir = output_pkl.parent / "temp_sample"
    os.makedirs(temp_dir, exist_ok=True)
    
    logging.info(f"Loading manifest from {data_pkl}")
    # Load the manifest
    with open(data_pkl, 'rb') as f:
        manifest = pickle.load(f)
    
    # Load and sample requests
    logging.info(f"Loading requests from {manifest['requests_file']}")
    with open(manifest["requests_file"], 'rb') as f:
        requests = pickle.load(f)
    
    # Sample requests (1%)
    sampled_requests = random.sample(list(requests), int(len(requests) * sample_ratio))

    id_set = set([r['item_id'] for r in sampled_requests])

    logging.info(f"Sampled {len(sampled_requests)} requests from {len(requests)} total")
    
    # Save sampled requests
    requests_file = temp_dir / "sampled_requests.pkl"
    with open(requests_file, 'wb') as f:
        pickle.dump(sampled_requests, f)
    logging.info(f"Saved sampled requests to {requests_file}")
    
    # Sample item pool
    total_items = manifest["total_items"]
    target_sample_size = int(total_items * sample_ratio)
    logging.info(f"Will sample {target_sample_size} items from {total_items} total")
    
    # Determine how many items to sample from each chunk
    chunk_files = manifest["item_pool_chunks"]
    
    # Process using reservoir sampling across chunks
    sampled_item_pool = []
    items_processed = 0
    
    for chunk_file in chunk_files:
        logging.info(f"Processing chunk {chunk_file}")
        with open(chunk_file, 'rb') as f:
            chunk = pickle.load(f)
        
        # Simple random sampling from this chunk
        # Adjust ratio to ensure we get the right total sample size
        remaining_chunks = len(chunk_files) - chunk_files.index(chunk_file)
        remaining_to_sample = target_sample_size - len(sampled_item_pool)
        remaining_to_process = total_items - items_processed
        
        # Calculate how many to take from this chunk
        # This ensures proportional sampling across chunks
        chunk_sample_size = min(int(remaining_to_sample * (len(chunk) / remaining_to_process)), len(chunk))
        
        remaining_chunk = []
        for item in chunk:
            if item['item_id'] in id_set:
                sampled_item_pool.append(item)
                chunk_sample_size -= 1
                id_set.remove(item['item_id'])
            else:
                remaining_chunk.append(item)

        if chunk_sample_size > 0:
            sample_size = min(chunk_sample_size, len(remaining_chunk))
            chunk_sample = random.sample(remaining_chunk, sample_size)
            sampled_item_pool.extend(chunk_sample)
            logging.info(f"Sampled {len(chunk_sample)} items from chunk of size {len(chunk)}")
        
        items_processed += len(chunk)
        gc.collect()  # Force garbage collection
    
    logging.info(f"Total sampled items: {len(sampled_item_pool)}")
    
    # Save sampled item pool
    item_pool_file = temp_dir / "sampled_item_pool.pkl"
    with open(item_pool_file, 'wb') as f:
        pickle.dump(sampled_item_pool, f)
    logging.info(f"Saved sampled item pool to {item_pool_file}")
    
    # Create new manifest
    new_manifest = {
        "requests_file": str(requests_file),
        "item_pool_file": str(item_pool_file),
        "total_items": len(sampled_item_pool),
        "original_manifest": str(data_pkl),
        "sample_ratio": sample_ratio
    }
    
    # Save new manifest
    with open(output_pkl, 'wb') as f:
        pickle.dump(new_manifest, f)
    
    logging.info(f"Sampling complete. New manifest saved to {output_pkl}")
    return new_manifest

def load_sampled_data(sample_pkl):
    """
    Load the sampled data for use.
    
    Args:
        sample_pkl (Path): Path to the sampled data manifest
        
    Returns:
        tuple: (item_pool, requests)
    """
    with open(sample_pkl, 'rb') as f:
        manifest = pickle.load(f)
    
    # Load requests
    with open(manifest["requests_file"], 'rb') as f:
        requests = pickle.load(f)
    
    # Load item pool
    with open(manifest["item_pool_file"], 'rb') as f:
        item_pool = pickle.load(f)
    
    logging.info(f"Loaded {len(item_pool)} sampled items and {len(requests)} sampled requests")
    return item_pool, requests

# Example usage:
if __name__ == "__main__":
    # Create the sample
    from utils import set_verbose
    set_verbose(1)
    create_small_sample()
    
    # Load the sample
    output_pkl='cache/sample_query_item.pkl'
    item_pool, requests = load_sampled_data(output_pkl)
    
    # Now you can use the sampled data for faster development and testing
    print(f"Sample size: {len(item_pool)} items, {len(requests)} requests")