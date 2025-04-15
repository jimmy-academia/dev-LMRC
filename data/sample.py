import pickle
import logging
import random
import os
from pathlib import Path
import gc

def create_small_sample(data_pkl='cache/queries_item_pool.pkl', 
                        output_pkl='cache/sample_query_item.pkl', 
                        sample_ratio=0.001):
    """Create a smaller sample of items and requests."""
    data_pkl = Path(data_pkl)
    output_pkl = Path(output_pkl)

    if output_pkl.exists():
        logging.info(f"Sample data already exists at {output_pkl}")
        with open(output_pkl, 'rb') as f:
            return pickle.load(f)
            
    logging.info(f"Creating new sample data at {output_pkl}")
    
    # Ensure directories exist
    os.makedirs(output_pkl.parent, exist_ok=True)
    temp_dir = output_pkl.parent / "temp_sample"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load the manifest
    with open(data_pkl, 'rb') as f:
        manifest = pickle.load(f)
    
    # Sample requests
    with open(manifest["requests_file"], 'rb') as f:
        requests = pickle.load(f)
    
    sampled_requests = random.sample(list(requests), int(len(requests) * sample_ratio))
    id_set = set([r['item_id'] for r in sampled_requests])
    
    # Save sampled requests
    requests_file = temp_dir / "sampled_requests.pkl"
    with open(requests_file, 'wb') as f:
        pickle.dump(sampled_requests, f)
    
    # Sample items
    total_items = manifest["total_items"]
    target_sample_size = int(total_items * sample_ratio)
    chunk_files = manifest["item_pool_chunks"]
    
    sampled_item_pool = []
    items_processed = 0
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            chunk = pickle.load(f)
        
        # First include items from requests
        remaining_chunk = []
        for item in chunk:
            if item['item_id'] in id_set:
                sampled_item_pool.append(item)
                id_set.remove(item['item_id'])
            else:
                remaining_chunk.append(item)
        
        # Then sample additional items
        remaining_to_sample = target_sample_size - len(sampled_item_pool)
        remaining_to_process = total_items - items_processed
        chunk_sample_size = int(remaining_to_sample * (len(chunk) / remaining_to_process))
        
        if chunk_sample_size > 0 and remaining_chunk:
            sample_size = min(chunk_sample_size, len(remaining_chunk))
            sampled_item_pool.extend(random.sample(remaining_chunk, sample_size))
        
        items_processed += len(chunk)
        gc.collect()
    
    # Save sampled item pool
    item_pool_file = temp_dir / "sampled_item_pool.pkl"
    with open(item_pool_file, 'wb') as f:
        pickle.dump(sampled_item_pool, f)
    
    # Create and save new manifest
    new_manifest = {
        "requests_file": str(requests_file),
        "item_pool_file": str(item_pool_file),
        "total_items": len(sampled_item_pool),
        "total_requests": len(sampled_requests),
        "original_manifest": str(data_pkl),
        "sample_ratio": sample_ratio
    }
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(new_manifest, f)
    
    logging.info(f"Sampling complete with {len(sampled_item_pool)} items and {len(sampled_requests)} requests")
    return new_manifest

def load_sampled_data(sample_pkl='cache/sample_query_item.pkl', 
                     data_pkl='cache/queries_item_pool.pkl',
                     sample_ratio=0.001,
                     auto_create=True):
    """Load sampled data, creating it first if it doesn't exist."""
    sample_pkl = Path(sample_pkl)
    
    # Auto-create if needed
    if not sample_pkl.exists():
        if auto_create:
            logging.info(f"Sample data not found. Creating it...")
            create_small_sample(data_pkl, sample_pkl, sample_ratio)
        else:
            logging.error(f"Sample data not found at {sample_pkl}")
            return None, None
    
    # Load the manifest
    with open(sample_pkl, 'rb') as f:
        manifest = pickle.load(f)
    
    # Load requests and items
    with open(manifest["requests_file"], 'rb') as f:
        requests = pickle.load(f)
    
    with open(manifest["item_pool_file"], 'rb') as f:
        item_pool = pickle.load(f)
    
    logging.info(f"Loaded {len(item_pool)} items and {len(requests)} requests")
    return item_pool, requests