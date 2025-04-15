import pickle
import logging
import random
import os
from pathlib import Path
import gc

def sample_items_and_requests_by_category(data_pkl='cache/queries_item_pool.pkl', 
                                         output_dir='cache/category_samples',
                                         category='Food',
                                         sample_count=100):
    """Sample items from a category and corresponding requests."""
    data_pkl = Path(data_pkl)
    output_dir = Path(output_dir)
    output_pkl = output_dir / f"{category}_with_requests.pkl"
    
    if output_pkl.exists():
        logging.info(f"Category sample with requests exists at {output_pkl}")
        with open(output_pkl, 'rb') as f:
            return pickle.load(f)
            
    logging.info(f"Creating category sample with requests for '{category}'")
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = output_dir / "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load manifest
    with open(data_pkl, 'rb') as f:
        manifest = pickle.load(f)
    
    # Collect items from the specified category
    category_items = []
    chunk_files = manifest.get("item_pool_chunks", [manifest.get("item_pool_file")])
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            chunk = pickle.load(f)
        
        matching_items = [item for item in chunk if item.get('category') == category]
        category_items.extend(matching_items)
        gc.collect()
    
    # Sample if needed
    if sample_count < len(category_items):
        category_items = random.sample(category_items, sample_count)
    
    # Create a set of item IDs for efficient lookup
    category_item_ids = {item['item_id'] for item in category_items}
    
    # Load and filter requests
    with open(manifest["requests_file"], 'rb') as f:
        all_requests = pickle.load(f)
    
    category_requests = [req for req in all_requests if req['item_id'] in category_item_ids]
    
    # Save the sampled data
    item_pool_file = temp_dir / f"{category}_items_with_req.pkl"
    with open(item_pool_file, 'wb') as f:
        pickle.dump(category_items, f)
    
    requests_file = temp_dir / f"{category}_requests.pkl"
    with open(requests_file, 'wb') as f:
        pickle.dump(category_requests, f)
    
    # Create and save manifest
    new_manifest = {
        "item_pool_file": str(item_pool_file),
        "requests_file": str(requests_file),
        "total_items": len(category_items),
        "total_requests": len(category_requests),
        "original_manifest": str(data_pkl),
        "category": category,
        "sample_count": sample_count
    }
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(new_manifest, f)
    
    logging.info(f"Created sample with {len(category_items)} items and {len(category_requests)} requests")
    return new_manifest

def load_category_sample(category='Food', 
                        base_dir='cache/category_samples', 
                        data_pkl='cache/queries_item_pool.pkl', 
                        sample_count=100,
                        auto_create=True):
    """Load items and requests for category, creating sample if it doesn't exist."""
    sample_pkl = Path(base_dir) / f"{category}_with_requests.pkl"
    
    # Auto-create if needed
    if not sample_pkl.exists():
        if auto_create:
            logging.info(f"Category sample not found. Creating it...")
            manifest = sample_items_and_requests_by_category(
                data_pkl=data_pkl,
                output_dir=base_dir,
                category=category,
                sample_count=sample_count
            )
        else:
            logging.error(f"Category sample not found at {sample_pkl}")
            return None, None
    else:
        with open(sample_pkl, 'rb') as f:
            manifest = pickle.load(f)
    
    # Load items and requests
    with open(manifest["item_pool_file"], 'rb') as f:
        item_pool = pickle.load(f)
    
    with open(manifest["requests_file"], 'rb') as f:
        requests = pickle.load(f)
    
    logging.info(f"Loaded {len(item_pool)} items and {len(requests)} requests for '{category}'")
    return item_pool, requests