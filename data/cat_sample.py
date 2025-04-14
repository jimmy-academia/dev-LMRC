import pickle
import logging
import random
import os
from pathlib import Path
import gc

from debug import *

def sample_items_by_category(data_pkl='cache/queries_item_pool.pkl', 
                             output_dir='cache/category_samples',
                             category='Food',
                             sample_count=100):
    data_pkl = Path(data_pkl)
    output_dir = Path(output_dir)
    output_pkl = output_dir / f"{category}.pkl"

    if output_pkl.exists():
        logging.info(f"Category items sample exists at {output_pkl}. Skipping.")
        with open(output_pkl, 'rb') as f:
            return pickle.load(f)
            
    logging.info(f"Creating category items sample at {output_pkl} for '{category}'")
    
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = output_dir / "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    with open(data_pkl, 'rb') as f:
        manifest = pickle.load(f)
    
    category_items = []
    chunk_files = manifest.get("item_pool_chunks", [manifest.get("item_pool_file")])
    
    for chunk_file in chunk_files:
        logging.info(f"Processing chunk {chunk_file}")
        with open(chunk_file, 'rb') as f:
            chunk = pickle.load(f)
        
        matching_items = [item for item in chunk if item.get('category') == category]
        logging.info(f"Found {len(matching_items)} '{category}' items in chunk of {len(chunk)}")
        
        category_items.extend(matching_items)
        gc.collect()
    
    if sample_count < len(category_items):
        category_items = random.sample(category_items, sample_count)
        logging.info(f"Sampled down to {len(category_items)} items")
    
    item_pool_file = temp_dir / f"{category}_items.pkl"
    with open(item_pool_file, 'wb') as f:
        pickle.dump(category_items, f)
    
    new_manifest = {
        "item_pool_file": str(item_pool_file),
        "total_items": len(category_items),
        "original_manifest": str(data_pkl),
        "category": category,
        "sample_count": sample_count
    }
    
    with open(output_pkl, 'wb') as f:
        pickle.dump(new_manifest, f)
    
    logging.info(f"Category items sampling complete. Manifest saved to {output_pkl}")
    return new_manifest

def load_category_items(category, base_dir='cache/category_samples'):
    sample_pkl = Path(base_dir) / f"{category}.pkl"
    
    if not sample_pkl.exists():
        logging.error(f"Sample file not found: {sample_pkl}")
        return None
    
    with open(sample_pkl, 'rb') as f:
        manifest = pickle.load(f)
    
    with open(manifest["item_pool_file"], 'rb') as f:
        item_pool = pickle.load(f)
    
    logging.info(f"Loaded {len(item_pool)} items for category '{manifest['category']}'")
    return item_pool

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    from utils import set_verbose
    set_verbose(1)
    
    category = 'Food'
    
    manifest = sample_items_by_category(
        data_pkl='cache/queries_item_pool.pkl',
        output_dir='cache/category_samples',
        category=category,
    )
    
    item_pool = load_category_items(category)
    
    print(f"Category '{category}' sample: {len(item_pool)} items")
    
    if item_pool:
        print("\nSample item structure:")
        sample_item = item_pool[0]
        for key, value in sample_item.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}... (truncated)")
            else:
                print(f"  {key}: {value}")