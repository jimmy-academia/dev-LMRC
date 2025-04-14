import pickle
import logging
import random
import os
from pathlib import Path
import gc
from debug import *

def sample_items_and_requests_by_category(data_pkl='cache/queries_item_pool.pkl', 
                                         output_dir='cache/category_samples',
                                         category='Food',
                                         sample_count=100):
    """
    Sample items from a specific category and their corresponding requests.
    
    Args:
        data_pkl (Path): Path to the original manifest pickle file
        output_dir (Path): Directory to save the output
        category (str): Category to sample from
        sample_count (int): Maximum number of items to sample
        
    Returns:
        dict: Manifest of the sampled data
    """
    data_pkl = Path(data_pkl)
    output_dir = Path(output_dir)
    output_pkl = output_dir / f"{category}_with_requests.pkl"
    
    if output_pkl.exists():
        logging.info(f"Category sample exists at {output_pkl}. Skipping.")
        with open(output_pkl, 'rb') as f:
            return pickle.load(f)
            
    logging.info(f"Creating category sample at {output_pkl} for '{category}'")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = output_dir / "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Load manifest
    with open(data_pkl, 'rb') as f:
        manifest = pickle.load(f)
    
    # Load and process item chunks
    category_items = []
    chunk_files = manifest.get("item_pool_chunks", [manifest.get("item_pool_file")])
    
    for chunk_file in chunk_files:
        logging.info(f"Processing item chunk {chunk_file}")
        with open(chunk_file, 'rb') as f:
            chunk = pickle.load(f)
        
        matching_items = [item for item in chunk if item.get('category') == category]
        logging.info(f"Found {len(matching_items)} '{category}' items in chunk of {len(chunk)}")
        
        category_items.extend(matching_items)
        gc.collect()
    
    # Sample items if needed
    if sample_count < len(category_items):
        category_items = random.sample(category_items, sample_count)
        logging.info(f"Sampled down to {len(category_items)} items")
    
    # Create a set of item_ids for efficient lookup
    category_item_ids = {item['item_id'] for item in category_items}
    
    # Load requests
    requests_file = manifest.get("requests_file")
    logging.info(f"Loading requests from {requests_file}")
    
    with open(requests_file, 'rb') as f:
        all_requests = pickle.load(f)
    
    # Filter requests to keep only those related to our category items
    category_requests = [req for req in all_requests if req['item_id'] in category_item_ids]
    logging.info(f"Found {len(category_requests)} requests related to {category} items")
    
    # Save the sampled items
    item_pool_file = temp_dir / f"{category}_items.pkl"
    with open(item_pool_file, 'wb') as f:
        pickle.dump(category_items, f)
    
    # Save the sampled requests
    requests_file = temp_dir / f"{category}_requests.pkl"
    with open(requests_file, 'wb') as f:
        pickle.dump(category_requests, f)
    
    # Create manifest
    new_manifest = {
        "item_pool_file": str(item_pool_file),
        "requests_file": str(requests_file),
        "total_items": len(category_items),
        "total_requests": len(category_requests),
        "original_manifest": str(data_pkl),
        "category": category,
        "sample_count": sample_count
    }
    
    # Save manifest
    with open(output_pkl, 'wb') as f:
        pickle.dump(new_manifest, f)
    
    logging.info(f"Category sampling complete. Manifest saved to {output_pkl}")
    return new_manifest

def load_category_sample(category, base_dir='cache/category_samples', 
                         data_pkl='cache/queries_item_pool.pkl', 
                         sample_count=100,
                         auto_create=True):
    """
    Load the sampled items and requests for a specific category.
    Automatically creates the sample if it doesn't exist and auto_create is True.
    
    Args:
        category (str): Category to load
        base_dir (Path): Base directory for samples
        data_pkl (Path): Path to the original manifest pickle file (used if sample needs creation)
        sample_count (int): Maximum number of items to sample (used if sample needs creation)
        auto_create (bool): Whether to automatically create the sample if it doesn't exist
        
    Returns:
        tuple: (item_pool, requests)
    """
    sample_pkl = Path(base_dir) / f"{category}_with_requests.pkl"
    
    # If sample doesn't exist and auto_create is True, create it
    if not sample_pkl.exists():
        if auto_create:
            logging.info(f"Sample file not found: {sample_pkl}. Creating it...")
            manifest = sample_items_and_requests_by_category(
                data_pkl=data_pkl,
                output_dir=base_dir,
                category=category,
                sample_count=sample_count
            )
        else:
            logging.error(f"Sample file not found: {sample_pkl}")
            return None, None
    else:
        # Load the existing manifest
        with open(sample_pkl, 'rb') as f:
            manifest = pickle.load(f)
    
    # Load item pool
    with open(manifest["item_pool_file"], 'rb') as f:
        item_pool = pickle.load(f)
    
    # Load requests
    with open(manifest["requests_file"], 'rb') as f:
        requests = pickle.load(f)
    
    logging.info(f"Loaded {len(item_pool)} items and {len(requests)} requests for category '{manifest['category']}'")
    return item_pool, requests

def sample_items_by_category(data_pkl='cache/queries_item_pool.pkl', 
                             output_dir='cache/category_samples',
                             category='Food',
                             sample_count=100):
    """
    Sample items from a specific category.
    
    Args:
        data_pkl (Path): Path to the original manifest pickle file
        output_dir (Path): Directory to save the output
        category (str): Category to sample from
        sample_count (int): Maximum number of items to sample
        
    Returns:
        dict: Manifest of the sampled data
    """
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

def load_category_items(category, base_dir='cache/category_samples',
                        data_pkl='cache/queries_item_pool.pkl',
                        sample_count=100,
                        auto_create=True):
    """
    Load the sampled items for a specific category.
    Automatically creates the sample if it doesn't exist and auto_create is True.
    
    Args:
        category (str): Category to load
        base_dir (Path): Base directory for samples
        data_pkl (Path): Path to the original manifest pickle file (used if sample needs creation)
        sample_count (int): Maximum number of items to sample (used if sample needs creation)
        auto_create (bool): Whether to automatically create the sample if it doesn't exist
        
    Returns:
        list: List of items from the specified category
    """
    sample_pkl = Path(base_dir) / f"{category}.pkl"
    
    # If sample doesn't exist and auto_create is True, create it
    if not sample_pkl.exists():
        if auto_create:
            logging.info(f"Sample file not found: {sample_pkl}. Creating it...")
            manifest = sample_items_by_category(
                data_pkl=data_pkl,
                output_dir=base_dir,
                category=category,
                sample_count=sample_count
            )
        else:
            logging.error(f"Sample file not found: {sample_pkl}")
            return None
    else:
        # Load the existing manifest
        with open(sample_pkl, 'rb') as f:
            manifest = pickle.load(f)
    
    with open(manifest["item_pool_file"], 'rb') as f:
        item_pool = pickle.load(f)
    
    logging.info(f"Loaded {len(item_pool)} items for category '{manifest['category']}'")
    return item_pool

def create_small_sample(data_pkl='cache/queries_item_pool.pkl', 
                        output_pkl='cache/sample_query_item.pkl', 
                        sample_ratio=0.001):
    """
    Read data from the previously created pickle files and create a smaller subset.
    
    Args:
        data_pkl (Path): Path to the original manifest pickle file
        output_pkl (Path): Path to save the new sampled data manifest
        sample_ratio (float): The fraction of data to sample
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
    
    # Sample requests
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

def load_sampled_data(sample_pkl='cache/sample_query_item.pkl', 
                      data_pkl='cache/queries_item_pool.pkl',
                      sample_ratio=0.001,
                      auto_create=True):
    """
    Load the sampled data for use.
    Automatically creates the sample if it doesn't exist and auto_create is True.
    
    Args:
        sample_pkl (Path): Path to the sampled data manifest
        data_pkl (Path): Path to the original manifest pickle file (used if sample needs creation)
        sample_ratio (float): The fraction of data to sample (used if sample needs creation)
        auto_create (bool): Whether to automatically create the sample if it doesn't exist
        
    Returns:
        tuple: (item_pool, requests)
    """
    sample_pkl = Path(sample_pkl)
    
    # If sample doesn't exist and auto_create is True, create it
    if not sample_pkl.exists():
        if auto_create:
            logging.info(f"Sample file not found: {sample_pkl}. Creating it...")
            manifest = create_small_sample(
                data_pkl=data_pkl,
                output_pkl=sample_pkl,
                sample_ratio=sample_ratio
            )
        else:
            logging.error(f"Sample file not found: {sample_pkl}")
            return None, None
    else:
        # Load the existing manifest
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    from utils import set_verbose
    set_verbose(1)
    
    # Example of using load_category_sample which will auto-create the sample if needed
    category = 'Food'
    item_pool, requests = load_category_sample(category)
    
    print(f"Category '{category}' sample: {len(item_pool)} items, {len(requests)} requests")
    
    if item_pool and requests:
        print("\nSample item structure:")
        sample_item = item_pool[0]
        for key, value in sample_item.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}... (truncated)")
            else:
                print(f"  {key}: {value}")
        
        print("\nSample request structure:")
        sample_request = requests[0]
        for key, value in sample_request.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}... (truncated)")
            else:
                print(f"  {key}: {value}")