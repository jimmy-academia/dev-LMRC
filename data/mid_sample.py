import pickle
import logging
import random
import os
from pathlib import Path
from collections import Counter, defaultdict

from utils import create_llm_client, system_struct, user_struct, set_verbose

def create_10k_sample(
    data_pkl='cache/queries_item_pool.pkl',
    output_pkl='cache/mid_sample_10k.pkl',
    item_count=10000,
    request_count=2000,
    quality_check=False,
    pause_frequency=0
):
    data_pkl, output_pkl = Path(data_pkl), Path(output_pkl)
    os.makedirs(output_pkl.parent, exist_ok=True)

    if output_pkl.exists():
        logging.info(f"Sample already exists at {output_pkl}")
        with open(output_pkl, 'rb') as f:
            return pickle.load(f)

    with open(data_pkl, 'rb') as f:
        manifest = pickle.load(f)

    item_pool = sample_items_by_distribution(manifest, item_count)
    item_ids = {item['item_id'] for item in item_pool}
    llm_client = create_llm_client() if quality_check else None

    requests = sample_requests_by_distribution(
        manifest, item_pool, request_count, llm_client, pause_frequency, use_llm_decision=quality_check
    )

    # Ensure all ground-truth items from selected requests are in item_pool
    request_item_ids = {req['item_id'] for req in requests}
    missing_item_ids = request_item_ids - item_ids
    if missing_item_ids:
        logging.info(f"Adding {len(missing_item_ids)} missing ground-truth items from requests.")
        additional_items = get_items_by_ids(manifest, missing_item_ids)
        item_pool.extend(additional_items)

    # Trim back to exactly item_count, but do not remove ground-truth items from requests
    request_item_ids = {req['item_id'] for req in requests}  # refresh
    item_pool_map = {item['item_id']: item for item in item_pool}
    trimmed = []
    for item_id, item in item_pool_map.items():
        if item_id in request_item_ids:
            trimmed.append(item)

    for item_id, item in item_pool_map.items():
        if item_id not in request_item_ids and len(trimmed) < item_count:
            trimmed.append(item)

    if len(trimmed) > item_count:
        trimmed = trimmed[:item_count]

    item_pool = trimmed

    # Final trim again if needed (preserve ground-truth request items)
    request_item_ids = {req['item_id'] for req in requests}
    item_pool_map = {item['item_id']: item for item in item_pool}
    final_trimmed = []
    for item_id, item in item_pool_map.items():
        if item_id in request_item_ids:
            final_trimmed.append(item)

    for item_id, item in item_pool_map.items():
        if item_id not in request_item_ids and len(final_trimmed) < item_count:
            final_trimmed.append(item)

    if len(final_trimmed) > item_count:
        final_trimmed = final_trimmed[:item_count]

    item_pool = final_trimmed

    logging.info(f"Final trimmed item_pool to {len(item_pool)} items (including all ground-truth request items).")
    final_counts = Counter([item.get('category', 'Unknown') for item in item_pool])
    print_distribution("Final Item Category Distribution", final_counts, len(item_pool))

    temp_dir = output_pkl.parent / f"{output_pkl.stem}_temp"
    os.makedirs(temp_dir, exist_ok=True)
    items_file = temp_dir / "mid_items.pkl"
    requests_file = temp_dir / "mid_requests.pkl"

    with open(items_file, 'wb') as f:
        pickle.dump(item_pool, f)
    with open(requests_file, 'wb') as f:
        pickle.dump(requests, f)

    manifest_out = {
        "requests_file": str(requests_file),
        "item_pool_file": str(items_file),
        "total_items": len(item_pool),
        "total_requests": len(requests),
        "original_manifest": str(data_pkl),
        "category_distribution": dict(Counter([i.get('category', 'Unknown') for i in item_pool])),
        "quality_checked": bool(llm_client)
    }

    with open(output_pkl, 'wb') as f:
        pickle.dump(manifest_out, f)

    logging.info(f"Saved sample with {len(item_pool)} items and {len(requests)} requests")
    item_categories = {item['item_id']: item.get('category', 'Unknown') for item in item_pool}
    sampled_request_counts = Counter([item_categories[req['item_id']] for req in requests])
    print_distribution("Sampled Request Category Distribution", sampled_request_counts, len(requests))
    
    return manifest_out

def print_distribution(title, dist, total):
    print(f"\n{title}:")
    entries = [f"{cat} {count} ({(count/total)*100:.1f}%)" for cat, count in sorted(dist.items(), key=lambda x: -x[1])]
    for i in range(0, len(entries), 3):
        row = entries[i:i+3]
        print("  ".join(f"{col:<30}" for col in row))

def sample_items_by_distribution(manifest, total_count):
    chunk_files = manifest.get("item_pool_chunks", [manifest.get("item_pool_file")])
    category_counts = Counter()
    items_by_category = defaultdict(list)

    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            for item in pickle.load(f):
                cat = item.get('category', 'Unknown')
                items_by_category[cat].append(item)
                category_counts[cat] += 1

    total_available = sum(category_counts.values())
    target_counts = {}
    remaining = total_count
    sorted_cats = sorted(category_counts.items(), key=lambda x: -x[1])
    for cat, count in sorted_cats:
        proportion = count / total_available
        allocated = max(int(proportion * total_count), 2 if proportion < 0.001 else 1)
        target_counts[cat] = allocated
        remaining -= allocated

    # Distribute remainder to most populous categories
    i = 0
    cat_list = list(target_counts.keys())
    while remaining > 0:
        cat = cat_list[i % len(cat_list)]
        target_counts[cat] += 1
        remaining -= 1
        i += 1

    print_distribution("Total Item Category Distribution", category_counts, total_available)

    selected_items = []
    total_selected = 0
    for cat, count in target_counts.items():
        available = items_by_category[cat]
        sample = random.sample(available, min(len(available), count))
        selected_items.extend(sample)
        total_selected += len(sample)

    sampled_counts = Counter([item.get('category', 'Unknown') for item in selected_items])
    print_distribution("Sampled Item Category Distribution", sampled_counts, len(selected_items))
    if total_selected < total_count:
        shortfall = total_count - total_selected
        # Adjusted target already distributes smoothly to reach total count)} items to reach {total_count} total.")

    print(f"Total Sampled Items: {len(selected_items)}")

    return selected_items

def sample_requests_by_distribution(manifest, item_pool, total_count, llm_client=None, pause_frequency=1, use_llm_decision=True):
    # Sample requests first and backfill their corresponding items into the item_pool
    item_by_id = {}
    item_categories = {}
    chunk_files = manifest.get("item_pool_chunks", [manifest.get("item_pool_file")])
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            for item in pickle.load(f):
                item_by_id[item['item_id']] = item
                item_categories[item['item_id']] = item.get('category', 'Unknown')

    
    with open(manifest["requests_file"], 'rb') as f:
        all_requests = pickle.load(f)

    category_requests = defaultdict(list)
    for req in all_requests:
        cat = item_categories[req['item_id']]
        category_requests[cat].append(req)

    total_requests_available = sum(len(reqs) for reqs in category_requests.values())
    target_counts = {
        cat: max(int((len(reqs) / total_requests_available) * total_count), 1)
        for cat, reqs in category_requests.items()
    }

    request_counts = {cat: len(reqs) for cat, reqs in category_requests.items()}
    print_distribution("Total Request Category Distribution", request_counts, total_requests_available)

    selected_requests = []
    checked = 0
    request_dist = Counter()

    for cat, target in target_counts.items():
        pool = category_requests[cat]
        random.shuffle(pool)
        for req in pool:
            if len([r for r in selected_requests if item_categories[r['item_id']] == cat]) >= target:
                break

            item = item_by_id.get(req['item_id'])
            if not item:
                found = get_items_by_ids(manifest, [req['item_id']])
                if not found:
                    continue
                item = found[0]
                item_by_id[item['item_id']] = item
                item_categories[item['item_id']] = item.get('category', 'Unknown')
                item_pool.append(item)

            decision, llm_output = True, ""
            if llm_client and use_llm_decision:
                decision, llm_output = check_request_quality(req, item, llm_client)
                if not decision:
                    continue

            if pause_frequency and (checked % pause_frequency == 0):
                print("\n\n=== PAUSE FOR CONFIRMATION ===")
                print(f"Checked: {checked}, Selected: {len(selected_requests)}")
                current_dist = Counter([item_categories[r['item_id']] for r in selected_requests])
                print_distribution("Sampled Request Category Distribution", current_dist, len(selected_requests))
                print(f"\nQuery: {req['query']}\nMetadata: {item.get('metadata', '')}")
                print(f"LLM Output: {llm_output.strip()}")
                user_input = input("Accept this request? (y/n): ").strip().lower()
                if user_input == 'n':
                    continue

            selected_requests.append(req)
            checked += 1

            if len(selected_requests) >= total_count:
                break

    return selected_requests[:total_count]

def get_items_by_ids(manifest, item_ids):
    chunk_files = manifest.get("item_pool_chunks", [manifest.get("item_pool_file")])
    found_items = []
    item_ids = set(item_ids)

    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            for item in pickle.load(f):
                if item['item_id'] in item_ids:
                    found_items.append(item)
                    item_ids.remove(item['item_id'])
                    if not item_ids:
                        break
        if not item_ids:
            break
    return found_items

def check_request_quality(request, item, llm_client):
    prompt = f"""
Evaluate if the following product search query is relevant to the product metadata.
The goal is to identify high-quality pairs where the query would naturally lead to this product.

Query: "{request['query']}"
Product Metadata: "{item.get('metadata', '')}"

Criteria:
1. The query should clearly match key aspects of the product
2. The query should not be overly generic
3. The query should not contain irrelevant requirements

Respond with only "ACCEPT" if the query is good quality and relevant to the product, or "REJECT" otherwise.
"""
    response = llm_client([
        system_struct("You evaluate the quality of search query to product matches."),
        user_struct(prompt)
    ])
    decision = "ACCEPT" in response.upper()
    return decision, response



# ##### SUBSAMPLE ##### #

def load_subsample(item_count=1000, source_pkl='cache/mid_sample_10k.pkl'):
    """
    Load a random subsample of specified size, creating it if it doesn't exist.
    Returns only items (no requests).
    """
    subsample_path = Path(f'cache/subsample_{item_count}.pkl')
    
    if subsample_path.exists():
        # Load existing subsample
        with open(subsample_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Load source items
        with open(source_pkl, 'rb') as f:
            source_manifest = pickle.load(f)
        
        with open(source_manifest["item_pool_file"], 'rb') as f:
            source_items = pickle.load(f)
        
        # Create random subsample
        selected_items = random.sample(source_items, min(item_count, len(source_items)))
        
        # Save subsample
        os.makedirs(subsample_path.parent, exist_ok=True)
        with open(subsample_path, 'wb') as f:
            pickle.dump(selected_items, f)
            
        logging.info(f"Created and saved subsample with {len(selected_items)} items")
        return selected_items


if __name__ == "__main__":
    set_verbose(1)
    create_10k_sample()



