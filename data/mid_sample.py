import pickle
import logging
import random
import os
from pathlib import Path
from collections import Counter, defaultdict

from utils import create_llm_client, system_struct, user_struct

def create_10k_sample(
    data_pkl='cache/queries_item_pool.pkl',
    output_pkl='cache/mid_sample_10k.pkl',
    item_count=10000,
    request_count=2000,
    quality_check=True,
    pause_frequency=1
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
        manifest, item_pool, request_count, llm_client, pause_frequency
    )

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
    return manifest_out

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
    target_counts = {
        cat: max(int((count / total_available) * total_count), 1)
        for cat, count in category_counts.items()
    }

    print("\nTotal Item Category Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    selected_items = []
    for cat, count in target_counts.items():
        available = items_by_category[cat]
        selected_items.extend(random.sample(available, min(len(available), count)))

    print("\nSampled Item Category Distribution:")
    sampled_counts = Counter([item.get('category', 'Unknown') for item in selected_items])
    for cat, count in sorted(sampled_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    return selected_items

def sample_requests_by_distribution(manifest, item_pool, total_count, llm_client=None, pause_frequency=1):
    item_by_id = {item['item_id']: item for item in item_pool}
    item_categories = {item['item_id']: item.get('category', 'Unknown') for item in item_pool}

    with open(manifest["requests_file"], 'rb') as f:
        all_requests = [r for r in pickle.load(f) if r['item_id'] in item_by_id]

    category_requests = defaultdict(list)
    for req in all_requests:
        cat = item_categories[req['item_id']]
        category_requests[cat].append(req)

    total_requests_available = sum(len(reqs) for reqs in category_requests.values())
    target_counts = {
        cat: max(int((len(reqs) / total_requests_available) * total_count), 1)
        for cat, reqs in category_requests.items()
    }

    print("\nTotal Request Category Distribution:")
    for cat, count in sorted({cat: len(reqs) for cat, reqs in category_requests.items()}.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

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
                continue

            decision, llm_output = True, ""
            if llm_client:
                decision, llm_output = check_request_quality(req, item, llm_client)
                if not decision:
                    continue

            # Pause for user confirmation
            if pause_frequency and (checked % pause_frequency == 0):
                print("\n\n=== PAUSE FOR CONFIRMATION ===")
                print(f"Checked: {checked}, Selected: {len(selected_requests)}")
                print(f"Sampled Request Category Distribution: {Counter([item_categories[r['item_id']] for r in selected_requests])}")
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    create_10k_sample()
