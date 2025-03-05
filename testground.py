import re
import time
import torch
import pickle
import json

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from debug import check

data_pkl = Path('cache/queries_item_pool.pkl')
if not data_pkl.exists():
    queries = load_dataset('McAuley-Lab/Amazon-C4')['test']
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
        pickle.dump((item_pool, queries), f)

else:
    with open(data_pkl, 'rb') as f:
        item_pool, queries = pickle.load(f)


print(queries[0])

def build_inverted_index(items):
    """
    Build an inverted index mapping each token (from the metadata field) 
    to a set of item indices.
    """
    inverted_index = defaultdict(set)
    for idx, item in enumerate(tqdm(items, desc='building invereted indexes...', ncols=88)):
        # Get metadata text; default to empty string if missing.
        metadata = item.get('metadata', '')
        # Tokenize: extract words and convert to lowercase.
        tokens = re.findall(r'\w+', metadata.lower())
        for token in tokens:
            inverted_index[token].add(idx)
    return inverted_index

def query_index(inverted_index, query):
    """
    Given an inverted index and a query string, return the set of item indices 
    that contain all tokens in the query.
    """
    tokens = re.findall(r'\w+', query.lower())
    # Retrieve the set of indices for each token found in the query.
    result_sets = [inverted_index[token] for token in tokens if token in inverted_index]
    if not result_sets:
        return set()  # Return empty set if none of the tokens are found.
    # Intersect sets to ensure all tokens are present.
    result = set.intersection(*result_sets)
    return result

def get_items_from_indices(items, indices):
    """
    Retrieve the full items given a list of indices.
    """
    return [items[i] for i in indices]

inverted_index_pkl = Path('cache/item_pool_inverted_index.pkl')
if not inverted_index.exists():
    inverted_index = build_inverted_index(item_pool)
    with open(inverted_index_pkl, 'wb') as f:
        pickle.dump(inverted_index, f)
else:
    with open(inverted_index_pkl, 'rb') as f:
        inverted_index = pickle.load(f)



key = input('type your guess!')
matching_indices = query_index(inverted_index, key)
# matching_items = get_items_from_indices(item_pool, matching_indices)
print("Total matching items:", len(matching_indices))

# Paginate results: for instance, show 10 items per page.
page_size = 10
matching_indices_list = list(matching_indices)
for i in range(0, len(matching_indices_list), page_size):
    page_indices = matching_indices_list[i:i+page_size]
    page_items = get_items_from_indices(item_pool, page_indices)
    print(f"\nPage {i // page_size + 1}:")
    for item in page_items:
        print(item)

    input('...pause...')
