import os
import re
import json
import random
import logging
import argparse
import math

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_verbose(verbose):
    # usage: logging.warning; logging.error; logging.info; logging.debug
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    else:
        level = logging.DEBUG  # fallback
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()],
    )

class NamespaceEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, argparse.Namespace):
            return obj.__dict__
        return super().default(obj)

def dumpj(dictionary, filepath):
    with open(filepath, "w") as f:
        obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
        # optional cosmetic reformatting
        obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
        obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
        obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
        f.write(obj)

def loadj(filepath):
    with open(filepath) as f:
        return json.load(f)

def readf(path):
    with open(path, 'r') as f:
        return f.read()

def compute_ndcg_at_50(ranked_items, ground_truth_id):
    """
    Compute NDCG@50 for a single query given:
      - ranked_items: list of item_ids in predicted ranking order
      - ground_truth_id: the correct relevant item_id
    We treat only that item as relevant (1) and the rest as (0).
    """
    limit = 50
    relevances = [1 if item_id == ground_truth_id else 0 for item_id in ranked_items[:limit]]
    dcg = 0.0
    for i, rel in enumerate(relevances, start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    # If relevant item is in top-50, ideal DCG = 1.0 for a single relevant item.
    idcg = 1.0 if ground_truth_id in ranked_items[:limit] else 0.0
    return dcg / idcg if idcg > 0 else 0.0
