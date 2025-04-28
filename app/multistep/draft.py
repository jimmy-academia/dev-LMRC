import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
import itertools

from data import load_subsample, load_sample
from utils import create_llm_client
from utils import loadj, dumpj
from utils import system_struct, user_struct

from .helper import create_prompt_dict, find_item_path, calculate_path_distance
from .prompt import sys_expert, category_gen_prompt, category_cluster_prompt, request_prompt

def multi_file_tree(item_pool, file_tree_path, call_llm, append=False):
    
    current_level = 0
    file_tree = defaultdict(list)
    log_file_tree = []

    suitable_size = 20 # stop branching when node contain less than this many items
    max_level = 7 # at most 7 levels

    for item in item_pool:
        file_tree[item['category']].append(item)

    recurrsive_organize(file_tree, call_llm, log_file_tree, suitable_size)

def recurrsive_organize(file_tree, call_llm, log_file_tree, suitable_size, root=''):
    for key, item_list in file_tree.values():
        if len(item_list) < suitable_size:
            file_tree[key] = {"item_ids":[item["item_id"] for item in item_list]}
        else:
            branch_root = root + '/' + key
            branch_list = create_branches(item_list, branch_root)
            branch_list = organize_branches(branch_list, branch_root)
            the_stump = defaultdict(list)
            for branch, item in zip(branch_list, item_list):
                the_stump[branch].append(item)

            file_tree[key] = the_stump
            recurrsive_organize(file_tree[key], call_llm, )

    return file_tree

def run(args, item_pool, requests):
    
    call_llm = create_llm_client(model=args.model)
    ensure_dir('app/multistep/output')
    ensure_dir('app/multistep/output/log')
    file_tree_path = f'app/multistep/output/file_tree_{len(item_pool)}.json'
    log_tree_path = f'app/multistep/output/log/file_tree_{len(item_pool)}.json'
    record_path = f'app/multistep/output/request_{len(item_pool)}.json'
    log_request_path = f'app/multistep/output/log/request_{len(item_pool)}.json'
    
    file_tree = multi_file_tree(item_pool, file_tree_path, log_tree_path, call_llm)
    file_tree_usage = call_llm.get_usage()
    logging.info(f"File tree preparation usage: {file_tree_usage}")
    
    call_llm.reset_usage()

    fulfill_requests(file_tree, requests, record_path, log_request_path, call_llm)
    request_usage = call_llm.get_usage()
    logging.info(f"Request fulfillment usage: {request_usage}")    

    # Calculate and log total usage
    total_cost = file_tree_usage["cost"] + request_usage["cost"]
    total_tokens = file_tree_usage["total_tokens"] + request_usage["total_tokens"]
    logging.info(f"Total cost: ${total_cost:.4f}, Total tokens: {total_tokens}")

