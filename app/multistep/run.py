import json
import logging
from collections import defaultdict
from datetime import datetime

from utils import create_llm_client, ensure_dir, loadj, dumpj, system_struct, user_struct
from .helper import create_prompt_dict, find_item_path, calculate_path_distance
from .prompt import sys_expert, category_gen_prompt, category_cluster_prompt, request_prompt

# Global parameters
SUITABLE_SIZE = 20
MAX_LEVEL = 7
LLM_LOG = []

def logged_llm_call(messages, call_llm, **kwargs):
    """Log LLM calls and return response."""
    response = call_llm(messages, **kwargs)
    LLM_LOG.append({
        "timestamp": datetime.now().isoformat(),
        "prompt": messages[-1]['content'] if messages else "",
        "response": response
    })
    return response

def create_branches(items, path, call_llm):
    pass

def organize_branches(branches, items, path, call_llm):
    pass

def recursive_organize(items, call_llm, path='', level=0):
    """Recursively organize items into a hierarchical structure."""
    if len(items) < SUITABLE_SIZE or level >= MAX_LEVEL:
        return {"item_ids": [item["item_id"] for item in items]}
    
    branch_list = create_branches(items, path, call_llm)
    branch_list = organize_branches(branch_list, items, path, call_llm)
    
    # Group items by branch
    branch_items = defaultdict(list)
    for item, branch in zip(items, branch_list):
        branch_items[branch].append(item)
    
    # Process each branch recursively
    result = {}
    for branch, branch_items in branch_items.items():
        new_path = f"{path}/{branch}" if path else branch
        result[branch] = recursive_organize(branch_items, call_llm, new_path, level + 1)
    
    return result

def multi_file_tree(item_pool, file_tree_path, log_path, call_llm):
    """Build a hierarchical file tree from item pool."""
    global LLM_LOG
    LLM_LOG = []
    
    # Group by top-level category
    categories = defaultdict(list)
    for item in tqdm(item_pool, desc='initial dump', ncols=88):
        categories[item['category']].append(item)
    
    # Process each category
    result = {}
    for category, items in tqdm(categories.items(), desc="reccursive...", ncols=88):
        result[category] = recursive_organize(items, call_llm, category, 1)
    
    # Save results
    dumpj(result, file_tree_path)
    dumpj(LLM_LOG, log_path)
    
    return result

def fulfill_requests(file_tree, requests, record_path, log_path, call_llm):
    """Process user requests against the file tree."""
    global LLM_LOG
    LLM_LOG = []
    
    results = []
    prompt_dict = create_prompt_dict(file_tree)
    
    for request in requests:
        actual_path = find_item_path(request["item_id"], file_tree)
        
        formatted_prompt = request_prompt.format(
            json.dumps(prompt_dict, indent=2), 
            request['query'], 
            request['item_id']
        )
        
        response = logged_llm_call([system_struct(sys_expert), user_struct(formatted_prompt)], call_llm)
        
        try:
            result = json.loads(response)
            predicted_path = result["Path"]
            
            is_correct = False
            distance = None
            
            if actual_path:
                pred_components = predicted_path.strip('/').split('/')[:-1]
                actual_components = actual_path.strip('/').split('/')[:-1]
                is_correct = pred_components == actual_components
                
                if not is_correct:
                    distance = calculate_path_distance(predicted_path, actual_path)
            
            results.append({
                "item_id": request["item_id"],
                "query": request["query"],
                "predicted_path": predicted_path,
                "actual_path": actual_path,
                "is_correct": is_correct,
                "distance": distance
            })
            
            logging.info(f"Request: {request['query']}")
            logging.info(f"{'✓ Correct' if is_correct else f'✗ Incorrect (Distance: {distance})'}")
            
        except Exception as e:
            logging.error(f"Error processing request {request['item_id']}: {e}")
    
    dumpj(results, record_path)
    dumpj(LLM_LOG, log_path)
    
    return results

def run(args, item_pool, requests):
    call_llm = create_llm_client(model=args.model)
    ensure_dir('app/multistep/output')
    ensure_dir('app/multistep/output/log')
    
    file_tree_path = f'app/multistep/output/file_tree_{len(item_pool)}.json'
    log_tree_path = f'app/multistep/output/log/file_tree_{len(item_pool)}.json'
    record_path = f'app/multistep/output/request_{len(item_pool)}.json'
    log_request_path = f'app/multistep/output/log/request_{len(item_pool)}.json'
    
    input('work on fulfill_requests!!!')

    file_tree = multi_file_tree(item_pool, file_tree_path, log_tree_path, call_llm)
    file_tree_usage = call_llm.get_usage()
    logging.info(f"File tree usage: {file_tree_usage}")
    
    call_llm.reset_usage()
    fulfill_requests(file_tree, requests, record_path, log_request_path, call_llm)
    request_usage = call_llm.get_usage()
    logging.info(f"Request usage: {request_usage}")
    
    total_cost = file_tree_usage["cost"] + request_usage["cost"]
    total_tokens = file_tree_usage["total_tokens"] + request_usage["total_tokens"]
    logging.info(f"Total: ${total_cost:.4f}, {total_tokens} tokens")