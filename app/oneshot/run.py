import json
import logging
from pathlib import Path

from utils import create_llm_client
from utils import loadj, dumpj
from utils import system_struct, user_struct
from utils import ensure_dir

from .helper import create_prompt_dict, find_item_path, calculate_path_distance
from .prompt import sys_expert, cat_prompt, request_prompt

def prepare_file_tree(item_pool, file_tree_path, log_tree_path, call_llm):

    file_tree_path = Path(file_tree_path)

    file_tree = {}
    log_file_tree = []

    call_llm.reset_usage()

    if file_tree_path.exists():
        logging.warning(f'file tree exists, loading {file_tree_path}')
        return loadj(file_tree_path)
        input('paused')

    logging.info(f'file tree does not exists, creating to {file_tree_path}...')
    for item in item_pool:

        prompt_dict = create_prompt_dict(file_tree)

        formatted_prompt = cat_prompt % (
            json.dumps(prompt_dict, indent=2),
            item['metadata'],
            item['summary'],
            item['category'],
            item['item_id']
        )

        # print(formatted_prompt)
        
        llm_response = call_llm([system_struct(sys_expert), user_struct(formatted_prompt)])
        parsed_response = json.loads(llm_response)
        parsed_response["accumulative_usage"] = call_llm.get_usage()
        path = parsed_response["Path"]
        log_file_tree.append(parsed_response)
        dumpj(log_file_tree, log_tree_path)

        # Split the path into components
        path_components = path.strip('/').split('/')
        
        # Update the directory structure
        current_level = file_tree
        for i in range(len(path_components) - 1):  # Skip the last component which is the item ID
            component = path_components[i]
            if component not in current_level:
                current_level[component] = {}
            current_level = current_level[component]

        # Add the item_id at the lowest level
        item_id = item['item_id']
        if item_id != path_components[-1]:
            logging.warning(f'LLM hallucinated {path_components[-1]} for {item_id}!')
        if "item_ids" not in current_level:
            current_level["item_ids"] = []
        # current_level["item_ids"].append(item_id + ":" + item['summary']) 
        current_level["item_ids"].append(item_id)

        logging.info(f"Added item {item_id} to path: {path}")
        dumpj(file_tree, file_tree_path)
        
    return file_tree


def fulfill_requests(file_tree, requests, record_path, log_request_path, call_llm):
    log_result = []
    Record = {'total':0, 'correct': 0, 'notfounderror': 0, 'messages':[]}

    call_llm.reset_usage()

    for request in requests:

        actual_path = find_item_path(request["item_id"], file_tree)

        prompt_dict = create_prompt_dict(file_tree)

        formatted_prompt = request_prompt.format(json.dumps(prompt_dict, indent=2), request['query'], request['item_id'])

        response = call_llm([system_struct(sys_expert), user_struct(formatted_prompt)])
        result = json.loads(response)
        result["usage"] = call_llm.get_usage()

        log_result.append(result)
        dumpj(log_result, log_request_path)

        path = result["Path"]
        is_correct = False
        print('==== ====')
        print(f"Request query: {request['query']}")
        if actual_path:
            path_components = path.strip('/').split('/')[:-1]  # Remove item_id
            actual_components = actual_path.strip('/').split('/')[:-1]
            is_correct = path_components == actual_components
            
            Record['total'] += 1 

            if is_correct:
                msg = f"✓ Correct path: {path}"
                Record['correct'] += 1 
            else:
                distance = calculate_path_distance(path, actual_path)
                msg = f"✗ Incorrect. Predicted: {path}, Actual: {actual_path}, Distance: {distance}"
        else:
            msg = f"{request["item_id"]}: Item not found in tree"
            Record['notfounderror'] += 1

        print(msg)
        Record['messages'].append(msg)
        Record['accuracy'] = Record['correct']/Record['total']
        dumpj(Record, record_path)

def run(args, item_pool, requests):
    
    call_llm = create_llm_client(model=args.model)
    ensure_dir('app/oneshot/output')
    ensure_dir('app/oneshot/output/log')
    file_tree_path = f'app/oneshot/output/file_tree_sample_{len(item_pool)}.json'
    log_tree_path = f'app/oneshot/output/log/file_tree_sample_{len(item_pool)}.json'
    record_path = f'app/oneshot/output/request_sample_{len(item_pool)}.json'
    log_request_path = f'app/oneshot/output/log/request_sample_{len(item_pool)}.json'

    file_tree = prepare_file_tree(item_pool, file_tree_path, log_tree_path, call_llm)
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



    """
    # if append and find_item_path(item["item_id"], file_tree) is not None:
            # continue

    """