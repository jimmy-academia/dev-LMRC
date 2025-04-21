import json
import logging
from pathlib import Path

from data import load_subsample, load_sample
from utils import set_verbose, create_llm_client
from utils import loadj, dumpj
from utils import system_struct, user_struct

from helper import create_prompt_dict, find_item_path, calculate_path_distance
from prompt import sys_expert, cat_prompt, request_prompt

def prepare_file_tree(item_pool, file_tree_path, call_llm, append=False):

    file_tree_path = Path(file_tree_path)

    file_tree = {}
    if file_tree_path.exists():
        file_tree = loadj(file_tree_path)
        if not append:
            logging.info(f'file tree exists, loading {file_tree_path}')
            return file_tree
    else:
        logging.info(f'file tree does not exists, creating to {file_tree_path}...')
        
    for item in item_pool:

        if append and find_item_path(item["item_id"], file_tree) is not None:
            continue

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
        path = parsed_response["Path"]
        
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
    


def main():
    item_count = 200
    test_count = 20
    file_tree_path = f'output/file_tree_sample_{item_count}.json'
    call_llm = create_llm_client()
    item_pool = load_subsample(item_count)
    file_tree = prepare_file_tree(item_pool, file_tree_path, call_llm)

    full_item_pool, requests =  load_sample()

    ## append items
    id_list = [request['item_id'] for request in requests[:test_count]]
    request_items = [item for item in full_item_pool if item['item_id'] in id_list]
    file_tree = prepare_file_tree(request_items, file_tree_path, call_llm, append=True)

    for request in requests[:test_count]:

        actual_path = find_item_path(request["item_id"], file_tree)

        prompt_dict = create_prompt_dict(file_tree)

        formatted_prompt = request_prompt.format(json.dumps(prompt_dict, indent=2), request['query'], request['item_id'])

        response = call_llm([system_struct(sys_expert), user_struct(formatted_prompt)])
        result = json.loads(response)

        path = result["Path"]
        is_correct = False
        print('==== ====')
        print(f"Request query: {request['query']}")
        if actual_path:
            path_components = path.strip('/').split('/')[:-1]  # Remove item_id
            actual_components = actual_path.strip('/').split('/')[:-1]
            is_correct = path_components == actual_components
            
            if is_correct:
                print("✓ Correct path")
                print(f">> The path: {path}")
            else:
                distance = calculate_path_distance(path, actual_path)
                print(f"✗ Incorrect.")
                print(f">> Predicted Path: {path}\n>> Actual path: {actual_path}\n === Distance: {distance} ===")
        else:
            print(f"{request["item_id"]}: Item not found in tree")
        



if __name__ == '__main__':
    set_verbose(1)
    main()

