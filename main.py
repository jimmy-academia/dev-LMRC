import json
import logging
from pathlib import Path

from data import load_subsample
from utils import set_verbose, create_llm_client
from utils import loadj, dumpj
from utils import system_struct, user_struct


def prepare_file_tree(item_pool, file_tree_path, call_llm):

    file_tree_path = Path(file_tree_path)
    if file_tree_path.exists():
        logging.info(f'file tree exists, loading {file_tree_path}')
        return loadj(file_tree_path)

    logging.info('file tree does not exists, creating...')
    file_tree = {}
    for item in item_pool:

        prompt_dict = create_prompt_dict(file_tree)

        formatted_prompt = cat_prompt % (
            json.dumps(prompt_dict, indent=2),
            str(item),
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
        current_level["item_ids"].append(item_id)

        print(f"Added item {item_id} to path: {path}")
        dumpj(file_tree, file_tree_path)
        
    return file_tree
    


def main():
    item_count = 200
    call_llm = create_llm_client()
    item_pool = load_subsample()
    file_tree = prepare_file_tree(item_pool, f'cache/file_tree_sample_{item_count}.json', call_llm)

if __name__ == '__main__':
    set_verbose(1)
    main()