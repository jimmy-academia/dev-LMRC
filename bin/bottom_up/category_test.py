import json
from pathlib import Path
import logging
from data import load_category_sample
from utils import set_verbose, create_llm_client
from utils import loadj, dumpj
from utils import system_struct, user_struct

sys_expert = "You are an expert in categorizing items into a hiearchical file path."
cat_prompt = """Start with root, and use category as the first path. 

- In level names, replace space with underscores.
- You must use all three levels
- Device the most generic names for each level subcategory; ensure that the 2nd and 3rd level subcategories are inclusive to other items; Do not use special brand names as subcategory.
- Use the existing directory if possible, or branch off any level if the category should be different.

Existing directory:
%s

What is the three level path of the following item?

%s


Return in the following format:
{
    "Reasoning": "....",
    "Path": "/%s/2nd-level/3rd-level/%s"
}"""


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
    
def create_prompt_dict(file_tree):
    """
    Create a version of the directory dictionary without item_ids for the prompt.
    """
    prompt_dict = {}
    
    def copy_without_items(src, dest):
        for key, value in src.items():
            if key != "item_ids":
                if isinstance(value, dict):
                    dest[key] = {}
                    copy_without_items(value, dest[key])
                else:
                    dest[key] = value
    
    copy_without_items(file_tree, prompt_dict)
    return prompt_dict 


def generate_path_for_request(request, file_tree, call_llm):
    """
    Generate a hierarchical path for a request using LLM.
    
    Args:
        request: Dictionary containing query, item_id, etc.
        file_tree: Current file tree structure
        call_llm: Function to call the LLM
    
    Returns:
        Dictionary with the generated path and reasoning
    """
    # Format the tree for the prompt
    tree_dict = create_prompt_dict(file_tree)
    
    # Create the prompt
    prompt = f"""You are an expert in finding the most appropriate location in an existing hierarchical directory.

Existing directory structure:
{json.dumps(tree_dict, indent=2)}

A user has described a product with this query:
"{request['query']}"

Your task is to:
1. Analyze the query carefully to understand what the product is
2. Examine the EXISTING directory structure
3. Find the BEST MATCHING path within the existing structure (don't create new paths)
4. Choose the most appropriate 3-level deep location (/level1/level2/level3/) where this item belongs

The product has ID: {request['item_id']}

Return your answer in the following format:
{{
    "Reasoning": "Your step-by-step analysis of why this path is appropriate",
    "Path": "/existing-level1/existing-level2/existing-level3/{request['item_id']}"
}}
"""    
    # Call the LLM
    response = call_llm([system_struct(sys_expert), user_struct(prompt)])
    
    # Parse the response
    result = json.loads(response)
    return result


def find_item_path(item_id, tree, path=""):
    """Find the path to an item in the tree."""
    # Check current level
    if "item_ids" in tree and item_id in tree["item_ids"]:
        return f"{path}/{item_id}"
        
    # Check subdirectories
    for key, value in tree.items():
        if key != "item_ids" and isinstance(value, dict):
            result = find_item_path(item_id, value, f"{path}/{key}")
            if result:
                return result
                
    return None

def calculate_path_distance(path1, path2):
    """
    Calculate a distance metric between two paths.
    
    Args:
        path1: First path string
        path2: Second path string
        
    Returns:
        int: Distance metric (0 = identical paths)
    """
    # Split paths into components
    components1 = path1.strip('/').split('/')
    components2 = path2.strip('/').split('/')
    
    # Remove item_id from calculation
    if len(components1) > 0:
        components1 = components1[:-1]
    if len(components2) > 0:
        components2 = components2[:-1]
    
    # Calculate common prefix length
    common_prefix = 0
    for i in range(min(len(components1), len(components2))):
        if components1[i] == components2[i]:
            common_prefix += 1
        else:
            break
    
    # Distance = unique components in both paths
    distance = (len(components1) - common_prefix) + (len(components2) - common_prefix)
    return distance

def main():
    catetegory = 'Food'
    call_llm = create_llm_client()
    item_pool, requests = load_category_sample(catetegory)

    file_tree = prepare_file_tree(item_pool, f'cache/file_tree_{catetegory}_sample.json', call_llm)

    for request in requests:
        result = generate_path_for_request(request, file_tree, call_llm)
        path = result["Path"]
        actual_path = find_item_path(request["item_id"], file_tree)
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
            print("? Item not found in tree")
        

if __name__ == '__main__':
    set_verbose(1)
    main()