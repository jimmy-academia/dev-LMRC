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

def prepare_file_tree_multistep(item_pool, file_tree_path, call_llm, append=False):
    file_tree_path = Path(file_tree_path)

    file_tree = {}
    if file_tree_path.exists():
        file_tree = loadj(file_tree_path)
        if not append:
            logging.info(f'File tree exists, loading {file_tree_path}')
            return file_tree
    else:
        logging.info(f'File tree does not exist, creating to {file_tree_path}...')
    
    # Step 1: Sort items by category (level 1)
    items_by_category = defaultdict(list)
    for item in item_pool:
        if append and find_item_path(item["item_id"], file_tree) is not None:
            continue
        items_by_category[item['category']].append(item)
    
    # Initialize file tree with top-level categories
    for category in items_by_category:
        if category not in file_tree:
            file_tree[category] = {}
    
    # Save initial file tree
    dumpj(file_tree, file_tree_path)
    
    # Process each category
    for category, items in items_by_category.items():
        logging.info(f"Processing category: {category} with {len(items)} items")
        
        # Step 2: Generate level 2 subcategories
        subcategories = generate_level2_subcategories(items, category, call_llm)
        
        # Add level 2 subcategories to file tree
        for subcategory, sub_items in subcategories.items():
            if subcategory not in file_tree[category]:
                file_tree[category][subcategory] = {}
            
            # Step 3: Generate level 3 subcategories for each level 2 category
            if len(sub_items) > 20:
                level3_subcategories = generate_level3_subcategories(sub_items, category, subcategory, call_llm)
                
                # Add level 3 subcategories to file tree
                for level3_subcat, level3_items in level3_subcategories.items():
                    if level3_subcat not in file_tree[category][subcategory]:
                        file_tree[category][subcategory][level3_subcat] = {}
                    
                    # Add items to level 3
                    if "item_ids" not in file_tree[category][subcategory][level3_subcat]:
                        file_tree[category][subcategory][level3_subcat]["item_ids"] = []
                    
                    for item in level3_items:
                        if item['item_id'] not in file_tree[category][subcategory][level3_subcat]["item_ids"]:
                            file_tree[category][subcategory][level3_subcat]["item_ids"].append(item['item_id'])
                            logging.info(f"Added item {item['item_id']} to path: /{category}/{subcategory}/{level3_subcat}")
            else:
                # Add items directly to level 2 if there aren't enough for level 3
                if "item_ids" not in file_tree[category][subcategory]:
                    file_tree[category][subcategory]["item_ids"] = []
                
                for item in sub_items:
                    if item['item_id'] not in file_tree[category][subcategory]["item_ids"]:
                        file_tree[category][subcategory]["item_ids"].append(item['item_id'])
                        logging.info(f"Added item {item['item_id']} to path: /{category}/{subcategory}")
        
        # Save file tree after processing each category
        dumpj(file_tree, file_tree_path)
    
    return file_tree

def generate_level2_subcategories(items, category, call_llm):
    """Generate level 2 subcategories based on item attributes"""
    logging.info(f"Generating level 2 subcategories for {category}")
    
    # First, generate subcategory tags for each item
    subcategory_candidates = []
    existing_subcats = set()
    
    for item in items:
        # Format prompt to generate subcategory for this item
        formatted_prompt = category_gen_prompt.format(
            category=category,
            existing_subcats=json.dumps(list(existing_subcats), indent=2),
            metadata=item['metadata'],
            summary=item.get('summary', item['metadata'])
        )
        
        # Call LLM to get subcategory suggestion
        llm_response = call_llm([system_struct(sys_expert), user_struct(formatted_prompt)])
        try:
            parsed_response = json.loads(llm_response)
            subcategory = parsed_response.get("SubcategoryName", "").strip()
            tags = parsed_response.get("Tags", [])
            
            if subcategory:
                subcategory_candidates.append({
                    "item_id": item["item_id"],
                    "subcategory": subcategory,
                    "tags": tags,
                    "metadata": item['metadata'],
                    "summary": item.get('summary', '')
                })
                existing_subcats.add(subcategory)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for item {item['item_id']}: {llm_response}")
    
    # Cluster the subcategories to get final level 2 categories
    subcategory_count = Counter([sc['subcategory'] for sc in subcategory_candidates])
    logging.info(f"Generated {len(subcategory_count)} unique subcategory candidates")
    
    # Only cluster if we have too many subcategories
    if len(subcategory_count) > 25:
        # Format prompt for clustering subcategories
        formatted_prompt = category_cluster_prompt.format(
            category=category,
            subcategory_candidates=json.dumps(subcategory_candidates, indent=2),
            target_range="5-25"
        )
        
        # Call LLM to get clustering suggestions
        llm_response = call_llm([system_struct(sys_expert), user_struct(formatted_prompt)])
        
        try:
            clustered_subcats = json.loads(llm_response)
            logging.info(f"Clustered into {len(clustered_subcats)} subcategories")
        except json.JSONDecodeError:
            logging.error(f"Failed to parse clustering response: {llm_response}")
            # Fallback: use the most common subcategories
            top_subcats = [sc for sc, count in subcategory_count.most_common(20)]
            clustered_subcats = {subcat: subcat for subcat in top_subcats}
    else:
        # If we have a reasonable number, use them directly
        clustered_subcats = {sc: sc for sc in subcategory_count.keys()}
    
    # Map items to their final subcategories
    items_by_subcategory = defaultdict(list)
    
    for item in items:
        # Find the subcategory assignment for this item
        subcategory = None
        for sc in subcategory_candidates:
            if sc['item_id'] == item['item_id']:
                # Get the mapped/clustered subcategory
                orig_subcat = sc['subcategory']
                if orig_subcat in clustered_subcats:
                    subcategory = clustered_subcats[orig_subcat]
                else:
                    # Find best match among clustered subcategories
                    subcategory = list(clustered_subcats.values())[0]  # Default
                    # Logic to find best match could be added here
                break
        
        # If no subcategory was found, assign to "Other"
        if not subcategory:
            subcategory = "Other"
        
        # Add item to appropriate subcategory
        items_by_subcategory[subcategory].append(item)
    
    return items_by_subcategory

def generate_level3_subcategories(items, category, subcategory, call_llm):
    """Generate level 3 subcategories for items in a level 2 subcategory"""
    logging.info(f"Generating level 3 subcategories for {category}/{subcategory}")
    
    # Similar approach to level 2, but adapted for level 3
    subcategory_candidates = []
    existing_subcats = set()
    
    for item in items:
        # Format prompt to generate level 3 subcategory
        formatted_prompt = category_gen_prompt.format(
            category=f"{category}/{subcategory}",
            existing_subcats=json.dumps(list(existing_subcats), indent=2),
            metadata=item['metadata'],
            summary=item.get('summary', item['metadata'])
        )
        
        # Call LLM to get subcategory suggestion
        llm_response = call_llm([system_struct(sys_expert), user_struct(formatted_prompt)])
        
        try:
            parsed_response = json.loads(llm_response)
            level3_subcat = parsed_response.get("SubcategoryName", "").strip()
            tags = parsed_response.get("Tags", [])
            
            if level3_subcat:
                subcategory_candidates.append({
                    "item_id": item["item_id"],
                    "subcategory": level3_subcat,
                    "tags": tags,
                    "metadata": item['metadata'],
                    "summary": item.get('summary', '')
                })
                existing_subcats.add(level3_subcat)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for item {item['item_id']}")
    
    # Cluster the subcategories if needed
    subcategory_count = Counter([sc['subcategory'] for sc in subcategory_candidates])
    
    # Only cluster if we have too many subcategories
    if len(subcategory_count) > 15:
        # Format prompt for clustering subcategories
        formatted_prompt = category_cluster_prompt.format(
            category=f"{category}/{subcategory}",
            subcategory_candidates=json.dumps(subcategory_candidates, indent=2),
            target_range="5-15"
        )
        
        # Call LLM to get clustering suggestions
        llm_response = call_llm([system_struct(sys_expert), user_struct(formatted_prompt)])
        
        try:
            clustered_subcats = json.loads(llm_response)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse clustering response")
            # Fallback: use the most common subcategories
            top_subcats = [sc for sc, count in subcategory_count.most_common(10)]
            clustered_subcats = {subcat: subcat for subcat in top_subcats}
    else:
        # If we have a reasonable number, use them directly
        clustered_subcats = {sc: sc for sc in subcategory_count.keys()}
    
    # Map items to their final subcategories
    items_by_subcategory = defaultdict(list)
    
    for item in items:
        # Find the subcategory assignment for this item
        subcategory_l3 = None
        for sc in subcategory_candidates:
            if sc['item_id'] == item['item_id']:
                # Get the mapped/clustered subcategory
                orig_subcat = sc['subcategory']
                if orig_subcat in clustered_subcats:
                    subcategory_l3 = clustered_subcats[orig_subcat]
                else:
                    # Find best match among clustered subcategories
                    subcategory_l3 = list(clustered_subcats.values())[0]  # Default
                break
        
        # If no subcategory was found, assign to "Other"
        if not subcategory_l3:
            subcategory_l3 = "Other"
        
        # Add item to appropriate subcategory
        items_by_subcategory[subcategory_l3].append(item)
    
    return items_by_subcategory

def run():
    item_count = 1000
    test_count = 20
    file_tree_path = f'output/file_tree_multistep_{item_count}.json'
    call_llm = create_llm_client()
    item_pool = load_subsample(item_count)
    
    file_tree = prepare_file_tree_multistep(item_pool, file_tree_path, call_llm)
    
    # Load test items
    full_item_pool, requests = load_sample()
    
    # Append test items to the tree if needed
    id_list = [request['item_id'] for request in requests[:test_count]]
    request_items = [item for item in full_item_pool if item['item_id'] in id_list]
    file_tree = prepare_file_tree_multistep(request_items, file_tree_path, call_llm, append=True)
    
    # Test the requests
    correct_count = 0
    total_distance = 0
    
    for request in requests[:test_count]:
        actual_path = find_item_path(request["item_id"], file_tree)
        
        prompt_dict = create_prompt_dict(file_tree)
        
        formatted_prompt = request_prompt.format(
            tree_dict=json.dumps(prompt_dict, indent=2),
            query=request['query'],
            item_id=request['item_id']
        )
        
        response = call_llm([system_struct(sys_expert), user_struct(formatted_prompt)])
        try:
            result = json.loads(response)
            path = result["Path"]
            
            print('==== ====')
            print(f"Request query: {request['query']}")
            
            if actual_path:
                path_components = path.strip('/').split('/')[:-1]  # Remove item_id
                actual_components = actual_path.strip('/').split('/')[:-1]
                is_correct = path_components == actual_components
                
                if is_correct:
                    correct_count += 1
                    print("✓ Correct path")
                    print(f">> The path: {path}")
                else:
                    distance = calculate_path_distance(path, actual_path)
                    total_distance += distance
                    print(f"✗ Incorrect.")
                    print(f">> Predicted Path: {path}\n>> Actual path: {actual_path}\n === Distance: {distance} ===")
            else:
                print(f"{request['item_id']}: Item not found in tree")
        except json.JSONDecodeError:
            print(f"Failed to parse response: {response}")
    
    # Print final results
    print(f"\n===== Results =====")
    print(f"Correct: {correct_count}/{test_count} ({100 * correct_count / test_count:.2f}%)")
    if test_count - correct_count > 0:
        print(f"Average distance for incorrect paths: {total_distance / (test_count - correct_count):.2f}")