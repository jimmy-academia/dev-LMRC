import json
import logging
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from utils import create_llm_client, ensure_dir, loadj, dumpj, system_struct, user_struct
from .helper import create_prompt_dict, find_item_path, calculate_path_distance
from .prompt import sys_expert, subcategory_prompt, consolidation_prompt, navigate_prompt

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
    """
    Create appropriate branches for a set of items.
    
    Args:
        items: List of item dictionaries with metadata and summary
        path: Current path in the hierarchy
        call_llm: Function to call the LLM
        
    Returns:
        List of branch names corresponding to each item
    """
    # Extract the current hierarchy from the path
    path_components = path.strip('/').split('/') if path else []
    current_level = len(path_components)
    
    # Get existing subcategories from previously processed items
    existing_subcats = set()
    branch_list = []
    
    # Format prompt template for getting subcategories
    
    for i, item in enumerate(items):
        # Create a prompt for this item
        prompt = subcategory_prompt.format(
            current_path=path,
            level=current_level + 1,
            existing_branches=', '.join(sorted(existing_subcats)) if existing_subcats else "None yet",
            metadata=item['metadata'],
            summary=item['summary']
        )
        
        # Call LLM to get subcategory suggestion
        response = logged_llm_call([
            system_struct(sys_expert), 
            user_struct(prompt)
        ], call_llm)
        
        try:
            result = json.loads(response)
            branch_name = result["BranchName"]
            
            # Add to our results
            branch_list.append(branch_name)
            existing_subcats.add(branch_name)
            
            # Log progress periodically
            if (i + 1) % len(items)//10 == 0 or i == len(items) - 1:
                logging.info(f"Processed {i + 1}/{len(items)} items at path '{path}', found {len(existing_subcats)} branches")
                
        except Exception as e:
            logging.error(f"Error processing item {item['item_id']} at path '{path}': {e}")
            # Use a generic branch name if there's an error
            default_branch = "Other"
            branch_list.append(default_branch)
            existing_subcats.add(default_branch)
    
    return branch_list

def organize_branches(branches, items, path, call_llm):
    """
    Organize and consolidate branches if there are too many, considering the full path context.
    
    Args:
        branches: List of branch names for each item
        items: List of item dictionaries
        path: Current path in the hierarchy (can be multiple levels deep)
        call_llm: Function to call the LLM
        
    Returns:
        Consolidated list of branch names for each item
    """
    from utils import system_struct, user_struct
    
    # If we have a reasonable number of branches, just return them
    unique_branches = set(branches)
    if len(unique_branches) <= 10:
        return branches
    
    # Extract the hierarchy from the path
    path_components = path.strip('/').split('/') if path else []
    current_level = len(path_components)
    
    # Group items by branch
    items_by_branch = defaultdict(list)
    for item, branch in zip(items, branches):
        items_by_branch[branch].append(item)
    
    # Create a simplified representation of each branch for the prompt
    branch_candidates = []
    for branch, branch_items in items_by_branch.items():
        # Take up to 3 sample items for each branch
        samples = branch_items[:3]
        sample_texts = [f"- {s['summary']}" for s in samples]
        branch_candidates.append(f"{branch} ({len(branch_items)} items):\n" + "\n".join(sample_texts))
    
    # Create a custom consolidation prompt
    
    prompt = consolidation_prompt.format(
        current_path=path,
        level=current_level + 1,
        branch_count=len(unique_branches),
        branch_candidates="\n\n".join(branch_candidates)
    )
    
    # Call LLM to get clustering suggestion
    response = logged_llm_call([
        system_struct(sys_expert), 
        user_struct(prompt)
    ], call_llm)
    
    try:
        # Parse the mapping
        mapping = json.loads(response)
        
        # Apply the mapping to the branches
        consolidated_branches = [mapping.get(branch, branch) for branch in branches]
        
        unique_consolidated = set(consolidated_branches)
        logging.info(f"Path '{path}': Consolidated from {len(unique_branches)} to {len(unique_consolidated)} branches")
        
        return consolidated_branches
        
    except Exception as e:
        logging.error(f"Error consolidating branches at path '{path}': {e}")
        # If there's an error, just return the original branches
        return branches

def recursive_organize(items, call_llm, path='', level=0):
    """Recursively organize items into a hierarchical structure."""
    if len(items) < SUITABLE_SIZE or level >= MAX_LEVEL:
        return {"item_ids": [item["item_id"] for item in items]}
    
    logging.info(f"Level {level}: Organizing {len(items)} items at path '{path}'")
    
    # Create branches for all items
    branch_list = create_branches(items, path, call_llm)
    
    # Organize/consolidate branches if needed
    branch_list = organize_branches(branch_list, items, path, call_llm)
    
    # Group items by branch
    branch_items = defaultdict(list)
    for item, branch in zip(items, branch_list):
        branch_items[branch].append(item)
    
    # Process each branch recursively
    result = {}
    for branch, branch_items_list in branch_items.items():
        new_path = f"{path}/{branch}" if path else branch
        result[branch] = recursive_organize(branch_items_list, call_llm, new_path, level + 1)
    
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
    """
    Process user requests against the file tree using a multi-step approach.
    Navigate through the hierarchy one level at a time for better precision.
    """
    
    global LLM_LOG
    LLM_LOG = []
    
    results = []
    
    # Create a navigation prompt template
    
    for request in requests:
        item_id = request["item_id"]
        query = request["query"]
        actual_path = find_item_path(item_id, file_tree)
        
        if not actual_path:
            logging.warning(f"Item {item_id} not found in file tree. Skipping request.")
            results.append({
                "item_id": item_id,
                "query": query,
                "predicted_path": None,
                "actual_path": None,
                "steps_taken": [],
                "is_correct": False,
                "error": "Item not found in file tree"
            })
            continue
        
        # Start navigation from the root
        current_node = file_tree
        current_path = ""
        navigation_steps = []
        
        # Navigate until we reach a leaf node (with "item_ids") or can't navigate further
        while "item_ids" not in current_node:
            # Get available branches at this level
            available_branches = [branch for branch in current_node.keys() if branch != "item_ids"]
            
            if not available_branches:
                break
                
            # Format the navigation prompt
            prompt = navigate_prompt.format(
                current_path=current_path if current_path else "/ (root)",
                available_branches=", ".join(available_branches),
                query=query
            )
            
            # Call LLM to get branch selection
            response = logged_llm_call([
                system_struct(sys_expert), 
                user_struct(prompt)
            ], call_llm)
            
            try:
                result = json.loads(response)
                selected_branch = result["SelectedBranch"]
                
                # Check if the selected branch exists
                if selected_branch not in available_branches:
                    logging.warning(f"Selected branch '{selected_branch}' not found. Using first available branch.")
                    selected_branch = available_branches[0]
                
                # Record the navigation step
                navigation_steps.append({
                    "path": current_path,
                    "available_branches": available_branches,
                    "selected_branch": selected_branch,
                    "reasoning": result.get("Reasoning", "")
                })
                
                # Update current node and path
                current_node = current_node[selected_branch]
                current_path = f"{current_path}/{selected_branch}" if current_path else selected_branch
                
            except Exception as e:
                logging.error(f"Error navigating for request {item_id}: {e}")
                break
        
        # Construct the final predicted path
        predicted_path = f"{current_path}/{item_id}" if current_path else f"/{item_id}"
        
        # Check if prediction is correct
        is_correct = False
        distance = None
        
        if actual_path:
            pred_components = predicted_path.strip('/').split('/')[:-1]
            actual_components = actual_path.strip('/').split('/')[:-1]
            is_correct = pred_components == actual_components
            
            if not is_correct:
                distance = calculate_path_distance(predicted_path, actual_path)
        
        result = {
            "item_id": item_id,
            "query": query,
            "predicted_path": predicted_path,
            "actual_path": actual_path,
            "steps_taken": navigation_steps,
            "is_correct": is_correct,
            "distance": distance
        }
        
        results.append(result)
        
        # Log outcome for this request
        status = "✓ Correct" if is_correct else f"✗ Incorrect (Distance: {distance})"
        logging.info(f"Request: {query}")
        logging.info(f"Path: {predicted_path}")
        logging.info(f"Status: {status}")
    
    # Save results and logs
    dumpj(results, record_path)
    dumpj(LLM_LOG, log_path)
    
    # Calculate and return statistics
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    logging.info(f"Request fulfillment complete. Accuracy: {accuracy:.2f} ({correct_count}/{total_count})")
    
    return results

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
    logging.info(f"File tree usage: {file_tree_usage}")
    
    call_llm.reset_usage()
    fulfill_requests(file_tree, requests, record_path, log_request_path, call_llm)
    request_usage = call_llm.get_usage()
    logging.info(f"Request usage: {request_usage}")
    
    total_cost = file_tree_usage["cost"] + request_usage["cost"]
    total_tokens = file_tree_usage["total_tokens"] + request_usage["total_tokens"]
    logging.info(f"Total: ${total_cost:.4f}, {total_tokens} tokens")

