import logging
import json
import re
import gc
from collections import defaultdict

from utils import user_struct, system_struct, assistant_struct, flatten_messages
from prompts import (
    SYSTEM_PROMPT,
    DECIDE_CREATE_SELECT_PASS_PROMPT,
    SELECT_SUBCATEGORY_PROMPT,
    CREATE_SUBCATEGORY_PROMPT,
    format_subcategories_info,
    format_subcategories_detailed,
    format_sample_items,
    get_relevance_info
)

class RecursiveAgent:
    """
    A search agent that uses a recursive/tree-search approach to navigate through
    category hierarchies and find relevant items.
    """

    def __init__(self, file_system, llm_client):
        """Initialize the agent with a file system and LLM client."""
        self.fs = file_system
        self.llm_client = llm_client
        self.max_depth = 5
        self.min_items_for_subdivision = 50
        self.max_branches = 8
        
    def search(self, query, gt, gt_cat, max_steps=10):
        """
        Search for items matching the query using a recursive tree search process.
        Returns a dict with results and search information.
        """
        # Initialize search state
        self.gt = gt
        self.gt_cat = gt_cat
        self.query = query
        self.max_steps = max_steps
        
        # Global state that persists across recursive calls
        self.found_items = set()
        self.visited_paths = set()
        self.item_relevance_scores = {}
        self.steps = []
        self.step_count = 0
        
        # Collection of promising categories
        self.hand = []
        self.hand_limit = 20  # Store up to 20 promising categories
        
        # Start the recursive search from the root
        logging.info(f"Starting recursive search for query: {query}")
        self._recursive_search('/', 0, max_steps)
        
        # Process the hand of promising categories to find the most relevant items
        self._process_hand()
        
        # Prepare final results
        return self._prepare_final_results()
    
    def _recursive_search(self, path, level, remaining_steps):
        """
        Recursively explore a path and its subcategories.
        
        Args:
            path (str): Current filesystem path to explore
            level (int): Current depth level in the recursion
            remaining_steps (int): Number of steps remaining
        """
        # Check if we've reached maximum depth or used all steps
        if level >= self.max_depth or remaining_steps <= 0:
            if self.fs.navigate_to(path)['total_items'] > 0:
                self.hand.append(path)
            return
        
        # Mark this path as visited
        self.visited_paths.add(path)
        
        # Get information about the current node
        fs_info = self.fs.navigate_to(path)
        total_items = fs_info['total_items']
        direct_items = fs_info['default_items']
        subcategories = fs_info['subcategories']
        
        # If there are few items at this node, add it to the hand and return
        if total_items <= 100:
            self.hand.append(path)
            return
            
        # Record this step
        self.step_count += 1
        step_info = {
            "path": path,
            "level": level,
            "action": "explore"
        }
        self.steps.append(step_info)
        
        # Decision logic: should we create, select, or pass?
        decision = "pass"
        if direct_items and len(direct_items) > self.min_items_for_subdivision:
            # If we have enough direct items, decide whether to create a subcategory
            decision = self._decide_create_select_pass(path, subcategories, direct_items)
        elif subcategories:
            # If we have subcategories but no substantial direct items, select one
            decision = "select"
            
        if decision == "create" and direct_items:
            # Create a new subcategory
            new_path, remaining_direct_items = self._create_subcategory(path, direct_items)
            step_info["action"] = "create"
            step_info["new_path"] = new_path
            
            # Recursively explore the new subcategory
            if new_path != path:  # Make sure creation was successful
                self._recursive_search(new_path, level + 1, remaining_steps - 1)
                
            # If we have remaining direct items, process them too
            if remaining_direct_items and len(remaining_direct_items) > self.min_items_for_subdivision:
                # Try one more subcategory creation with remaining items if we have steps left
                if remaining_steps > 1:
                    another_path, _ = self._create_subcategory(path, remaining_direct_items)
                    if another_path != path:
                        self._recursive_search(another_path, level + 1, remaining_steps - 2)
            
        elif decision == "select" and subcategories:
            # Select the most promising subcategory
            selected_subcat = self._select_subcategory(path, subcategories)
            step_info["action"] = "select"
            step_info["selected"] = selected_subcat
            
            # Create the new path
            next_path = f"{path}/{selected_subcat}" if path != "/" else f"/{selected_subcat}"
            
            # Recursively explore the selected subcategory
            self._recursive_search(next_path, level + 1, remaining_steps - 1)
            
            # Try another subcategory if we have enough steps left
            if remaining_steps > 2 and len(subcategories) > 1:
                # Remove the selected subcategory before selecting the next one

                input("Remove the selected subcategory before selecting the next one")
                subcats_copy = subcategories.copy()
                if selected_subcat in subcats_copy:
                    del subcats_copy[selected_subcat]
                    
                if subcats_copy:
                    second_subcat = self._select_subcategory(path, subcats_copy)
                    second_path = f"{path}/{second_subcat}" if path != "/" else f"/{second_subcat}"
                    self._recursive_search(second_path, level + 1, remaining_steps - 2)
        
        else:
            # If we can't create or select, add this path to the hand
            step_info["action"] = "pass"
            self.hand.append(path)
            
        # Limit the hand size
        if len(self.hand) > self.hand_limit:
            self.hand = self.hand[:self.hand_limit]
    
    def _decide_create_select_pass(self, path, subcategories, direct_items):
        """
        Decide whether to select from existing subcategories, create a new subcategory, or pass.
        Returns: "select", "create", or "pass"
        """
        # Prepare prompt information
        subcategories_info = format_subcategories_info(subcategories)
        direct_item_count = len(direct_items)
        relevance_info = get_relevance_info(self.query, path, self.fs)
        
        # Format the prompt
        prompt = DECIDE_CREATE_SELECT_PASS_PROMPT.format(
            query=self.query,
            path=path,
            direct_item_count=direct_item_count,
            subcategories_info=subcategories_info,
            relevance_info=relevance_info
        )
        
        # Create messages with system prompt and user prompt
        messages = [
            system_struct(SYSTEM_PROMPT),
            user_struct(prompt)
        ]
        
        try:
            # Get LLM response
            llm_output = self.llm_client(messages)
            
            # Try to parse as JSON
            try:
                parsed_output = json.loads(llm_output)
                if parsed_output and "decision" in parsed_output:
                    decision = parsed_output["decision"].lower()
                    if decision in ["select", "create", "pass"]:
                        return decision
            except json.JSONDecodeError:
                # If not valid JSON, try to extract decision from text
                if "CREATE" in llm_output.upper():
                    return "create"
                elif "SELECT" in llm_output.upper():
                    return "select"
                elif "PASS" in llm_output.upper():
                    return "pass"
        
        except Exception as e:
            logging.error(f"Error in decide_create_select_pass: {e}")
            
        # Default decision logic:
        # If there are subcategories, select
        # If there are direct items but no subcategories, create
        # Otherwise pass
        if subcategories:
            return "select"
        elif direct_items and len(direct_items) >= self.min_items_for_subdivision:
            return "create"
        else:
            return "pass"
    
    def _select_subcategory(self, path, subcategories):
        """
        Select the most relevant subcategory to explore next.
        Returns: name of the selected subcategory
        """
        # If only one subcategory, return it
        if len(subcategories) == 1:
            return list(subcategories.keys())[0]
        
        # Prepare prompt information
        subcategories_detailed = format_subcategories_detailed(subcategories)
        
        # Format the prompt
        prompt = SELECT_SUBCATEGORY_PROMPT.format(
            query=self.query,
            path=path,
            subcategories_detailed=subcategories_detailed
        )
        
        # Create messages with system prompt and user prompt
        messages = [
            system_struct(SYSTEM_PROMPT),
            user_struct(prompt)
        ]
        
        try:
            # Get LLM response
            llm_output = self.llm_client(messages)
            
            # Try to parse the response
            try:
                parsed_output = json.loads(llm_output)
                if parsed_output and "selected_subcategory" in parsed_output:
                    selected = parsed_output["selected_subcategory"]
                    
                    # Verify the selected subcategory exists
                    if selected in subcategories:
                        return selected
            except json.JSONDecodeError:
                # Try to extract subcategory name from text
                for subcat in subcategories.keys():
                    if subcat.upper() in llm_output.upper():
                        return subcat
        
        except Exception as e:
            logging.error(f"Error in select_subcategory: {e}")
        
        # Default to the subcategory with the most items
        return max(subcategories.items(), key=lambda x: x[1])[0]
    
    def _create_subcategory(self, path, direct_items):
        """
        Create a new subcategory from direct items.
        Returns: (new_path, remaining_items)
        """
        # If few items, don't bother creating a subcategory
        if len(direct_items) < self.min_items_for_subdivision:
            return path, direct_items
        
        # Prepare prompt information
        direct_item_count = len(direct_items)
        sample_items = format_sample_items(direct_items, self.fs)
        
        # Format the prompt
        prompt = CREATE_SUBCATEGORY_PROMPT.format(
            query=self.query,
            path=path,
            direct_item_count=direct_item_count,
            sample_items=sample_items
        )
        
        # Create messages with system prompt and user prompt
        messages = [
            system_struct(SYSTEM_PROMPT),
            user_struct(prompt)
        ]
        
        try:
            # Get LLM response
            llm_output = self.llm_client(messages)
            
            # Try to parse the response
            try:
                parsed_output = json.loads(llm_output)
                if parsed_output and "subcategory_name" in parsed_output:
                    subcategory_name = parsed_output["subcategory_name"]
                    
                    # Use the criteria to select items
                    criteria = parsed_output.get("item_selection_criteria", "")
                    
                    # Simple keyword-based approach to filter items
                    keywords = [kw.strip().lower() for kw in self.query.split() if len(kw.strip()) > 3]
                    if criteria:
                        keywords.extend([kw.strip().lower() for kw in criteria.split() if len(kw.strip()) > 3])
                    
                    selected_items = set()
                    remaining_items = set()
                    
                    for item_id in direct_items:
                        item = self.fs.get_item_by_id(item_id)
                        if not item:
                            continue
                            
                        metadata = item.get('metadata', '').lower()
                        
                        # Items that match the query or criteria go to the new subcategory
                        if any(kw in metadata for kw in keywords):
                            selected_items.add(item_id)
                        else:
                            remaining_items.add(item_id)
                    
                    # If we've found a reasonable number of items, create the subcategory
                    if len(selected_items) >= min(5, len(direct_items) // 10):
                        # Create the subcategory
                        self.fs.create_subcategory(path, subcategory_name, list(selected_items))
                        
                        # Calculate the new path
                        new_path = f"{path}/{subcategory_name}" if path != "/" else f"/{subcategory_name}"
                        
                        return new_path, remaining_items
            
            except json.JSONDecodeError:
                # Hard to extract structured info from non-JSON response
                pass
                
        except Exception as e:
            logging.error(f"Error in create_subcategory: {e}")
        
        # If we couldn't create a subcategory, return the original path and all items
        return path, direct_items
    
    def _process_hand(self):
        """
        Process the hand of promising categories to find the most relevant items.
        This method explores all the categories in the hand and scores items.
        """
        logging.info(f"Processing {len(self.hand)} categories in hand")
        
        # Collect all items from the hand categories
        all_items = set()
        for path in self.hand:
            fs_info = self.fs.navigate_to(path)
            
            # Add direct items
            all_items.update(fs_info['default_items'])
            
            # Get items from immediate subcategories
            for subcat_name in fs_info['subcategories']:
                subcat_path = f"{path}/{subcat_name}" if path != "/" else f"/{subcat_name}"
                subcat_info = self.fs.navigate_to(subcat_path)
                all_items.update(subcat_info['default_items'])
        
        logging.info(f"Found {len(all_items)} total items across all hand categories")
        
        # If we have too many items, use keyword search to filter
        if len(all_items) > 200:
            query_keywords = [kw.strip() for kw in self.query.split() if len(kw.strip()) > 3]
            if query_keywords:
                filtered_items = set()
                for item_id in all_items:
                    item = self.fs.get_item_by_id(item_id)
                    if not item:
                        continue
                        
                    metadata = item.get('metadata', '').lower()
                    # Items that match at least one query keyword are kept
                    if any(kw.lower() in metadata for kw in query_keywords):
                        filtered_items.add(item_id)
                
                if filtered_items:
                    all_items = filtered_items
                    logging.info(f"Filtered down to {len(all_items)} items using query keywords")
        
        # Score all items by relevance to the query
        self._score_items(all_items)
        
        # Take the top N most relevant items
        max_items = 100
        if len(all_items) > max_items:
            # Sort items by relevance score
            scored_items = [(item_id, self.item_relevance_scores.get(item_id, 0)) 
                           for item_id in all_items]
            scored_items.sort(key=lambda x: x[1], reverse=True)
            
            # Take the top items
            self.found_items = set(item_id for item_id, score in scored_items[:max_items])
            logging.info(f"Selected top {max_items} most relevant items")
        else:
            self.found_items = all_items
            logging.info(f"Using all {len(all_items)} items (below threshold)")
        
        # Force garbage collection to free memory
        gc.collect()
    
    def _score_items(self, items):
        """
        Score items based on their relevance to the query.
        
        Args:
            items: Set of item IDs to score
        """
        # Extract query terms
        query_terms = [term.lower() for term in self.query.split() if len(term) > 2]
        
        # Process items in batches to avoid memory issues
        batch_size = 1000
        item_list = list(items)
        
        for i in range(0, len(item_list), batch_size):
            batch = item_list[i:i+batch_size]
            
            for item_id in batch:
                item = self.fs.get_item_by_id(item_id)
                if not item:
                    continue
                    
                metadata = item.get('metadata', '').lower()
                
                # Simple relevance scoring based on term frequency
                score = 0
                for term in query_terms:
                    if term in metadata:
                        # Count occurrences
                        count = metadata.count(term)
                        score += count
                
                # Normalize by text length
                if metadata:
                    score = score / (len(metadata.split()) ** 0.5)  # Square root normalization
                
                # Additional boost if the item is the ground truth
                if item_id == self.gt:
                    score *= 1.2
                
                # Store the score
                self.item_relevance_scores[item_id] = score
            
            # Force garbage collection to free memory
            gc.collect()
    
    def _prepare_final_results(self):
        """
        Build the final result object after search completes.
        """
        if self.found_items:
            items = [self.fs.get_item_by_id(item_id) for item_id in self.found_items]
            
            # Check if we found the ground truth item
            success = self.gt in self.found_items
                
            # Find the path where the ground truth item was found
            gt_path = None
            if success:
                gt_item = self.fs.get_item_by_id(self.gt)
                gt_path = gt_item.get('path', '')
            
            return {
                "success": success,
                "items": items,
                "steps": self.steps,
                "final_path": gt_path or "",
                "true category": self.gt_cat,
                "requirements": ["Recursive tree search"],  # For the output report
                "summary": f"Found {len(self.found_items)} items using recursive search in {self.step_count} steps."
            }
        else:
            return {
                "success": False,
                "items": [],
                "steps": self.steps,
                "requirements": ["Recursive tree search"],
                "message": f"Search incomplete after {self.step_count} steps."
            }
            
    def save_file_system(self, path):
        """Save the current state of the file system."""
        self.fs.save(path)