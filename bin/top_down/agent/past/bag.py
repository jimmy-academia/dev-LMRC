import logging
import json
import re
import uuid
from collections import defaultdict

from utils import user_struct, system_struct, assistant_struct, flatten_messages
from prompts import (
    SYSTEM_PROMPT,
    QUERY_PROMPT,
)

# New system prompt for the bag-based approach
BAG_SYSTEM_PROMPT = """You are a helpful search assistant that helps users find products based on their query.
You have access to a file system of products organized by categories, and a special "bag" feature
that lets you temporarily collect, inspect, and refine items before finalizing subcategories.

You can:
1. Navigate to different category paths
2. Search for products using keywords  
3. Create and manage bags of items
4. Refine bag contents
5. Create new subcategories from your curated bags
6. Tag items with relevant attributes

The "bag" concept allows you to:
- Collect candidate items in a temporary workspace
- Inspect them to ensure they match criteria
- Add or remove items as needed
- Finalize by creating a subcategory or returning as results

Follow these steps to find the most relevant products:
1. Analyze the query to understand what the user is looking for
2. Navigate to the most relevant category
3. Create a bag to collect potential matches
4. Refine the bag by adding or removing items
5. Create subcategories from refined bags or use them for the final answer
"""

BAG_STEP_PROMPT = """
You are now working with a bag-based search system. Your goal is to achieve the user's query by performing a single reasoning step and choosing from an available set of actions.

Important: You must output valid JSON in the following format:
{
  "reasoning": "<your hidden reasoning here>",
  "action": {
    "name": "<one of the available actions>",
    "params": {
        // key-value pairs relevant to the action
    }
  }
}

Available actions:
- Navigate: Move to a subdirectory/category
  - name: "Navigate"
  - params: { "path": "/some/path" }
- Search: Search within the current path
  - name: "Search"
  - params: { "keywords": ["keyword1", "keyword2", ...] }
- CreateBag: Create a new collection bag
  - name: "CreateBag"
  - params: { "name": "bag_name", "description": "purpose of this bag" }
- AddToBag: Add items to a bag
  - name: "AddToBag"
  - params: { "bag_id": "bag_id", "item_ids": ["id1", "id2", ...] }
- RemoveFromBag: Remove items from a bag
  - name: "RemoveFromBag"
  - params: { "bag_id": "bag_id", "item_ids": ["id1", "id2", ...] }
- InspectBag: Get detailed information about bag contents
  - name: "InspectBag"
  - params: { "bag_id": "bag_id", "sample_size": 5 }
- RefineBag: Filter the bag contents based on criteria
  - name: "RefineBag"
  - params: { "bag_id": "bag_id", "criteria": "some criteria", "mode": "keep or remove" }
- CreateSubcategoryFromBag: Create a new subcategory using bag contents
  - name: "CreateSubcategoryFromBag"
  - params: { "bag_id": "bag_id", "parent_path": "/path", "name": "subcategory_name" }
- DeleteBag: Remove a bag when it's no longer needed
  - name: "DeleteBag"
  - params: { "bag_id": "bag_id" }
- Complete: Finish the search with items from one or more bags
  - name: "Complete"
  - params: { "bag_ids": ["bag_id1", "bag_id2", ...] }

Current state:
Path: {current_path}
Total items at this path: {total_items}
Subcategories: {subcategories}
Number of items not in subcategories: {default_items_count}
Bags: {bags}
Step {step_count} of {max_steps}

Remember your goal is to find products matching: "{query}"

Return only valid JSON. Do NOT include extra keys.
"""

class BagAgent:
    """
    A search agent that uses a "bag" or "backpack" approach to collect,
    inspect, and refine items before finalizing subcategories or results.
    """

    def __init__(self, file_system, llm_client):
        """Initialize the agent with a file system and LLM client."""
        self.fs = file_system
        self.llm_client = llm_client
        
        # Initialize bag storage
        self.bags = {}  # Maps bag_id to bag info
        
    def search(self, query, gt, gt_cat, max_steps=5):
        """
        Search for items matching the query using a bag-based approach.
        Returns a dict with results and search information.
        """
        # Initialize search state
        self.gt = gt
        self.gt_cat = gt_cat
        state = {
            "query": query,
            "current_path": "/",
            "steps": [],
            "found_items": set(),
            "step_count": 0,
            "complete": False
        }
        
        # Clear any existing bags
        self.bags = {}

        # Prepare the initial conversation
        base_messages = [
            system_struct(BAG_SYSTEM_PROMPT),
            user_struct(QUERY_PROMPT.format(query=query))
        ]

        # Main search loop
        msg_pt = 0
        while not state["complete"] and state["step_count"] < max_steps:
            # Get current file system information
            fs_info = self.fs.navigate_to(state["current_path"])
            
            # Build prompt with bag information
            step_prompt = BAG_STEP_PROMPT.format(
                current_path=state["current_path"],
                total_items=fs_info['total_items'],
                subcategories=fs_info['subcategories'],
                default_items_count=len(fs_info['default_items']),
                bags=self._format_bags_for_prompt(),
                step_count=state["step_count"] + 1,
                max_steps=max_steps,
                query=query
            )

            # Prepare messages
            step_messages = base_messages.copy()
            step_messages.append(user_struct(step_prompt))

            # Single LLM call for reasoning + action
            print(f"\n ======= STEP {state['step_count']+1} =======")
            print(f"--- STEP {state['step_count']+1} RAW INPUT ---\n{flatten_messages(step_messages[msg_pt:])}")

            llm_output = self.llm_client(step_messages)

            # For debugging/logging
            print(f"--- STEP {state['step_count']+1} RAW OUTPUT ---\n{llm_output}\n")

            # Parse the LLM's JSON output
            parsed_output = self._parse_llm_json_output(llm_output)
            if not parsed_output:
                # If parsing failed, log error & break
                error_message = "ERROR: Could not parse LLM JSON output."
                logging.error(error_message)
                state["steps"].append({"error": error_message, "raw_output": llm_output})
                break

            # Extract reasoning and action
            reasoning = parsed_output.get("reasoning", "")
            action_dict = parsed_output.get("action", {})
            action_name = action_dict.get("name", "")
            action_params = action_dict.get("params", {})

            # Store the step for debugging or review
            step_record = {
                "reasoning": reasoning,
                "action": action_dict,
            }
            state["steps"].append(step_record)

            # Execute the parsed action
            action_result = self._execute_action(action_name, action_params, state)
            step_record["result"] = action_result
            
            print(f"--- STEP {state['step_count']+1} STEP OPS ---\n{step_record}\n")

            # Add the LLM's output & result to the conversation context
            base_messages.append(assistant_struct(json.dumps(parsed_output)))
            base_messages.append(system_struct(action_result))
            msg_pt = len(base_messages)

            # Increment step
            state["step_count"] += 1

            input(f'pause before step {state["step_count"] + 1}')
        input(f'pause before finish search.')

        # Prepare final results
        return self._prepare_final_results(state)
        
    def _execute_action(self, action_name, params, state):
        """
        Execute the specified action based on action_name.
        """
        # Standard actions from original agent
        if action_name == "Navigate":
            path = params.get("path", "/")
            return self._navigate(path, state)

        elif action_name == "Search":
            keywords = params.get("keywords", [])
            return self._search(keywords, state)
        
        elif action_name == "Complete":
            bag_ids = params.get("bag_ids", [])
            return self._complete_from_bags(bag_ids, state)
            
        # Bag-specific actions
        elif action_name == "CreateBag":
            name = params.get("name", "Untitled Bag")
            description = params.get("description", "")
            return self._create_bag(name, description, state)
            
        elif action_name == "AddToBag":
            bag_id = params.get("bag_id", "")
            item_ids = params.get("item_ids", [])
            return self._add_to_bag(bag_id, item_ids, state)
            
        elif action_name == "RemoveFromBag":
            bag_id = params.get("bag_id", "")
            item_ids = params.get("item_ids", [])
            return self._remove_from_bag(bag_id, item_ids, state)
            
        elif action_name == "InspectBag":
            bag_id = params.get("bag_id", "")
            sample_size = params.get("sample_size", 5)
            return self._inspect_bag(bag_id, sample_size, state)
            
        elif action_name == "RefineBag":
            bag_id = params.get("bag_id", "")
            criteria = params.get("criteria", "")
            mode = params.get("mode", "keep")
            return self._refine_bag(bag_id, criteria, mode, state)
            
        elif action_name == "CreateSubcategoryFromBag":
            bag_id = params.get("bag_id", "")
            parent_path = params.get("parent_path", state["current_path"])
            name = params.get("name", "New Subcategory")
            return self._create_subcategory_from_bag(bag_id, parent_path, name, state)
            
        elif action_name == "DeleteBag":
            bag_id = params.get("bag_id", "")
            return self._delete_bag(bag_id, state)
            
        else:
            return f"Unrecognized action '{action_name}'. Must be one of the available actions."
    
    def _navigate(self, path, state):
        """Navigate to a specific category path."""
        if not path.startswith('/'):
            path = f"/{path}"
        fs_info = self.fs.navigate_to(path)

        if fs_info['total_items'] > 0:
            state["current_path"] = path
            result = f"Successfully navigated to {path}."
        else:
            result = f"Path {path} not found or contains no items."

        return result

    def _search(self, keywords, state):
        """Search for items containing any of the keywords."""
        items = self.fs.keyword_search(keywords, state["current_path"])

        result = f"Found {len(items)} items matching keywords: {keywords}."
        if items:
            sample_items = items[:3]
            result += f" Example item IDs: {sample_items}"

        return result
    
    def _create_bag(self, name, description, state):
        """Create a new bag for collecting items."""
        # Generate a unique ID for the bag
        bag_id = f"bag_{uuid.uuid4().hex[:8]}"
        
        # Create the bag
        self.bags[bag_id] = {
            "id": bag_id,
            "name": name,
            "description": description,
            "items": set(),
            "created_at_path": state["current_path"]
        }
        
        return f"Created new bag '{name}' with ID: {bag_id}"
    
    def _add_to_bag(self, bag_id, item_ids, state):
        """Add items to a bag."""
        if bag_id not in self.bags:
            return f"Error: Bag with ID {bag_id} not found."
        
        # Add items to the bag
        for item_id in item_ids:
            self.bags[bag_id]["items"].add(item_id)
        
        return f"Added {len(item_ids)} items to bag '{self.bags[bag_id]['name']}'. Total items: {len(self.bags[bag_id]['items'])}"
    
    def _remove_from_bag(self, bag_id, item_ids, state):
        """Remove items from a bag."""
        if bag_id not in self.bags:
            return f"Error: Bag with ID {bag_id} not found."
        
        # Remove items from the bag
        removed_count = 0
        for item_id in item_ids:
            if item_id in self.bags[bag_id]["items"]:
                self.bags[bag_id]["items"].remove(item_id)
                removed_count += 1
        
        return f"Removed {removed_count} items from bag '{self.bags[bag_id]['name']}'. Remaining items: {len(self.bags[bag_id]['items'])}"
    
    def _inspect_bag(self, bag_id, sample_size, state):
        """Get detailed information about bag contents."""
        if bag_id not in self.bags:
            return f"Error: Bag with ID {bag_id} not found."
        
        bag = self.bags[bag_id]
        items = list(bag["items"])
        
        # Get a sample of items to inspect
        sample = items[:min(sample_size, len(items))]
        sample_details = []
        
        for item_id in sample:
            item = self.fs.get_item_by_id(item_id)
            if item:
                # Extract a brief summary of the item
                metadata = item.get("metadata", "")
                summary = metadata[:100] + "..." if len(metadata) > 100 else metadata
                sample_details.append(f"ID: {item_id}, {summary}")
        
        result = (
            f"Bag: {bag['name']}\n"
            f"Description: {bag['description']}\n"
            f"Total items: {len(bag['items'])}\n"
            f"Created at path: {bag['created_at_path']}\n\n"
            f"Sample items ({len(sample_details)}):\n"
        )
        
        for i, detail in enumerate(sample_details, 1):
            result += f"{i}. {detail}\n"
        
        return result
    
    def _refine_bag(self, bag_id, criteria, mode, state):
        """Filter bag contents based on criteria."""
        if bag_id not in self.bags:
            return f"Error: Bag with ID {bag_id} not found."
        
        bag = self.bags[bag_id]
        items = list(bag["items"])
        
        # This is a simplified implementation of refinement
        # A more sophisticated version would use the LLM to evaluate each item
        # against the criteria
        
        refined_items = set()
        removed_items = set()
        
        # Process each item
        for item_id in items:
            item = self.fs.get_item_by_id(item_id)
            if not item:
                continue
                
            # Check if item matches criteria
            # This is a simple keyword match - could be enhanced
            metadata = item.get("metadata", "").lower()
            matches_criteria = all(word.lower() in metadata for word in criteria.lower().split())
            
            # Keep or remove based on mode
            if (mode == "keep" and matches_criteria) or (mode == "remove" and not matches_criteria):
                refined_items.add(item_id)
            else:
                removed_items.add(item_id)
        
        # Update bag with refined items
        bag["items"] = refined_items
        
        return (
            f"Refined bag '{bag['name']}' based on criteria: '{criteria}' (mode: {mode}).\n"
            f"Kept {len(refined_items)} items, removed {len(removed_items)} items."
        )
    
    def _create_subcategory_from_bag(self, bag_id, parent_path, name, state):
        """Create a new subcategory using bag contents."""
        if bag_id not in self.bags:
            return f"Error: Bag with ID {bag_id} not found."
        
        bag = self.bags[bag_id]
        items = list(bag["items"])
        
        if not items:
            return f"Error: Bag '{bag['name']}' is empty. Cannot create subcategory."
        
        # Create the subcategory
        moved_count = self.fs.create_subcategory(parent_path, name, items)
        
        if moved_count > 0:
            # Update current path to the new subcategory
            new_path = f"{parent_path}/{name}" if parent_path != "/" else f"/{name}"
            state["current_path"] = new_path
            
            return f"Created subcategory '{name}' with {moved_count} items from bag '{bag['name']}'."
        else:
            return f"Failed to create subcategory '{name}'. No items were moved."
    
    def _delete_bag(self, bag_id, state):
        """Remove a bag when it's no longer needed."""
        if bag_id not in self.bags:
            return f"Error: Bag with ID {bag_id} not found."
        
        bag_name = self.bags[bag_id]["name"]
        item_count = len(self.bags[bag_id]["items"])
        
        # Delete the bag
        del self.bags[bag_id]
        
        return f"Deleted bag '{bag_name}' containing {item_count} items."
    
    def _complete_from_bags(self, bag_ids, state):
        """Finish the search with items from one or more bags."""
        if not bag_ids:
            return "Error: No bags specified for completion."
        
        # Collect items from all specified bags
        all_items = set()
        found_bags = []
        
        for bag_id in bag_ids:
            if bag_id in self.bags:
                all_items.update(self.bags[bag_id]["items"])
                found_bags.append(self.bags[bag_id]["name"])
        
        if not all_items:
            return "Error: No items found in the specified bags."
        
        # Set the found items in the state
        state["found_items"] = all_items
        state["complete"] = True
        
        return f"Search completed with {len(all_items)} items from bags: {', '.join(found_bags)}."
    
    def _parse_llm_json_output(self, llm_text):
        """Parse the LLM output as JSON with code fence handling."""
        try:
            text = llm_text.strip()
            # Remove Markdown code fences if present
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:])
                if text.endswith("```"):
                    text = "\n".join(text.split("\n")[:-1])
            
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return None
    
    def _format_bags_for_prompt(self):
        """Format bags information for inclusion in the prompt."""
        if not self.bags:
            return "No bags created yet."
        
        bag_info = []
        for bag_id, bag in self.bags.items():
            bag_info.append({
                "id": bag_id,
                "name": bag["name"],
                "items_count": len(bag["items"]),
                "created_at": bag["created_at_path"]
            })
            
        return json.dumps(bag_info, indent=2)
    
    def _prepare_final_results(self, state):
        """Build the final result object after search completes."""
        if state["found_items"]:
            items = [self.fs.get_item_by_id(item_id) for item_id in state["found_items"]]
            
            success = False
            if self.gt in state["found_items"]:
                success = True
                
            return {
                "success": success,
                "items": items,
                "steps": state["steps"],
                "final_path": state["current_path"],
                "true category": self.gt_cat,
                "requirements": ["Bag-based organization"],  # For the output report
                "summary": f"Found {len(state['found_items'])} items using bag-based approach in {state['step_count']} steps."
            }
        else:
            return {
                "success": False,
                "items": [],
                "steps": state["steps"],
                "requirements": ["Bag-based organization"],
                "message": f"Search incomplete after {state['step_count']} steps."
            }
    
    def save_file_system(self, path):
        """Save the current state of the file system."""
        self.fs.save(path)


# For compatibility with the main.py, expose the class with the same name
ReactAgent = BagAgent