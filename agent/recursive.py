import logging
import json
import re
from collections import defaultdict
from utils import parse_llm_output

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
        
    def search(self, query, gt, gt_cat):
        """
        Search for items matching the query using a recursive tree search process.
        Returns a dict with results and search information.
        """
        self.gt = gt
        self.gt_cat = gt_cat
        self.query = query

        # Global state that persists across recursive calls
        self.steps = []
        self.step_count = 0
        
        self.hand = []
        self.hand_limit = 120

        logging.info(f"Starting recursive search for query: {query}")

        self._recursive_search('/', 0)

        input("## process self.hand down to 12 leaf categories (1200 items)##")
        input("## process final item down to 100 items!! ##")
        return self._prepare_final_results()

    def _recursive_search(self, path, depth):

        if depth >= self.max_depth:
            logging.info(f"Max depth reached at {path}, stopping recursion")
            return

        input('ready to take next step?')

        # Get information about the current node
        fs_info = self.fs.navigate_to(path)
        total_item_count = fs_info['total_items'] # int
        direct_items = fs_info['direct_items']  # set
        subcategories = fs_info['subcategories'] # dict

        if total_item_count > 100:
            while (direct_items or subcategories) and len(self.hand) <= self.hand_limit:

                decision = None
                if len(direct_items) > 0:
                    decision = self.decide_create_select_pass(subcategories, direct_items)
                elif subcategories: 
                    decision = 'select'
                else:
                    # it does not seem good to select or create from this level
                    break

                self.step_count += 1

                if decision=='select':
                    selected_subcat = self.select_subcategory(path, subcategories)
                    subcategories.pop(selected_subcat)
                    next_path = f"{path}/{selected_subcat}" if path != "/" else f"/{selected_subcat}"
                    logging.info(f"Step {self.step_count}: Select {next_path} from {subcategories}")

                elif decision == 'create':
                    next_path, direct_items = self.create_subcategory(path, direct_items)
                    logging.info(f"Step {self.step_count}: Create {next_path}")

                elif decision == 'pass':
                    logging.info(f"Step {self.step_count}: Decide to pass")

                    break
                    
                self._recursive_search(next_path, depth + 1)

        else:
            self.hand.append(path)
            if len(self.hand) >= self.hand_limit:
            logging.info(f"Reached hand limit of {self.hand_limit} leaf categories, stopping search")
                return
                
    def decide_create_select_pass(self, path, subcategories, direct_items):
        """
        Decide whether to select from existing subcategories, create a new subcategory, or pass.
        Returns: "SELECT", "CREATE", or "PASS"
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Get LLM response
        llm_output = self.llm_client(messages)
        parsed_output = parse_llm_output(llm_output)
        
        if parsed_output and "decision" in parsed_output:
            decision = parsed_output["decision"].upper()
            if decision in ["SELECT", "CREATE", "PASS"]:
                logging.info(f"Decision for path {path}: {decision}")
                step_info = self.steps[-1]  # Update the last step info
                step_info["decision"] = decision
                step_info["reasoning"] = parsed_output.get("reasoning", "No reasoning provided")
                return decision
                
        # Default to SELECT if we have subcategories, otherwise CREATE
        return "SELECT" if subcategories else "CREATE"
    
    def select_subcategory(self, path, subcategories):
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Get LLM response
        llm_output = self.llm_client(messages)
        parsed_output = parse_llm_output(llm_output)
        
        if parsed_output and "selected_subcategory" in parsed_output:
            selected = parsed_output["selected_subcategory"]
            
            # Verify the selected subcategory exists
            if selected in subcategories:
                logging.info(f"Selected subcategory: {selected}")
                step_info = self.steps[-1]  # Update the last step info
                step_info["selected_subcategory"] = selected
                step_info["selection_reasoning"] = parsed_output.get("reasoning", "No reasoning provided")
                return selected
            else:
                logging.warning(f"Selected subcategory {selected} not found in {subcategories}")
        else:
            logging.warning("Failed to parse LLM output for subcategory selection")
            
        
        # Default to the subcategory with the most items
        default_choice = max(subcategories.items(), key=lambda x: x[1])[0]
        logging.info(f"Defaulting to subcategory with most items: {default_choice}")
        return default_choice
    
    def create_subcategory(self, path, direct_items):
        """
        Create a new subcategory from direct items.
        Returns: (new_path, remaining_items)
        """
        # If few items, default Other
        if len(direct_items) < self.min_items_for_subdivision:
            logging.info(f"Too few items ({len(direct_items)}) to create subcategory")
            selected_items = direct_items
        else:
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            # Get LLM response
            llm_output = self.llm_client(messages)
            parsed_output = parse_llm_output(llm_output)
            
            if parsed_output and "subcategory_name" in parsed_output:
                subcategory_name = parsed_output["subcategory_name"]
                
                # Use the selection criteria to filter items
                criteria = parsed_output.get("item_selection_criteria", "")
                
                # For now, let's use a simple keyword-based approach to select items
                # This could be enhanced with a more sophisticated approach using another LLM call
                keywords = [kw.strip().lower() for kw in self.query.split() if len(kw.strip()) > 3]
                keywords.extend([kw.strip().lower() for kw in criteria.split() if len(kw.strip()) > 3])
                
                selected_items = set()
                for item_id in direct_items:
                    item = self.fs.get_item_by_id(item_id)
                    if item:
                        metadata = item.get('metadata', '').lower()
                        if any(kw in metadata for kw in keywords):
                            selected_items.add(item_id)
            
            # If we selected a reasonable number of items, create the subcategory
            # if len(selected_items) >= min(5, len(direct_items) // 10):
                
            # Create the subcategory in the file system
            self.fs.create_subcategory(path, subcategory_name, selected_items)
            
            # Calculate the new path
            new_path = f"{path}/{subcategory_name}" if path != "/" else f"/{subcategory_name}"
            
            # Calculate remaining items
            remaining_items = direct_items - selected_items
            
            logging.info(f"Created subcategory {new_path} with {len(selected_items)} items")
            step_info = self.steps[-1]  # Update the last step info
            step_info["created_subcategory"] = subcategory_name
            step_info["items_moved"] = len(selected_items)
            step_info["creation_reasoning"] = parsed_output.get("reasoning", "No reasoning provided")
            
            return new_path, remaining_items

        else:
            logging.warning("Failed to parse LLM output for subcategory creation")
            
        
        return new_path, direct_items

