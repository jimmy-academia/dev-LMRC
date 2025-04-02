import logging
import json
import re
from collections import defaultdict

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


        ## process self.hand down ##
        ## process final item collect!! ##
        return self._prepare_final_results()

    def _recursive_search(self, path, depth):

        if depth >= self.max_depth:
            logging.info(f"Max depth reached at {path}, stopping recursion")
            return

        self.step_count += 1
        logging.info(f"Step {self.step_count}: Exploring path {path} at depth {depth}")

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

                if decision== 'select' or len(direct_items) == 0:
                    selected_subcat = self.select_subcategory(path, subcategories)
                    subcategories.pop(selected_subcat)
                    next_path = f"{path}/{selected_subcat}" if path != "/" else f"/{new_subcat}"
                elif decision == 'create':
                    next_path, direct_items = self.create_subcategory(path, direct_items)
                else:
                    # it does not seem good to select or create from this level
                    break

                self._recursive_search(next_path, depth + 1)

        else:
            self.hand.append(path)
            if len(self.hand) >= self.hand_limit:
            logging.info(f"Reached hand limit of {self.hand_limit} leaf categories, stopping search")
                return
                

    def decide_create_select_pass(self, subcategories, direct_items):
        # first select
        # then consider create
        # finally pass

        pass 

    def select_subcategory(self, path, subcategories):
        pass 

    def create_subcategory(slef, path, direct_items):
        pass 

