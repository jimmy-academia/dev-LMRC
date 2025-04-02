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
        
        # Global state that persists across recursive calls
        self.found_items = set()
        self.visited_paths = set()
        self.item_relevance_scores = {}
        self.steps = []
        self.step_count = 0
        
    def search(self, query, gt, gt_cat):
        """
        Search for items matching the query using a recursive tree search process.
        Returns a dict with results and search information.
        """
        # Initialize search state
        self.gt = gt
        self.gt_cat = gt_cat
        self.query = query

        self.hand = []
        self.hand_limit = 100

        logging.info(f"Starting recursive search for query: {query}")
        self._recursive_search('/', 0)

    def _recursive_search(self, path):

        # Get information about the current node
        fs_info = self.fs.navigate_to(path)
        total_item_count = fs_info['total_items']
        if total_item_count > 100:
            while total_item_count > 50 and len(self.hand) <= self.hand_limit:

                decision = None
                if ("""condition: not fully subcategorized"""):
                    # decision = make_decision_form_or_select

                if decision== 'select' or ("""condition: fully subcategorized """):
                    "conduct action select subcategory"
                else:
                    "conduct action form subcategory"

                self._recursive_search("subcategory path")

                "remove subcategory from consideration"

        else:
            for item in "current category":
                "conduct action: examine item and decide whether to put in hand or not."

        # Analyze this node with LLM
        node_analysis = self._analyze_node(path, fs_info, level)
        
        # Record this step
        self.step_count += 1
        self.steps.append({
            "path": path,
            "level": level,
            "analysis": node_analysis["reasoning"],
            "decision": node_analysis["decision"]
        })
        
        # Process the node based on the analysis decision
        action = node_analysis["decision"]["action"]
        params = node_analysis["decision"]["params"]
        
        if action == "backtrack":
            # This node is not relevant, so we backtrack
            logging.info(f"Backtracking from {path}")
            return []
        
        elif action == "search":
            # Search for items at this node
            keywords = params.get("keywords", [])
            items = self.fs.keyword_search(keywords, path)
            
            # Score and add relevant items
            relevant_items = self._score_and_filter_items(items, params.get("min_relevance", 0.5))
            
            # Add to global found items
            self.found_items.update(relevant_items)
            
            if params.get("explore_subcategories", False) and level < self.max_depth - 1:
                # Also explore subcategories
                self._explore_subcategories(path, fs_info, level)
            
            return list(relevant_items)
        
        elif action == "explore":
            # Explore specific subcategories
            return self._explore_subcategories(path, fs_info, level, params.get("subcategories", []))
        
        elif action == "create":
            # Create new subcategories and distribute items
            return self._create_subcategories(path, params.get("subcategories", []), level)
        
        else:
            logging.warning(f"Unknown action: {action}")
            return []
    
    def _analyze_node(self, path, fs_info, level):
        """
        Use the LLM to analyze the current node and decide what to do.
        
        Returns:
            dict: Analysis and decision for this node
        """
        # Prepare prompt with node information
        node_prompt = ANALYZE_NODE_PROMPT.format(
            path=path,
            total_items=fs_info['total_items'],
            subcategories=fs_info['subcategories'],
            direct_items=len(fs_info['default_items']),
            level=level,
            max_depth=self.max_depth,
            query=self.query
        )
        
        # Build conversation with context
        messages = [
            system_struct(RECURSIVE_SYSTEM_PROMPT),
            user_struct(QUERY_PROMPT.format(query=self.query)),
            user_struct(node_prompt)
        ]
        
        # Get LLM analysis
        llm_output = self.llm_client(messages)
        
        # Parse the LLM's JSON output
        return self._parse_llm_json_output(llm_output)
    
    def _explore_subcategories(self, path, fs_info, level, specific_subcategories=None):
        """
        Explore subcategories recursively, either all or specific ones.
        
        Args:
            path (str): Current path
            fs_info (dict): Information about the current node
            level (int): Current depth level
            specific_subcategories (list): If provided, only explore these subcategories
            
        Returns:
            list: Relevant items found across all explored subcategories
        """
        relevant_items = []
        
        # Get subcategories to explore
        subcategories = fs_info['subcategories']
        
        if specific_subcategories:
            # Filter to only explore specific subcategories
            subcategories = {k: v for k, v in subcategories.items() if k in specific_subcategories}
        
        # Recursively explore each subcategory
        for subcat_name in subcategories:
            subcat_path = f"{path}/{subcat_name}" if path != '/' else f"/{subcat_name}"
            
            # Skip if already visited
            if subcat_path in self.visited_paths:
                continue
                
            # Recursively explore this subcategory
            subcat_items = self._recursive_search(subcat_path, level + 1)
            relevant_items.extend(subcat_items)
            
            # Check if we've reached the step limit
            if self.step_count >= self.max_steps:
                break
        
        return relevant_items
    
    def _create_subcategories(self, path, subcategories, level):
        """
        Create new subcategories and distribute items.
        
        Args:
            path (str): Current path
            subcategories (list): List of subcategories to create, each with:
                - name: Name of the subcategory
                - keywords: Keywords to search for items to include
                - exclusive: Whether this subcategory should not overlap with others
                
        Returns:
            list: Relevant items found in the created subcategories
        """
        relevant_items = []
        
        # First, find all items at the current path
        all_items_at_path = self.fs.navigate_to(path)['default_items']
        
        # Search and categorize items
        for subcat_info in subcategories:
            subcat_name = subcat_info['name']
            keywords = subcat_info['keywords']
            
            # Search for items matching the subcategory keywords
            items = self.fs.keyword_search(keywords, path)
            
            # Only include items that are directly at this path
            items = [item for item in items if item in all_items_at_path]
            
            if items:
                # Create the subcategory
                new_path = f"{path}/{subcat_name}" if path != '/' else f"/{subcat_name}"
                self.fs.create_subcategory(path, subcat_name, items)
                
                # If specified, recursively explore the new subcategory
                if subcat_info.get("explore", True) and level < self.max_depth - 1:
                    # Mark as visited so we don't revisit
                    self.visited_paths.add(new_path)
                    
                    # Recursively search this new subcategory
                    subcat_items = self._recursive_search(new_path, level + 1)
                    relevant_items.extend(subcat_items)
                
                # Check if we've reached the step limit
                if self.step_count >= self.max_steps:
                    break
        
        return relevant_items
    
    def _score_and_filter_items(self, items, min_relevance=0.5):
        """
        Score items based on relevance to query and filter out low scores.
        This is a simplified scoring mechanism that could be enhanced with
        the LLM for better relevance assessment.
        
        Args:
            items (list): Item IDs to score
            min_relevance (float): Minimum relevance score to keep (0-1)
            
        Returns:
            set: Filtered set of relevant item IDs
        """
        relevant_items = set()
        
        # Get item details
        for item_id in items:
            item = self.fs.get_item_by_id(item_id)
            if not item:
                continue
                
            # Simple relevance scoring based on query terms in metadata
            # A real implementation would use more sophisticated methods
            query_terms = self.query.lower().split()
            metadata = item.get('metadata', '').lower()
            
            # Count matches and calculate a simple score
            matches = sum(1 for term in query_terms if term in metadata)
            score = matches / len(query_terms) if query_terms else 0
            
            # Store the score
            self.item_relevance_scores[item_id] = score
            
            # Keep items above the threshold
            if score >= min_relevance:
                relevant_items.add(item_id)
                
        return relevant_items
    
    def _parse_llm_json_output(self, llm_text):
        """
        Parse the LLM output as JSON, handling code fences if present.
        
        Returns:
            dict: Parsed JSON output or a default structure if parsing fails
        """
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
            # Return a default structure
            return {
                "reasoning": "Error parsing LLM output",
                "decision": {
                    "action": "search",
                    "params": {
                        "keywords": [self.query]
                    }
                }
            }
    
    def _prepare_final_results(self):
        """
        Build the final result object after search completes.
        """
        if self.found_items:
            items = [self.fs.get_item_by_id(item_id) for item_id in self.found_items]
            
            success = False
            if self.gt in self.found_items:
                success = True
                
            # Find the path where the ground truth item was found
            gt_path = None
            if success and self.gt in self.found_items:
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


# For compatibility with the main.py, expose the class with the same name
ReactAgent = RecursiveAgent