import re
import time
import torch
import pickle
import json
import math
import random
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Any, Optional, Union

# Imports from HF datasets or your local caching logic
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import openai
from utils import readf

########################################
# 1. Data Loading and Base Structures
########################################

def prepare_item_query(data_pkl):
    """Load or prepare the item pool and queries."""
    if not data_pkl.exists():
        queries = load_dataset('McAuley-Lab/Amazon-C4')['test']
        filepath = hf_hub_download(
            repo_id='McAuley-Lab/Amazon-C4',
            filename='sampled_item_metadata_1M.jsonl',
            repo_type='dataset'
        )
        item_pool = []
        with open(filepath, 'r') as f:
            for line in f:
                item_pool.append(json.loads(line.strip()))
        with open(data_pkl, 'wb') as f:
            pickle.dump((item_pool, queries), f)
    else:
        with open(data_pkl, 'rb') as f:
            item_pool, queries = pickle.load(f)
    return item_pool, queries

@dataclass
class ItemMetadata:
    """Rich metadata for items with path-like organization and tagging."""
    item_id: str
    category: str
    raw_metadata: str
    subcategories: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    @property
    def path(self) -> str:
        """Return the full path including category and subcategories."""
        if not self.subcategories:
            return f"/{self.category}"
        return f"/{self.category}/{'/'.join(self.subcategories)}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "category": self.category,
            "raw_metadata": self.raw_metadata,
            "subcategories": self.subcategories,
            "tags": list(self.tags)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ItemMetadata':
        """Create from dictionary."""
        result = cls(
            item_id=data["item_id"],
            category=data["category"],
            raw_metadata=data["raw_metadata"]
        )
        result.subcategories = data.get("subcategories", [])
        result.tags = set(data.get("tags", []))
        return result

########################################
# 2. Virtual File System for Categories
########################################

class CategoryFileSystem:
    """Virtual file system for hierarchical organization of items."""
    def __init__(self):
        # Maps path -> list of item_ids
        self.directories: Dict[str, Set[str]] = defaultdict(set)
        # Maps item_id -> ItemMetadata
        self.items: Dict[str, ItemMetadata] = {}
        # Maps tag -> set of item_ids
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
    def add_item(self, item: ItemMetadata) -> None:
        """Add an item to the file system."""
        self.items[item.item_id] = item
        self.directories[item.path].add(item.item_id)
        for tag in item.tags:
            self.tag_index[tag].add(item.item_id)
    
    def organize_item(self, item_id: str, new_path: str) -> None:
        """Move an item to a new path."""
        if item_id not in self.items:
            return
        
        item = self.items[item_id]
        old_path = item.path
        
        # Remove from old path
        if old_path in self.directories:
            self.directories[old_path].discard(item_id)
            
        # Parse and set new subcategories
        parts = new_path.strip('/').split('/')
        if len(parts) > 0:
            item.category = parts[0]
            item.subcategories = parts[1:] if len(parts) > 1 else []
        
        # Add to new path
        self.directories[item.path].add(item_id)
    
    def add_tag(self, item_id: str, tag: str) -> None:
        """Add a tag to an item."""
        if item_id not in self.items:
            return
            
        item = self.items[item_id]
        if tag not in item.tags:
            item.tags.add(tag)
            self.tag_index[tag].add(item_id)
    
    def remove_tag(self, item_id: str, tag: str) -> None:
        """Remove a tag from an item."""
        if item_id not in self.items:
            return
            
        item = self.items[item_id]
        if tag in item.tags:
            item.tags.remove(tag)
            self.tag_index[tag].discard(item_id)
    
    def get_items_by_path(self, path: str) -> List[str]:
        """Get all items in a directory and its subdirectories."""
        result = set()
        for directory, items in self.directories.items():
            if directory == path or directory.startswith(f"{path}/"):
                result.update(items)
        return list(result)
    
    def get_items_by_tag(self, tag: str) -> List[str]:
        """Get all items with a specific tag."""
        return list(self.tag_index.get(tag, set()))
    
    def get_items_by_tags(self, tags: List[str], require_all: bool = True) -> List[str]:
        """Get items that have the given tags."""
        if not tags:
            return []
            
        if require_all:
            # Items must have ALL tags
            result = set(self.tag_index.get(tags[0], set()))
            for tag in tags[1:]:
                result.intersection_update(self.tag_index.get(tag, set()))
        else:
            # Items must have ANY of the tags
            result = set()
            for tag in tags:
                result.update(self.tag_index.get(tag, set()))
                
        return list(result)
    
    def get_subdirectories(self, path: str) -> List[str]:
        """Get all immediate subdirectories of a path."""
        subdirs = set()
        prefix = path.rstrip('/') + '/'
        for directory in self.directories.keys():
            if directory.startswith(prefix):
                # Get the next path component
                remainder = directory[len(prefix):]
                if '/' in remainder:
                    subdir = remainder.split('/')[0]
                    subdirs.add(prefix + subdir)
                else:
                    subdirs.add(directory)
        return list(subdirs)
    
    def get_common_tags(self, item_ids: List[str], min_count: int = 2) -> List[Tuple[str, int]]:
        """Find common tags among a set of items."""
        tag_counts = Counter()
        for item_id in item_ids:
            if item_id in self.items:
                tag_counts.update(self.items[item_id].tags)
        
        return [(tag, count) for tag, count in tag_counts.items() if count >= min_count]
    
    def suggest_subcategories(self, path: str, llm_client: Any) -> List[str]:
        """Use LLM to suggest potential subcategories for items in a path."""
        item_ids = self.get_items_by_path(path)
        if not item_ids:
            return []
            
        # Sample items to analyze
        sample_size = min(5, len(item_ids))
        sample_items = [self.items[item_id] for item_id in random.sample(item_ids, sample_size)]
        
        # Build prompt for the LLM
        descriptions = "\n\n".join([
            f"Item {i+1}: {item.raw_metadata[:300]}..." 
            for i, item in enumerate(sample_items)
        ])
        
        prompt = f"""
        I have a collection of items in the category path '{path}'. 
        Here are {sample_size} representative items:
        
        {descriptions}
        
        Based on these items, suggest 3-5 meaningful subcategories that would help organize
        these types of products better. Format your response as a comma-separated list
        of subcategory names only.
        """
        
        # Call LLM to get subcategories
        response = llm_client(prompt)
        
        # Parse the response
        subcategories = [s.strip() for s in response.split(',')]
        return subcategories

########################################
# 3. Search Indexing System
########################################

class EnhancedIndex:
    """Enhanced search index with category, text, and tag-based search."""
    def __init__(self, item_pool):
        # Text-based inverted index
        self.text_index = defaultdict(set)
        # Category-based index
        self.category_index = defaultdict(set)
        # Category-to-subcategory suggestions (LLM generated)
        self.subcategory_suggestions = {}
        
        # Build the indexes
        self._build_indexes(item_pool)
    
    def _build_indexes(self, item_pool):
        """Build the search indexes from the item pool."""
        for idx, item in enumerate(tqdm(item_pool, desc="Building search indexes", ncols=88)):
            # Index by category
            category = item.get('category', '')
            if category:
                self.category_index[category].add(idx)
            
            # Build text index
            metadata = item.get('metadata', '')
            tokens = re.findall(r'\w+', metadata.lower())
            for token in tokens:
                self.text_index[token].add(idx)
    
    def search_by_text(self, query: str, category_filter: str = None) -> Set[int]:
        """Search items by text, optionally filtered by category."""
        tokens = re.findall(r'\w+', query.lower())
        if not tokens:
            return set()
        
        # Start with all matching the first token
        result = self.text_index.get(tokens[0], set())
        
        # Intersect with other tokens
        for token in tokens[1:]:
            result &= self.text_index.get(token, set())
        
        # Filter by category if requested
        if category_filter and category_filter in self.category_index:
            result &= self.category_index[category_filter]
            
        return result
    
    def search_by_category(self, category: str) -> Set[int]:
        """Get all items in a specific category."""
        return self.category_index.get(category, set())
    
    def get_all_categories(self) -> List[str]:
        """Get a list of all categories."""
        return list(self.category_index.keys())

########################################
# 4. LLM Client for Intelligence
########################################

def create_llm_client():
    """Create a function to call the LLM."""
    openai.api_key = readf(".openaikey").strip()
    
    def call_llm(prompt, model="gpt-4", temperature=0.2, max_tokens=500):
        """Call the LLM with a prompt and return the response."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant specialized in product search and organization."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return "Error processing request."
    
    return call_llm

########################################
# 5. REACT Agent Implementation
########################################

class ReactAgent:
    """Agent that uses Reasoning, Action, Observation loop for search."""
    def __init__(self, item_pool, fs, llm_client):
        # Data sources
        self.item_pool = item_pool
        self.fs = fs
        self.llm_client = llm_client
        
        # Search state tracking
        self.enhanced_index = EnhancedIndex(item_pool)
        self.search_results = []
        self.current_query = ""
        self.current_reasoning = ""
        
        # Search context state
        self.context = {
            "requirements": [],
            "current_path": "/",
            "explored_paths": set(),
            "relevant_tags": set(),
            "previously_viewed_items": set(),
            "category_insights": {}  # Maps category -> observations
        }
    
    def initialize_file_system(self):
        """Process all items into the file system."""
        print("Initializing category file system...")
        for idx, item in enumerate(tqdm(self.item_pool, desc="Processing items", ncols=88)):
            meta = ItemMetadata(
                item_id=item.get('item_id', ''),
                category=item.get('category', ''),
                raw_metadata=item.get('metadata', '')
            )
            self.fs.add_item(meta)
    
    def search(self, query, max_steps=10):
        """Run the REACT loop for a search query."""
        self.current_query = query
        
        # Reset search state
        self.search_results = []
        self.context = {
            "requirements": [],
            "current_path": "/",
            "explored_paths": set(),
            "relevant_tags": set(),
            "previously_viewed_items": set(),
            "category_insights": {}
        }
        
        # Extract search requirements using LLM
        self._analyze_requirements(query)
        
        # Begin REACT loop
        for step in range(max_steps):
            print(f"\n=== Step {step+1}/{max_steps} ===")
            
            # 1. Reasoning - Decide what to do next
            reasoning = self._reasoning_step()
            self.current_reasoning = reasoning
            print(f"Reasoning:\n{reasoning[:300]}..." if len(reasoning) > 300 else reasoning)
            
            # 2. Action - Execute the chosen action
            action = self._determine_next_action(reasoning)
            print(f"Action: {action['type']}")
            
            # 3. Observation - Record the results
            observation = self._execute_action(action)
            print(f"Observation Summary: {observation[:100]}..." if len(observation) > 100 else observation)
            
            # Check if search is complete
            if "SEARCH_COMPLETE" in action['type']:
                break
        
        # Return final, organized results
        return self._finalize_results()
    
    def _analyze_requirements(self, query):
        """Extract key requirements from the search query using LLM."""
        prompt = f"""
        Analyze this product search query: "{query}"
        
        Extract specific requirements the user is looking for, including:
        1. Product features or characteristics
        2. Use cases or purposes
        3. Target audience
        4. Any constraints (price, size, etc.)
        
        List each requirement as a separate bullet point.
        """
        
        response = self.llm_client(prompt)
        
        # Extract requirements from bullet points
        requirements = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                requirements.append(line[2:])
            elif line.startswith('•'):
                requirements.append(line[1:].strip())
        
        self.context["requirements"] = requirements
        
        # Also determine likely categories
        category_prompt = f"""
        For this search query: "{query}"
        
        Which of these Amazon product categories would be most relevant? Select up to 3.
        Categories: {', '.join(self.enhanced_index.get_all_categories())}
        
        Format your response as a comma-separated list only.
        """
        
        category_response = self.llm_client(category_prompt)
        suggested_categories = [c.strip() for c in category_response.split(',')]
        
        # Store the top categories for exploration
        self.context["top_categories"] = suggested_categories[:3]
    
    def _reasoning_step(self):
        """Reason about the next best action based on current state."""
        # Format the current state
        current_path = self.context["current_path"]
        
        # Get items at current path
        items_in_path = self.fs.get_items_by_path(current_path)
        
        # Get a sample of items if we have some
        sample_items_text = ""
        if items_in_path:
            sample_size = min(3, len(items_in_path))
            sample_ids = random.sample(items_in_path, sample_size)
            
            samples = []
            for i, item_id in enumerate(sample_ids):
                item = self.fs.items[item_id]
                tags = ", ".join(item.tags) if item.tags else "No tags"
                samples.append(f"Sample {i+1}:\nPath: {item.path}\nTags: {tags}\nDesc: {item.raw_metadata[:150]}...")
            
            sample_items_text = "\n\n".join(samples)
        
        # Format requirements
        requirements_text = "\n".join([f"- {req}" for req in self.context["requirements"]])
        
        # Format recent search results if any
        recent_results = ""
        if self.search_results:
            recent_size = min(3, len(self.search_results))
            recent_samples = []
            for i, idx in enumerate(self.search_results[:recent_size]):
                item_id = self.item_pool[idx].get('item_id')
                if item_id in self.fs.items:
                    item = self.fs.items[item_id]
                    tags = ", ".join(item.tags) if item.tags else "No tags"
                    recent_samples.append(f"Result {i+1}:\nPath: {item.path}\nTags: {tags}\nDesc: {item.raw_metadata[:150]}...")
            
            recent_results = "\n\n".join(recent_samples)
        
        # Get subdirectories
        subdirs = self.fs.get_subdirectories(current_path)
        
        # Build the reasoning prompt
        prompt = f"""
        ## Search Context
        Query: "{self.current_query}"
        
        Requirements:
        {requirements_text}
        
        Current Path: {current_path}
        Items in current path: {len(items_in_path)}
        Subdirectories: {', '.join(subdirs) if subdirs else 'None'}
        
        Relevant Tags: {', '.join(self.context['relevant_tags']) if self.context['relevant_tags'] else 'None'}
        
        {f"Recent Search Results:\n{recent_results}" if recent_results else "No recent search results."}
        
        {f"Current Items Sample:\n{sample_items_text}" if sample_items_text else "No items in current path."}
        
        ## Reasoning Task
        As a search agent, reason about the best next action to find products matching the query:
        
        1. Consider the search requirements and the current state
        2. Assess if you need to navigate to a specific category
        3. Determine if you should search by text, create subcategories, or assign tags
        4. Evaluate if current items match the requirements
        
        Think step by step to determine the most effective next action.
        """
        
        return self.llm_client(prompt, max_tokens=800)
    
    def _determine_next_action(self, reasoning):
        """Determine the next action based on the reasoning."""
        # Potential actions the agent can take
        actions = [
            "NAVIGATE_TO_CATEGORY",
            "NAVIGATE_TO_PATH",
            "TEXT_SEARCH",
            "CREATE_SUBCATEGORIES",
            "ASSIGN_TAGS",
            "FILTER_BY_TAGS",
            "EVALUATE_ITEMS",
            "SEARCH_COMPLETE"
        ]
        
        # Have the LLM decide which action to take
        action_prompt = f"""
        Based on your reasoning:
        
        {reasoning}
        
        Choose the next action from these options:
        {', '.join(actions)}
        
        Format your response as:
        ACTION: <action_name>
        PARAMS: <parameters for the action>
        
        Example:
        ACTION: NAVIGATE_TO_CATEGORY
        PARAMS: Electronics
        
        Or:
        ACTION: TEXT_SEARCH
        PARAMS: waterproof bluetooth speaker
        """
        
        action_response = self.llm_client(action_prompt, max_tokens=200)
        
        # Parse the action response
        action_type = None
        action_params = None
        
        for line in action_response.strip().split('\n'):
            line = line.strip()
            if line.startswith('ACTION:'):
                action_type = line[7:].strip()
            elif line.startswith('PARAMS:'):
                action_params = line[7:].strip()
        
        # Default to EVALUATE_ITEMS if parsing failed
        if not action_type:
            for action in actions:
                if action.lower() in reasoning.lower():
                    action_type = action
                    break
            
            if not action_type:
                action_type = "EVALUATE_ITEMS"
        
        return {
            "type": action_type,
            "params": action_params
        }
    
    def _execute_action(self, action):
        """Execute the determined action and return observations."""
        action_type = action["type"]
        params = action["params"]
        
        if action_type == "NAVIGATE_TO_CATEGORY":
            return self._action_navigate_to_category(params)
        elif action_type == "NAVIGATE_TO_PATH":
            return self._action_navigate_to_path(params)
        elif action_type == "TEXT_SEARCH":
            return self._action_text_search(params)
        elif action_type == "CREATE_SUBCATEGORIES":
            return self._action_create_subcategories()
        elif action_type == "ASSIGN_TAGS":
            return self._action_assign_tags()
        elif action_type == "FILTER_BY_TAGS":
            return self._action_filter_by_tags(params)
        elif action_type == "EVALUATE_ITEMS":
            return self._action_evaluate_items()
        elif action_type == "SEARCH_COMPLETE":
            return "Search complete. Finalizing results."
        else:
            return f"Unknown action type: {action_type}"
    
    def _action_navigate_to_category(self, category_name):
        """Navigate to a root category."""
        if not category_name:
            # Try to pick from top categories if available
            if hasattr(self.context, "top_categories") and self.context["top_categories"]:
                category_name = self.context["top_categories"][0]
            else:
                # Default to first category in the index
                categories = self.enhanced_index.get_all_categories()
                if categories:
                    category_name = categories[0]
                else:
                    return "No category specified and no categories available."
        
        # Update current path
        self.context["current_path"] = f"/{category_name}"
        self.context["explored_paths"].add(self.context["current_path"])
        
        # Get items in this category
        items = self.fs.get_items_by_path(self.context["current_path"])
        subdirs = self.fs.get_subdirectories(self.context["current_path"])
        
        # Sample a few items to show
        sample_size = min(3, len(items))
        if sample_size > 0:
            sample_ids = random.sample(items, sample_size)
            samples = []
            for i, item_id in enumerate(sample_ids):
                item = self.fs.items[item_id]
                samples.append(f"Sample {i+1}: {item.raw_metadata[:150]}...")
            samples_text = "\n".join(samples)
        else:
            samples_text = "No items in this category."
        
        return f"Navigated to category /{category_name}\nItems: {len(items)}\nSubdirectories: {len(subdirs)}\n\n{samples_text}"
    
    def _action_navigate_to_path(self, path):
        """Navigate to a specific path in the file system."""
        # Handle relative paths
        if not path.startswith('/'):
            current = self.context["current_path"]
            if path == "..":
                # Go up one level
                parts = current.strip('/').split('/')
                if len(parts) > 1:
                    path = '/' + '/'.join(parts[:-1])
                else:
                    path = '/'
            else:
                # Append to current path
                path = current.rstrip('/') + '/' + path
        
        # Update current path
        self.context["current_path"] = path
        self.context["explored_paths"].add(path)
        
        # Get items and subdirectories
        items = self.fs.get_items_by_path(path)
        subdirs = self.fs.get_subdirectories(path)
        
        # Sample items
        sample_size = min(3, len(items))
        if sample_size > 0:
            sample_ids = random.sample(items, sample_size)
            samples = []
            for i, item_id in enumerate(sample_ids):
                item = self.fs.items[item_id]
                samples.append(f"Sample {i+1}: {item.raw_metadata[:150]}...")
            samples_text = "\n".join(samples)
        else:
            samples_text = "No items in this path."
        
        return f"Navigated to {path}\nItems: {len(items)}\nSubdirectories: {len(subdirs)}\n\n{samples_text}"
    
    def _action_text_search(self, query):
        """Perform a text search, optionally scoped to current path."""
        if not query:
            return "No search query specified."
        
        # Determine if we should scope to current category
        current_path = self.context["current_path"]
        category_filter = None
        if current_path != '/':
            category = current_path.strip('/').split('/')[0]
            category_filter = category
        
        # Perform the search
        search_results = self.enhanced_index.search_by_text(query, category_filter)
        self.search_results = list(search_results)
        
        # If no results, try without category filter
        if not search_results and category_filter:
            search_results = self.enhanced_index.search_by_text(query)
            self.search_results = list(search_results)
        
        # Sample results to show
        sample_size = min(5, len(search_results))
        samples_text = ""
        
        if sample_size > 0:
            samples = []
            sample_indices = random.sample(self.search_results, sample_size)
            
            for i, idx in enumerate(sample_indices):
                item = self.item_pool[idx]
                item_id = item.get('item_id')
                
                # Add to previously viewed
                self.context["previously_viewed_items"].add(item_id)
                
                # Get path and tags if available
                path = "/"
                tags = []
                if item_id in self.fs.items:
                    fs_item = self.fs.items[item_id]
                    path = fs_item.path
                    tags = list(fs_item.tags)
                
                samples.append(
                    f"Result {i+1}:\nPath: {path}\n" + 
                    (f"Tags: {', '.join(tags)}\n" if tags else "") + 
                    f"Description: {item.get('metadata', '')[:150]}..."
                )
            
            samples_text = "\n\n".join(samples)
        else:
            samples_text = "No results found."
        
        return f"Text search for '{query}' found {len(search_results)} results.\n\n{samples_text}"
    
    def _action_create_subcategories(self):
        """Create subcategories for the current path."""
        current_path = self.context["current_path"]
        
        # Get items in current path
        items = self.fs.get_items_by_path(current_path)
        if not items:
            return "No items in current path to categorize."
        
        # Use LLM to suggest subcategories
        subcategories = self.fs.suggest_subcategories(current_path, self.llm_client)
        
        if not subcategories:
            return "Failed to generate subcategories."
        
        # Sample items to categorize
        sample_size = min(50, len(items))
        sample_ids = random.sample(items, sample_size)
        
        # Process in batches
        batch_size = 5
        batches = [sample_ids[i:i+batch_size] for i in range(0, len(sample_ids), batch_size)]
        
        items_organized = 0
        
        for batch in batches:
            # Format items for the LLM
            batch_text = "\n".join([
                f"Item {i+1}: {self.fs.items[item_id].raw_metadata[:200]}..." 
                for i, item_id in enumerate(batch)
            ])
            
            # Create prompt for categorization
            prompt = f"""
            Classify each item into one of these subcategories:
            {', '.join(subcategories)}
            
            {batch_text}
            
            Format your response as:
            ITEM 1: subcategory_name
            ITEM 2: subcategory_name
            ...
            """
            
            response = self.llm_client(prompt)
            
            # Parse and apply classifications
            for line in response.strip().split('\n'):
                if ':' in line and line.strip().upper().startswith('ITEM '):
                    try:
                        item_num = int(line.split(':')[0].strip().upper().replace('ITEM ', '')) - 1
                        if 0 <= item_num < len(batch):
                            subcategory = line.split(':', 1)[1].strip()
                            if subcategory in subcategories:
                                item_id = batch[item_num]
                                new_path = f"{current_path.rstrip('/')}/{subcategory}"
                                self.fs.organize_item(item_id, new_path)
                                items_organized += 1
                    except (ValueError, IndexError):
                        continue
        
        # Organize remaining items by similarity
        if len(items) > sample_size:
            # Collect examples for each subcategory
            examples_by_subcategory = defaultdict(list)
            for item_id in sample_ids:
                item = self.fs.items[item_id]
                parts = item.path.strip('/').split('/')
                if len(parts) > 1:  # Has a subcategory
                    subcategory = parts[-1]
                    examples_by_subcategory[subcategory].append(item)
            
            # Function to find best matching subcategory
            def find_best_subcategory(item):
                item_tokens = set(re.findall(r'\w+', item.raw_metadata.lower()))
                best_score = -1
                best_subcat = None
                
                for subcat, examples in examples_by_subcategory.items():
                    total_score = 0
                    for ex in examples:
                        ex_tokens = set(re.findall(r'\w+', ex.raw_metadata.lower()))
                        score = len(item_tokens.intersection(ex_tokens))
                        total_score += score
                    
                    avg_score = total_score / len(examples) if examples else 0
                    if avg_score > best_score:
                        best_score = avg_score
                        best_subcat = subcat
                
                return best_subcat
            
            # Process remaining items
            remaining_ids = [id for id in items if id not in sample_ids]
            for item_id in remaining_ids:
                item = self.fs.items[item_id]
                best_subcat = find_best_subcategory(item)
                
                if best_subcat:
                    new_path = f"{current_path.rstrip('/')}/{best_subcat}"
                    self.fs.organize_item(item_id, new_path)
                    items_organized += 1
        
        return f"Created {len(subcategories)} subcategories: {', '.join(subcategories)}\nOrganized {items_organized} items into subcategories."
    
    def _action_assign_tags(self):
        """Assign meaningful tags to items in the current view."""
        # Determine which items to tag (search results or current path)
        if self.search_results:
            # Use search results
            items_to_tag = []
            for idx in self.search_results[:50]:  # Limit to first 50 for efficiency
                item = self.item_pool[idx]
                item_id = item.get('item_id')
                if item_id in self.fs.items:
                    items_to_tag.append(item_id)
        else:
            # Use current path
            current_path = self.context["current_path"]
            items_to_tag = self.fs.get_items_by_path(current_path)[:50]
        
        if not items_to_tag:
            return "No items to tag."
        
        # Sample items for tag generation
        sample_size = min(10, len(items_to_tag))
        sample_ids = random.sample(items_to_tag, sample_size)
        
        # Get sample items metadata
        sample_text = ""
        for i, item_id in enumerate(sample_ids):
            item = self.fs.items[item_id]
            sample_text += f"Item {i+1}: {item.raw_metadata[:200]}...\n\n"
        
        # Generate tags using LLM
        tag_prompt = f"""
        Based on these product descriptions, generate 5-10 important tags that would be useful for filtering and organizing these items.
        Focus on product attributes like material, size, style, features, use cases, target audience, etc.
        
        {sample_text}
        
        Format your response as a comma-separated list of tags only.
        """
        
        tag_response = self.llm_client(tag_prompt)
        
        # Parse tags
        tags = [tag.strip() for tag in tag_response.split(',')]
        unique_tags = set(tags)
        
        # Now determine which tags apply to which items
        tagged_count = 0
        
        # Process in batches
        batch_size = 5
        batches = [items_to_tag[i:i+batch_size] for i in range(0, len(items_to_tag), batch_size)]
        
        for batch in batches:
            # Format items for the LLM
            batch_text = "\n".join([
                f"Item {i+1}: {self.fs.items[item_id].raw_metadata[:200]}..." 
                for i, item_id in enumerate(batch)
            ])
            
            # Create prompt for tagging
            prompt = f"""
            For each item, assign appropriate tags from this list:
            {', '.join(unique_tags)}
            
            {batch_text}
            
            Format your response as:
            ITEM 1: tag1, tag2, tag3
            ITEM 2: tag4, tag5
            ...
            
            Only use tags from the provided list.
            """
            
            response = self.llm_client(prompt)
            
            # Parse and apply tags
            for line in response.strip().split('\n'):
                if ':' in line and line.strip().upper().startswith('ITEM '):
                    try:
                        item_num = int(line.split(':')[0].strip().upper().replace('ITEM ', '')) - 1
                        if 0 <= item_num < len(batch):
                            item_tags = [t.strip() for t in line.split(':', 1)[1].strip().split(',')]
                            item_id = batch[item_num]
                            
                            # Add valid tags
                            for tag in item_tags:
                                if tag in unique_tags:
                                    self.fs.add_tag(item_id, tag)
                                    # Track relevant tags for search context
                                    self.context["relevant_tags"].add(tag)
                            
                            tagged_count += 1
                    except (ValueError, IndexError):
                        continue
        
        return f"Generated {len(unique_tags)} tags: {', '.join(unique_tags)}\nTagged {tagged_count} items."
    
    def _action_filter_by_tags(self, tags_param):
        """Filter items by tags."""
        if not tags_param:
            # Use relevant tags from context
            tags = list(self.context["relevant_tags"])
            if not tags:
                return "No tags specified for filtering."
        else:
            # Parse tags from parameter
            tags = [t.strip() for t in tags_param.split(',')]
        
        # Get items with these tags
        items = self.fs.get_items_by_tags(tags, require_all=True)
        
        if not items:
            # Try with any match instead of all matches
            items = self.fs.get_items_by_tags(tags, require_all=False)
        
        # Update search results with matching items
        # We need to map item_ids back to indices in the item_pool
        self.search_results = []
        for i, item in enumerate(self.item_pool):
            item_id = item.get('item_id', '')
            if item_id in items:
                self.search_results.append(i)
        
        # Sample results to show
        sample_size = min(5, len(items))
        samples_text = ""
        
        if sample_size > 0:
            sample_ids = random.sample(items, sample_size)
            samples = []
            
            for i, item_id in enumerate(sample_ids):
                item = self.fs.items[item_id]
                all_tags = ', '.join(item.tags)
                samples.append(f"Result {i+1}:\nPath: {item.path}\nTags: {all_tags}\nDescription: {item.raw_metadata[:150]}...")
            
            samples_text = "\n\n".join(samples)
        else:
            samples_text = "No matching items found."
        
        return f"Filtered by tags [{', '.join(tags)}]\nFound {len(items)} matching items.\n\n{samples_text}"
    
    def _action_evaluate_items(self):
        """Evaluate current items against search requirements."""
        # Determine which items to evaluate
        items_to_evaluate = []
        
        if self.search_results:
            # Use search results
            for idx in self.search_results[:10]:  # Limit to top 10
                item = self.item_pool[idx]
                item_id = item.get('item_id')
                if item_id in self.fs.items:
                    items_to_evaluate.append((idx, item_id))
        else:
            # Use current path
            current_path = self.context["current_path"]
            path_items = self.fs.get_items_by_path(current_path)
            
            for i, item_id in enumerate(path_items[:10]):  # Limit to top 10
                for idx, item in enumerate(self.item_pool):
                    if item.get('item_id') == item_id:
                        items_to_evaluate.append((idx, item_id))
                        break
        
        if not items_to_evaluate:
            return "No items to evaluate."
        
        # Format requirements
        requirements_text = "\n".join([f"{i+1}. {req}" for i, req in enumerate(self.context["requirements"])])
        
        # Evaluate each item
        evaluations = []
        
        for item_index, (idx, item_id) in enumerate(items_to_evaluate):
            item = self.item_pool[idx]
            fs_item = self.fs.items[item_id]
            
            # Format item details
            item_text = f"""
            Item {item_index+1}:
            Category: {fs_item.category}
            Path: {fs_item.path}
            Tags: {', '.join(fs_item.tags)}
            Description: {fs_item.raw_metadata}
            """
            
            # Create evaluation prompt
            prompt = f"""
            Evaluate how well this product matches the search requirements:
            
            Requirements:
            {requirements_text}
            
            {item_text}
            
            Rate each requirement on a scale of 1-5 where:
            1 = Does not match
            5 = Perfect match
            
            Format your response as:
            Requirement 1: (score) explanation
            Requirement 2: (score) explanation
            ...
            OVERALL: (score) overall assessment
            """
            
            evaluation = self.llm_client(prompt)
            
            # Extract overall score
            overall_score = 0
            for line in evaluation.strip().split('\n'):
                if line.strip().upper().startswith('OVERALL:'):
                    try:
                        score_text = re.search(r'\((\d+)\)', line)
                        if score_text:
                            overall_score = int(score_text.group(1))
                    except (ValueError, AttributeError):
                        overall_score = 0
            
            # Add to evaluations
            evaluations.append({
                "item_index": idx,
                "item_id": item_id,
                "evaluation": evaluation,
                "overall_score": overall_score
            })
        
        # Sort evaluations by score
        evaluations.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Update search results based on evaluations
        self.search_results = [e["item_index"] for e in evaluations]
        
        # Format evaluation summary
        summary = []
        for i, eval_data in enumerate(evaluations[:5]):  # Show top 5
            item = self.item_pool[eval_data["item_index"]]
            summary.append(f"Item {i+1} (Score: {eval_data['overall_score']}/5):\n" +
                          f"ID: {item.get('item_id')}\n" +
                          f"Description: {item.get('metadata', '')[:100]}...")
        
        return f"Evaluated {len(evaluations)} items against {len(self.context['requirements'])} requirements.\n\n" + "\n\n".join(summary)
    
    def _finalize_results(self):
        """Organize and return final search results."""
        if not self.search_results:
            return {
                "success": False,
                "message": "No matching items found for the query.",
                "items": []
            }
        
        # Limit to top results
        top_results = self.search_results[:20]
        
        # Format results
        results = []
        for idx in top_results:
            item = self.item_pool[idx]
            item_id = item.get('item_id')
            
            result = {
                "item_id": item_id,
                "metadata": item.get('metadata', ''),
                "category": item.get('category', '')
            }
            
            # Add file system organization info if available
            if item_id in self.fs.items:
                fs_item = self.fs.items[item_id]
                result["path"] = fs_item.path
                result["tags"] = list(fs_item.tags)
            
            results.append(result)
        
        # Generate a summary of the search process
        summary_prompt = f"""
        Summarize the search process for query: "{self.current_query}"
        
        Requirements identified:
        {", ".join(self.context["requirements"])}
        
        Paths explored:
        {", ".join(self.context["explored_paths"])}
        
        Relevant tags:
        {", ".join(self.context["relevant_tags"])}
        
        Write a 3-5 sentence summary of how the search was performed and what kind of results were found.
        """
        
        search_summary = self.llm_client(summary_prompt)
        
        return {
            "success": True,
            "message": "Search completed successfully.",
            "summary": search_summary,
            "requirements": self.context["requirements"],
            "relevant_tags": list(self.context["relevant_tags"]),
            "items": results
        }

########################################
# 6. New Main Function & Utilities
########################################

def save_file_system(fs, path):
    """Save the file system state to disk."""
    data = {
        "directories": {k: list(v) for k, v in fs.directories.items()},
        "items": {k: v.to_dict() for k, v in fs.items.items()},
        "tag_index": {k: list(v) for k, v in fs.tag_index.items()}
    }
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"File system saved to {path}")

def load_file_system(path):
    """Load the file system state from disk."""
    fs = CategoryFileSystem()
    
    if not Path(path).exists():
        return fs
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    # Restore directories
    for path, item_ids in data["directories"].items():
        fs.directories[path] = set(item_ids)
    
    # Restore items
    for item_id, item_dict in data["items"].items():
        fs.items[item_id] = ItemMetadata.from_dict(item_dict)
    
    # Restore tag index
    for tag, item_ids in data["tag_index"].items():
        fs.tag_index[tag] = set(item_ids)
    
    print(f"File system loaded from {path} with {len(fs.items)} items")
    return fs

def create_output_report(results, output_path):
    """Create a detailed HTML report of search results."""
    if not results["success"]:
        return
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results - {results["summary"][:50]}...</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            .item {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
            .item:hover {{ background-color: #f9f9f9; }}
            .tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 10px 0; }}
            .tag {{ background-color: #eee; padding: 3px 8px; border-radius: 10px; font-size: 12px; }}
            .path {{ color: #666; font-style: italic; }}
            .item-id {{ color: #999; font-size: 12px; }}
            h1 {{ color: #333; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .requirements {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Search Results</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>{results["summary"]}</p>
        </div>
        
        <div class="requirements">
            <h2>Search Requirements</h2>
            <ul>
    """
    
    for req in results["requirements"]:
        html_content += f"<li>{req}</li>\n"
    
    html_content += """
            </ul>
        </div>
        
        <h2>Top Results</h2>
    """
    
    for item in results["items"]:
        # Format tags
        tags_html = ""
        if "tags" in item:
            for tag in item["tags"]:
                tags_html += f'<span class="tag">{tag}</span>'
        
        html_content += f"""
        <div class="item">
            <h3>{item["metadata"][:100]}...</h3>
            <div class="item-id">ID: {item["item_id"]}</div>
            <div class="path">Path: {item.get("path", item["category"])}</div>
            <div class="tags">{tags_html}</div>
            <p>{item["metadata"]}</p>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Output report saved to {output_path}")

def ensure_cache_dir():
    """Ensure cache directory exists."""
    Path('cache').mkdir(exist_ok=True)

def main():
    """Main function to run the improved search agent."""
    ensure_cache_dir()
    
    # Load data
    data_pkl = Path('cache/queries_item_pool.pkl')
    item_pool, queries = prepare_item_query(data_pkl)
    
    # Load or initialize file system
    fs_pkl = Path('cache/category_file_system.pkl')
    fs = load_file_system(fs_pkl)
    
    # Create LLM client
    llm_client = create_llm_client()
    
    # Create and initialize agent
    agent = ReactAgent(item_pool, fs, llm_client)
    
    # Initialize file system if needed
    if len(fs.items) == 0:
        agent.initialize_file_system()
        save_file_system(fs, fs_pkl)
    
    # Use a sample query from the dataset
    query = queries[0]['query']
    print(f"Query: {query}")
    
    # Run the search process
    results = agent.search(query, max_steps=10)
    
    # Save updated file system
    save_file_system(fs, fs_pkl)
    
    # Create an output report
    create_output_report(results, Path('cache/search_results.html'))
    
    print("Search complete.")
    
    # Print top matches
    if results["success"]:
        print("\nTop Matches:")
        for i, item in enumerate(results["items"][:5]):
            print(f"{i+1}. {item['metadata'][:100]}...")
            if "path" in item:
                print(f"   Path: {item['path']}")
            if "tags" in item:
                print(f"   Tags: {', '.join(item['tags'])}")
            print()
    else:
        print(f"\nSearch failed: {results['message']}")

if __name__ == "__main__":
    main()