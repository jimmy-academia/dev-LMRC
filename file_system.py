import re
import pickle
import logging
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

class Node:
    """
    Represents a node in the hierarchical file system.
    Each node maintains information about its items and subcategories.
    """
    def __init__(self, name, parent=None):
        self.name = name
        self.items = set()  # Items directly at this node (not in subcategories)
        self.subcategories = {}  # Maps subcategory name to Node object
        self.total_item_count = 0  # Total items at this node and in all subcategories
        self.parent = parent
    
    def _update_parent_count(self, delta):
        if self.parent is not None: 
            self.parent.update_total_count(delta)


    def add_item(self, item_id):
        """Add an item directly to this node."""
        self.items.add(item_id)
        self.total_item_count += 1
        self._update_parent_count(1)
        
    def remove_item(self, item_id):
        """Remove an item from this node."""
        if item_id in self.items:
            self.items.remove(item_id)
            self.total_item_count -= 1
            self._update_parent_count(-1)
            return True
        return False
    
    def update_total_count(self, delta):
        """Update the total item count for this node and propagate upward."""
        self.total_item_count += delta
        if self.parent is not None:
            self._update_parent_count(delta)

    def get_subcategory(self, name):
        """Get a subcategory by name or create it if it doesn't exist."""
        if name not in self.subcategories:
            self.subcategories[name] = Node(name)
        return self.subcategories[name]
    
    def get_subcategories_info(self):
        """Get information about the subcategories."""
        return {name: node.total_item_count for name, node in self.subcategories.items()}
    
    def __repr__(self):
        return f"Node({self.name}, {len(self.items)} direct items, {self.total_item_count} total items, {len(self.subcategories)} subcategories)"

class FileSystem:
    """
    A hierarchical file system for organizing and retrieving items.
    Uses a tree structure for efficient navigation and lookups.
    """
    
    def __init__(self, item_pool):
        """Initialize the file system with items from the item pool."""
        self.item_pool = item_pool
        self.id_to_item = {}  # Maps item_id to item for O(1) lookup
        self.id_to_paths = {}  # Maps item_id to its current path
        self.tags = defaultdict(set)  # Maps tags to sets of item_ids
        self.item_tags = defaultdict(set)  # Maps item_ids to their tags
        self.inverted_index = defaultdict(set)  # Word -> set of item_ids
        
        # Create the root node
        self.root = Node('root')
        
        # Place items in their initial categories and build the index
        self._initialize_categories()
        self._build_inverted_index()
    
    def _initialize_categories(self):
        """Place items in their initial categories."""
        logging.info("Initializing categories...")
        for idx, item in enumerate(tqdm(self.item_pool, desc="Organizing items", ncols=88)):
            item_id = item.get('item_id')
            category = item.get('category', 'Uncategorized')
            
            # Store the item for O(1) lookup
            self.id_to_item[item_id] = item
            
            # Store the item's path
            path = f"/{category}"
            self.id_to_paths[item_id] = path
            
            # Add the item to the appropriate node
            self._add_item_to_path(item_id, path)
    
    def _add_item_to_path(self, item_id, path):
        """Add an item to a specific path in the hierarchy."""
        if not path.startswith('/'):
            path = f"/{path}"
            
        # Split the path into components
        components = path.strip('/').split('/')
        
        # Start at the root
        current = self.root
        
        # Navigate through the path, creating nodes as needed
        for i, component in enumerate(components):
            # For all components except the last, create subcategory nodes
            if i < len(components) - 1:
                current = current.get_subcategory(component)
            else:
                # For the last component, add the item directly
                current_node = current.get_subcategory(component)
                current_node.add_item(item_id)
    
    def _build_inverted_index(self):
        """Build an inverted index for faster keyword searches."""
        logging.info("Building inverted index...")
        for idx, item in enumerate(tqdm(self.item_pool, desc="Indexing items", ncols=88)):
            # Get metadata text; default to empty string if missing
            metadata = item.get('metadata', '')
            item_id = item.get('item_id')
            
            # Process metadata text to lowercase
            metadata_lower = metadata.lower()
            
            # Tokenize: extract words and convert to lowercase
            # Store both individual words and pairs of adjacent words
            tokens = set(re.findall(r'\b\w+\b', metadata_lower))
            
            # Index individual words
            for token in tokens:
                self.inverted_index[token].add(item_id)
                
            # Additionally, index pairs of words (for common phrases)
            words = re.findall(r'\b\w+\b', metadata_lower)
            if len(words) >= 2:
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i+1]}"
                    self.inverted_index[bigram].add(item_id)
                    
            # Also index trigrams for common phrases of 3 words
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                    self.inverted_index[trigram].add(item_id)
    
    def get_item_by_id(self, item_id):
        """Get an item by its ID - O(1) operation."""
        if item_id not in self.id_to_item:
            return None
            
        # Create a copy to avoid modifying the original
        result = self.id_to_item[item_id].copy()
        
        # Add path and tags if available
        if item_id in self.paths:
            result['path'] = self.paths[item_id]
        if item_id in self.item_tags:
            result['tags'] = list(self.item_tags[item_id])
            
        return result
    
    def _get_node_at_path(self, path):
        """Get the node at a specific path."""
        if not path.startswith('/'):
            path = f"/{path}"
            
        # Root path is a special case
        if path == '/':
            return self.root
            
        # Split the path into components
        components = path.strip('/').split('/')
        
        # Start at the root
        current = self.root
        
        # Navigate through the path
        for component in components:
            if component in current.subcategories:
                current = current.subcategories[component]
            else:
                # Path not found
                return None
                
        return current
        
    def navigate_to(self, path):
        """
        Navigate to a path and return information about that location.
        Returns a dict with:
        - total_items: number of items at this path and subpaths
        - subcategories: dict mapping subcategory names to item counts
        - direct_items: items at this path not in a subcategory
        """
        node = self._get_node_at_path(path)
        
        if not node:
            # Path not found, return empty result
            return {
                'total_items': 0,
                'subcategories': {},
                'direct_items': set()
            }
            
        return {
            'total_items': node.total_item_count,
            'subcategories': node.get_subcategories_info(),
            'direct_items': node.items
        }
        
    def keyword_search(self, keywords, path=None):
        """
        Search for items containing ANY of the keywords within the given path.
        If path is None, search all items.
        Returns a list of item_ids.
        
        Keywords can be either a single string or a list of keywords.
        If a single string is provided, it will be split by commas.
        """
        # Handle input as either a string or a list
        if isinstance(keywords, str):
            # Split by commas and strip whitespace
            keyword_list = [k.strip() for k in keywords.split(',')]
        else:
            keyword_list = keywords
        
        if not keyword_list:
            return []
        
        # Process each keyword to ensure proper matching
        processed_keywords = []
        for kw in keyword_list:
            # Remove any surrounding quotes or spaces
            kw = kw.strip().strip('"\'')
            if kw:
                processed_keywords.append(kw)
        
        # Get search results for each keyword
        results = set()
        for keyword in processed_keywords:
            # Search for exact multi-word phrases or individual words
            if ' ' in keyword:
                # For multi-word phrases, search for items that contain all words
                words = keyword.lower().split()
                keyword_matches = set()
                
                for item_id, item in self.id_to_item.items():
                    metadata = item.get('metadata', '').lower()
                    # Check if ALL words from the keyword phrase appear in the metadata
                    if all(word in metadata for word in words):
                        keyword_matches.add(item_id)
            else:
                # For single words, use the inverted index
                keyword_matches = self.inverted_index.get(keyword.lower(), set())
            
            # Add keyword matches to overall results
            results.update(keyword_matches)
        
        # If path is specified, filter results to only include items at or below that path
        if path:
            node = self._get_node_at_path(path)
            if not node:
                return []
                
            # Get all items at this node and in its subcategories
            path_items = self._get_all_items_under_node(node)
            
            # Filter results to only include items at or below the path
            results = results.intersection(path_items)
        
        return list(results)
    
    def _get_all_items_under_node(self, node):
        """Recursively get all items under a node."""
        all_items = set(node.items)
        
        # Recursively get items from subcategories
        for subcat_node in node.subcategories.values():
            all_items.update(self._get_all_items_under_node(subcat_node))
            
        return all_items
    
    def create_subcategory(self, parent_path, subcategory_name, item_ids):
        """
        Create a new subcategory under the parent path and move specified items into it.
        Returns the number of items moved.
        """
        if not parent_path.startswith('/'):
            parent_path = f"/{parent_path}"
        
        # Check if we're already at the maximum depth (3 levels)
        if parent_path.count('/') >= 3:
            logging.warning(f"Cannot create subcategory under {parent_path}: maximum depth reached")
            return 0
        
        # Get the parent node
        parent_node = self._get_node_at_path(parent_path)
        if not parent_node:
            logging.warning(f"Parent path {parent_path} not found")
            return 0
            
        # Create the new path
        new_path = f"{parent_path}/{subcategory_name}" if parent_path != "/" else f"/{subcategory_name}"
        
        # Get all items under the parent node
        parent_items = self._get_all_items_under_node(parent_node)
        
        # Ensure all items exist in the parent path or its subpaths
        valid_items = set(item_ids).intersection(parent_items)
        
        if not valid_items:
            logging.warning(f"No valid items to move to {new_path}")
            return 0
        
        # Create the subcategory node if it doesn't exist
        if subcategory_name not in parent_node.subcategories:
            parent_node.subcategories[subcategory_name] = Node(subcategory_name)
        
        # Get the subcategory node
        subcat_node = parent_node.subcategories[subcategory_name]
        
        # Move items to the new subcategory
        moved_count = 0
        for item_id in valid_items:
            # Get the old path and node
            old_path = self.paths[item_id]
            old_node = self._get_node_at_path(old_path)
            
            # Remove from old node
            if old_node and old_node.remove_item(item_id):
                # Update the item's path
                self.paths[item_id] = new_path
                
                # Add to new subcategory
                subcat_node.add_item(item_id)
                moved_count += 1
        
        logging.info(f"Created subcategory {new_path} with {moved_count} items")
        return moved_count
    
    def add_tag(self, item_ids, tag):
        """Add a tag to specified items."""
        tag = tag.lower()
        for item_id in item_ids:
            self.item_tags[item_id].add(tag)
            self.tags[tag].add(item_id)
        return len(item_ids)
    
    def get_items_by_tag(self, tag, path=None):
        """
        Get all items with the specified tag.
        If path is provided, only return items within that path.
        """
        tag = tag.lower()
        items = self.tags.get(tag, set())
        
        if path:
            node = self._get_node_at_path(path)
            if not node:
                return []
                
            # Get all items at this node and in its subcategories
            path_items = self._get_all_items_under_node(node)
            
            # Filter items to only include those at or below the path
            items = items.intersection(path_items)
        
        return list(items)
    
    def verify_categories(self):
        """Print statistics about categories."""
        print("\nCategory statistics:")
        
        # Get all top-level categories and their item counts
        categories = []
        for name, node in self.root.subcategories.items():
            categories.append((name, node.total_item_count))
        
        # Sort by category name
        categories.sort()
        
        # Print three categories per row
        for i in range(0, len(categories), 3):
            row = []
            for j in range(3):
                if i + j < len(categories):
                    cat, count = categories[i + j]
                    row.append(f"{cat}: {count}")
            
            print("  " + "  |  ".join(row))
    
    def remove_tag(self, item_ids, tag):
        """Remove a tag from specified items."""
        tag = tag.lower()
        removed_count = 0
        
        for item_id in item_ids:
            # Check if the item has this tag
            if tag in self.item_tags[item_id]:
                # Remove from item_tags
                self.item_tags[item_id].remove(tag)
                
                # Remove from tags
                self.tags[tag].remove(item_id)
                
                # If tag has no more items, remove it
                if not self.tags[tag]:
                    del self.tags[tag]
                    
                # If item has no more tags, clean up
                if not self.item_tags[item_id]:
                    del self.item_tags[item_id]
                    
                removed_count += 1
                
        return removed_count
    
    def save(self, path):
        """Save the file system to a pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"File system saved to {path}")