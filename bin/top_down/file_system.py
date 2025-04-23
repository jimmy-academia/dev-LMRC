import re
import os
import gc
import pickle
import logging
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

class Node:
    """
    Memory-optimized Node implementation for the hierarchical file system.
    Each node maintains information about its items and subcategories.
    """
    def __init__(self, name, parent=None):
        self.name = name
        self.items = set()  # Only store item IDs
        self.subcategories = {}  # Maps subcategory name to Node object
        self.total_item_count = 0  # Total items at this node and in all subcategories
        self.parent = parent  # Store reference to parent for upward propagation
    
    def add_item(self, item_id):
        """Add an item directly to this node."""
        self.items.add(item_id)
        self._update_counts(1)
        
    def remove_item(self, item_id):
        """Remove an item from this node."""
        if item_id in self.items:
            self.items.remove(item_id)
            self._update_counts(-1)
            return True
        return False
    
    def _update_counts(self, delta):
        """Update counts efficiently without deep recursion."""
        node = self
        while node:
            node.total_item_count += delta
            node = node.parent
    
    def get_subcategory(self, name):
        """Get a subcategory by name or create it if it doesn't exist."""
        if name not in self.subcategories:
            self.subcategories[name] = Node(name, self)
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
    
    def __init__(self, item_pool, cache_dir="cache", batch_size=5000):
        """Initialize the file system with items from the item pool."""
        self.item_pool = item_pool
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize data structures
        self.id_to_item = {}  # Maps item_id to item for O(1) lookup
        self.paths = {}       # Maps item_id to its current path
        self.tags = defaultdict(set)  # Maps tags to sets of item_ids
        self.item_tags = defaultdict(set)  # Maps item_ids to their tags
        self.inverted_index = None  # Will be loaded or built
        
        # Create the root node
        self.root = Node('root')
        
        # Place items in their initial categories
        self._initialize_categories()
        
        # Build or load the inverted index
        self._load_or_build_inverted_index()
    
    def _initialize_categories(self):
        """Place items in their initial categories in batches."""
        logging.info("Initializing categories...")
        item_map_file = os.path.join(self.cache_dir, "item_map.pkl")
        node_file = os.path.join(self.cache_dir, "node_tree.pkl")
        paths_file = os.path.join(self.cache_dir, "paths.pkl")
        
        # Check if we can load a cached tree structure
        if os.path.exists(node_file):
            logging.info("Loading tree structure from cache...")
            with open(node_file, 'rb') as f:
                self.root = pickle.load(f)
            with open(item_map_file, 'rb') as f:
                self.id_to_item = pickle.load(f)
            with open(paths_file, 'rb') as f:
                self.paths = pickle.load(f)
            return
        
        # Process items in batches
        total_items = len(self.item_pool)
        for i in range(0, total_items, self.batch_size):
            batch_end = min(i + self.batch_size, total_items)
            batch = self.item_pool[i:batch_end]
            
            for item in tqdm(batch, desc=f"Organizing items {i+1}-{batch_end}", ncols=88):
                item_id = item.get('item_id')
                category = item.get('category', 'Uncategorized')
                
                # Store the item for O(1) lookup
                self.id_to_item[item_id] = item
                
                # Store the item's path
                path = f"/{category}"
                self.paths[item_id] = path
                
                # Add the item to the appropriate node
                self._add_item_to_path(item_id, path)
            
            # Save intermediate progress
            if i % (self.batch_size * 5) == 0 or batch_end == total_items:
                with open(item_map_file, 'wb') as f:
                    pickle.dump(self.id_to_item, f)
                with open(paths_file, 'wb') as f:
                    pickle.dump(self.paths, f)
                with open(node_file, 'wb') as f:
                    pickle.dump(self.root, f)
            
            # Force garbage collection after each batch
            gc.collect()
        
        # Final save
        with open(item_map_file, 'wb') as f:
            pickle.dump(self.id_to_item, f)
        with open(paths_file, 'wb') as f:
            pickle.dump(self.paths, f)
        with open(node_file, 'wb') as f:
            pickle.dump(self.root, f)
            
    def _load_or_build_inverted_index(self):
        """Load the inverted index from cache or build it."""
        index_file = os.path.join(self.cache_dir, "inverted_index.pkl")
        
        if os.path.exists(index_file):
            logging.info(f"Loading inverted index from {index_file}")
            with open(index_file, 'rb') as f:
                self.inverted_index = pickle.load(f)
        else:
            logging.info("Building inverted index (this may take a while)...")
            self.inverted_index = self._build_inverted_index()
            
            # Save the index
            with open(index_file, 'wb') as f:
                pickle.dump(self.inverted_index, f)
    
    def _build_inverted_index(self):
        """Build an inverted index for faster keyword searches."""
        inverted_index = defaultdict(set)
        temp_indices_dir = os.path.join(self.cache_dir, "temp_indices")
        os.makedirs(temp_indices_dir, exist_ok=True)
        
        # Track progress for resuming if needed
        progress_file = os.path.join(temp_indices_dir, "index_progress.pkl")
        if os.path.exists(progress_file):
            with open(progress_file, 'rb') as f:
                processed_items = pickle.load(f)
            logging.info(f"Resuming indexing. Already processed {len(processed_items)} items")
        else:
            processed_items = set()
        
        # Process items in smaller batches to manage memory
        total_items = len(self.item_pool)
        batch_size = min(self.batch_size, 1000)  # Smaller batch size for indexing
        
        for start_idx in range(0, total_items, batch_size):
            end_idx = min(start_idx + batch_size, total_items)
            batch_id = f"batch_{start_idx}_{end_idx}"
            batch_file = os.path.join(temp_indices_dir, f"{batch_id}.pkl")
            
            # Skip if already processed
            if batch_id in processed_items:
                continue
                
            # Create a temporary index for this batch
            batch_index = defaultdict(set)
            
            for i in tqdm(range(start_idx, end_idx), desc=f"Indexing batch {start_idx+1}-{end_idx}", ncols=88):
                if i >= total_items:
                    break
                    
                item = self.item_pool[i]
                item_id = item.get('item_id')
                metadata = item.get('metadata', '')
                
                # Skip if no metadata
                if not metadata:
                    continue
                    
                # Process metadata text to lowercase
                metadata_lower = metadata.lower()
                
                # Index individual words
                for token in set(re.findall(r'\b\w+\b', metadata_lower)):
                    if len(token) >= 3:  # Skip very short words
                        batch_index[token].add(item_id)
                
                # Create bigrams (pairs of words) - only for shorter texts
                words = re.findall(r'\b\w+\b', metadata_lower)
                if len(words) >= 2 and len(words) <= 100:  # Skip very long texts
                    for i in range(len(words) - 1):
                        if len(words[i]) >= 3 and len(words[i+1]) >= 3:
                            bigram = f"{words[i]} {words[i+1]}"
                            batch_index[bigram].add(item_id)
            
            # Save the batch index
            with open(batch_file, 'wb') as f:
                pickle.dump(batch_index, f)
            
            # Update processed items
            processed_items.add(batch_id)
            with open(progress_file, 'wb') as f:
                pickle.dump(processed_items, f)
            
            # Merge into main index
            for token, ids in batch_index.items():
                inverted_index[token].update(ids)
            
            # Clean up batch memory
            del batch_index
            gc.collect()
        
        # Clean up temporary files
        for filename in os.listdir(temp_indices_dir):
            if filename.startswith("batch_"):
                os.remove(os.path.join(temp_indices_dir, filename))
        os.remove(progress_file)
        os.rmdir(temp_indices_dir)
        
        return inverted_index
    
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
    
    def get_item_by_id(self, item_id):
        """Get an item by its ID."""
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
        - default_items: items at this path not in a subcategory
        """
        node = self._get_node_at_path(path)
        
        if not node:
            # Path not found, return empty result
            return {
                'total_items': 0,
                'subcategories': {},
                'default_items': set()
            }
            
        return {
            'total_items': node.total_item_count,
            'subcategories': node.get_subcategories_info(),
            'default_items': node.items
        }
        
    def keyword_search(self, keywords, path=None):
        """
        Search for items containing ANY of the keywords within the given path.
        If path is None, search all items.
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
                
                # Check if the exact phrase is in the index first
                if keyword.lower() in self.inverted_index:
                    keyword_matches = self.inverted_index[keyword.lower()].copy()
                else:
                    # Fall back to searching for items containing all words
                    word_matches = [self.inverted_index.get(word, set()) for word in words]
                    if word_matches:
                        # Start with all items matching the first word
                        keyword_matches = word_matches[0].copy()
                        # Then intersect with items matching subsequent words
                        for matches in word_matches[1:]:
                            keyword_matches.intersection_update(matches)
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
            parent_node.subcategories[subcategory_name] = Node(subcategory_name, parent_node)
        
        # Get the subcategory node
        subcat_node = parent_node.subcategories[subcategory_name]
        
        # Process items in smaller batches to prevent memory issues
        batch_size = 1000
        all_valid_items = list(valid_items)
        moved_count = 0
        
        for i in range(0, len(all_valid_items), batch_size):
            batch = all_valid_items[i:i+batch_size]
            
            for item_id in batch:
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
            
            # Force garbage collection after each batch
            gc.collect()
        
        logging.info(f"Created subcategory {new_path} with {moved_count} items")
        return moved_count
    
    def add_tag(self, item_ids, tag):
        """Add a tag to specified items."""
        tag = tag.lower()
        tagged_count = 0
        
        for item_id in item_ids:
            if item_id in self.id_to_item:  # Only tag items that exist
                self.item_tags[item_id].add(tag)
                self.tags[tag].add(item_id)
                tagged_count += 1
                
        return tagged_count
    
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
            if tag in self.item_tags.get(item_id, set()):
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
        """
        Save the file system to a pickle file, directly storing the node structure.
        This approach prioritizes operational efficiency over storage size.
        """
        logging.info(f"Saving file system to {path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
        
        # Save the complete file system object
        # Note: This includes the root node structure with all subcategories
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        logging.info(f"File system saved to {path}")
    
    @classmethod
    def load(cls, path, item_pool):
        """
        Load a file system from a pickle file, directly restoring the node structure.
        
        Args:
            path: Path to the pickle file
            item_pool: Original item pool (kept for backward compatibility)
            
        Returns:
            A FileSystem instance with the complete tree structure
        """
        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"File system file not found: {path}")
        
        try:
            # Directly load the entire FileSystem object
            with open(path, 'rb') as f:
                fs = pickle.load(f)
            
            # If needed, update the item_pool reference (usually not necessary)
            if hasattr(fs, 'item_pool') and fs.item_pool is not item_pool:
                fs.item_pool = item_pool
            
            logging.info(f"File system loaded from {path}")
            return fs
        except Exception as e:
            logging.error(f"Error loading file system: {e}")
            # If direct loading fails, try to create a new file system
            logging.info("Creating new file system as fallback")
            return cls(item_pool)