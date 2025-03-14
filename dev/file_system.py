import re
import pickle
import logging
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

class FileSystem:
    """
    A hierarchical file system for organizing and retrieving items.
    Supports categorization, path navigation, keyword search, and tagging.
    """
    
    def __init__(self, item_pool=None):
        """Initialize the file system with items from the item pool."""
        self.item_pool = item_pool or []
        self.id_to_item = {}  # Maps item_id to item for O(1) lookup
        self.paths = {}  # Maps item_id to its current path
        self.path_to_items = defaultdict(set)  # Maps paths to sets of item_ids
        self.tags = defaultdict(set)  # Maps tags to sets of item_ids
        self.item_tags = defaultdict(set)  # Maps item_ids to their tags
        self.inverted_index = defaultdict(set)  # Word -> set of item_ids
        
        # Place items in their initial categories and build the index
        if item_pool:
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
            self.paths[item_id] = path
            self.path_to_items[path].add(item_id)
    
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
        
    def navigate_to(self, path):
        """
        Navigate to a path and return information about that location.
        Returns a dict with:
        - total_items: number of items at this path and subpaths
        - subcategories: dict mapping subcategory names to item counts
        - default_items: items at this path not in a subcategory
        """
        if not path.startswith('/'):
            path = f"/{path}"
            
        # Get items directly at this path
        items_at_path = self.path_to_items.get(path, set())
        
        # Find all immediate subcategories and count items
        subcategories = {}
        
        # Special case for root path: show all top-level categories
        if path == '/':
            for subpath, items in self.path_to_items.items():
                # Only include paths with one component after root (e.g., /Electronics)
                if subpath.count('/') == 1 and subpath != '/':
                    # Extract category name from path
                    category = subpath[1:]  # Remove leading /
                    subcategories[category] = len(items)
        else:
            # Regular case: find immediate subcategories of the current path
            prefix = path + '/' if path != '/' else '/'
            for subpath, items in self.path_to_items.items():
                # Check if this is an immediate subcategory of the current path
                if subpath.startswith(prefix) and subpath.count('/') == path.count('/') + 1:
                    subcat_name = subpath.split('/')[-1]
                    subcategories[subcat_name] = len(items)
        
        # Get all items at this path and in subdirectories
        all_items = set(items_at_path)
        
        # For non-root paths, include items from subdirectories
        if path != '/':
            prefix = path + '/'
            for subpath, items in self.path_to_items.items():
                if subpath.startswith(prefix):
                    all_items.update(items)
        else:
            # For root path, include all items
            for _, items in self.path_to_items.items():
                all_items.update(items)
        
        return {
            'total_items': len(all_items),
            'subcategories': subcategories,
            'default_items': items_at_path
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
            if not path.startswith('/'):
                path = f"/{path}"
            
            # Get all items at or below the path
            path_items = set()
            prefix = path + '/' if path != '/' else '/'
            
            # Include items directly at this path
            if path in self.path_to_items:
                path_items.update(self.path_to_items[path])
            
            # Include items in subpaths
            for subpath, items in self.path_to_items.items():
                if subpath.startswith(prefix):
                    path_items.update(items)
            
            results = results.intersection(path_items)
        
        return list(results)
    
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
        
        # Create the new path
        new_path = f"{parent_path}/{subcategory_name}"
        
        # Get all items at or below the parent path
        parent_items = set()
        prefix = parent_path + '/' if parent_path != '/' else '/'
        
        if parent_path in self.path_to_items:
            parent_items.update(self.path_to_items[parent_path])
        
        for subpath, items in self.path_to_items.items():
            if subpath.startswith(prefix):
                parent_items.update(items)
        
        # Ensure all items exist in the parent path or its subpaths
        valid_items = set(item_ids).intersection(parent_items)
        
        if not valid_items:
            logging.warning(f"No valid items to move to {new_path}")
            return 0
        
        # Move items to the new subcategory
        moved_count = 0
        for item_id in valid_items:
            # Remove from old path
            old_path = self.paths[item_id]
            self.path_to_items[old_path].remove(item_id)
            
            # Update the item's path
            self.paths[item_id] = new_path
            self.path_to_items[new_path].add(item_id)
            moved_count += 1
            
            # Clean up empty paths
            if not self.path_to_items[old_path]:
                del self.path_to_items[old_path]
        
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
            if not path.startswith('/'):
                path = f"/{path}"
            
            # Get all items at or below the path
            path_items = set()
            prefix = path + '/' if path != '/' else '/'
            
            if path in self.path_to_items:
                path_items.update(self.path_to_items[path])
            
            for subpath, items_at_subpath in self.path_to_items.items():
                if subpath.startswith(prefix):
                    path_items.update(items_at_subpath)
            
            items = items.intersection(path_items)
        
        return list(items)
    
    def verify_categories(self):
        """Print statistics about categories."""
        print("\nCategory statistics:")
        
        # Get all top-level categories and their item counts
        categories = []
        for path, items in sorted(self.path_to_items.items()):
            # Only look at top-level categories (paths with one component after root)
            if path.count('/') == 1:
                category = path[1:]  # Remove leading /
                categories.append((category, len(items)))
        
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

# This function will be implemented in data.py