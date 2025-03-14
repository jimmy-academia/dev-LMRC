import re
import random
from typing import List, Dict, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import pickle
from tqdm import tqdm

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
        # Normalize path (ensure it has leading slash)
        if not path.startswith('/'):
            path = '/' + path
            
        # For root path, collect items from all top-level categories
        if path == '/':
            result = set()
            for directory, items in self.directories.items():
                if directory.count('/') == 1:  # Only top-level categories
                    result.update(items)
            return list(result)
        
        # Direct lookup first
        result = set(self.directories.get(path, set()))
        
        # Also add from subdirectories if we're looking for a category/subcategory
        for directory, items in self.directories.items():
            if directory != path and directory.startswith(f"{path}/"):
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
        from collections import Counter
        tag_counts = Counter()
        for item_id in item_ids:
            if item_id in self.items:
                tag_counts.update(self.items[item_id].tags)
        
        return [(tag, count) for tag, count in tag_counts.items() if count >= min_count]
    
    # def suggest_subcategories(self, path: str, llm_client: Any) -> List[str]:
    #     """Use LLM to suggest potential subcategories for items in a path."""
    #     item_ids = self.get_items_by_path(path)
    #     if not item_ids:
    #         return []
            
    #     # Sample items to analyze
    #     sample_size = min(5, len(item_ids))
    #     sample_items = [self.items[item_id] for item_id in random.sample(item_ids, sample_size)]
        
    #     # Build prompt for the LLM
    #     descriptions = "\n\n".join([
    #         f"Item {i+1}: {item.raw_metadata[:300]}..." 
    #         for i, item in enumerate(sample_items)
    #     ])
        
    #     prompt = f"""
    #     I have a collection of items in the category path '{path}'. 
    #     Here are {sample_size} representative items:
        
    #     {descriptions}
        
    #     Based on these items, suggest 3-5 meaningful subcategories that would help organize
    #     these types of products better. Format your response as a comma-separated list
    #     of subcategory names only.
    #     """
        
    #     # Call LLM to get subcategories
    #     response = llm_client(prompt)
        
    #     # Parse the response
    #     subcategories = [s.strip() for s in response.split(',')]
    #     return subcategories

########################################
# 3. Search Indexing System
########################################

class EnhancedIndex:
    """Enhanced search index with category, text, and tag-based search."""
    def __init__(self, item_pool=None, cache_path=None):
        # Text-based inverted index
        self.text_index = defaultdict(set)
        self.category_index = defaultdict(set)
        
        # Try to load from cache first if path is provided
        if cache_path and Path(cache_path).exists():
            try:
                print(f"Loading search index from {cache_path}...")
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.text_index = defaultdict(set, {k: set(v) for k, v in data.get('text_index', {}).items()})
                    self.category_index = defaultdict(set, {k: set(v) for k, v in data.get('category_index', {}).items()})
                    self.subcategory_suggestions = data.get('subcategory_suggestions', {})
                print(f"Search index loaded with {len(self.text_index)} tokens and {len(self.category_index)} categories")
                return
            except Exception as e:
                print(f"Error loading cache: {e}, building index from scratch...")
        
        # If we get here, either no cache or couldn't load it
        # Build the indexes
        if item_pool:
            print("Building search indexes...")
            self._build_indexes(item_pool)
            
            # Save to cache if path provided
            if cache_path:
                try:
                    print(f"Saving search index to {cache_path}...")
                    data = {
                        'text_index': {k: list(v) for k, v in self.text_index.items()},
                        'category_index': {k: list(v) for k, v in self.category_index.items()},
                        'subcategory_suggestions': self.subcategory_suggestions
                    }
                    with open(cache_path, 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print("Search index saved successfully")
                except Exception as e:
                    print(f"Error saving cache: {e}")
    
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
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "text_index": {k: list(v) for k, v in self.text_index.items()},
            "category_index": {k: list(v) for k, v in self.category_index.items()},
            "subcategory_suggestions": self.subcategory_suggestions
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EnhancedIndex':
        """Create from dictionary."""
        index = cls()
        
        # Restore text index
        for token, indices in data["text_index"].items():
            index.text_index[token] = set(indices)
        
        # Restore category index
        for category, indices in data["category_index"].items():
            index.category_index[category] = set(indices)
        
        # Restore subcategory suggestions
        index.subcategory_suggestions = data.get("subcategory_suggestions", {})
        
        return index



def load_file_system(path, item_pool=None):
    """Load the file system state from disk."""
    ## only save initial category for now, do dynamic increase path.0, path.1.... latter

    fs = CategoryFileSystem()
    
    if not Path(path).exists():
        initialize_file_system(fs, item_pool)
        save_file_system(fs, path)
        logging.info(f"File system initialized from item_pool with {len(fs.items)} items and saved to {path}.")
        return fs
    else:
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
        
        logging.info(f"File system loaded from {path} with {len(fs.items)} items")
        return fs

def save_file_system(fs, path):
    """Save the file system state to disk."""
    data = {
        "directories": {k: list(v) for k, v in fs.directories.items()},
        "items": {k: v.to_dict() for k, v in fs.items.items()},
        "tag_index": {k: list(v) for k, v in fs.tag_index.items()}
    }
    
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    
    logging.info(f"File system saved to {path}")


def initialize_file_system(fs, item_pool):
    """Process all items into the file system with their existing categories."""
    print("Initializing category file system...")
    
    # Track counts for verification
    category_counts = defaultdict(int)
    items_processed = 0
    items_with_category = 0
    
    logging.info(f"File system saved to {path}")
    for idx, item in enumerate(tqdm(item_pool, desc="Processing items", ncols=88)):
        # Extract item data ensuring we capture the existing category
        item_id = item.get('item_id', f'unknown_{idx}')
        category = item.get('category', '')  # Existing category from item_pool
        metadata = item.get('metadata', '')
        
        # Skip empty categories
        if not category or category.strip() == '':
            continue
            
        items_with_category += 1
        category_counts[category] += 1
        
        # Create item metadata object
        meta = ItemMetadata(
            item_id=item_id,
            category=category,  # Use the category from the item_pool
            raw_metadata=metadata
        )
        
        # Add to file system which will place it in the correct category directory
        fs.add_item(meta)
        items_processed += 1
        
        # Print progress periodically
        if idx % 100000 == 0 and idx > 0:
            print(f"Processed {idx} items, {items_with_category} with categories...")
        
    # Print category statistics
    print(f"\nCategory distribution for {items_processed} items:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} items")
        
    # Verify items are in directories
    total_in_dirs = sum(len(items) for items in fs.directories.values())
    print(f"\nItems in directories: {total_in_dirs}")
    
    # Debug: Check the first few directories
    print("\nSample directory contents:")
    for i, (path, items) in enumerate(list(fs.directories.items())[:5]):
        print(f"  {path}: {len(items)} items")
        if items:
            sample_id = next(iter(items))
            print(f"    Sample item: {fs.items[sample_id].raw_metadata[:50]}...")