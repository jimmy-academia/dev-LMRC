def create_prompt_dict(file_tree):
    """
    Create a version of the directory dictionary without item_ids for the prompt.
    """
    prompt_dict = {}
    
    def copy_without_items(src, dest):
        for key, value in src.items():
            if key != "item_ids":
                if isinstance(value, dict):
                    dest[key] = {}
                    copy_without_items(value, dest[key])
                else:
                    dest[key] = value
    
    copy_without_items(file_tree, prompt_dict)
    return prompt_dict 


def find_item_path(item_id, tree, path=""):
    """Find the path to an item in the tree."""
    # Check current level
    if "item_ids" in tree and item_id in tree["item_ids"]:
        return f"{path}/{item_id}"
        
    # Check subdirectories
    for key, value in tree.items():
        if key != "item_ids" and isinstance(value, dict):
            result = find_item_path(item_id, value, f"{path}/{key}")
            if result:
                return result
                
    return None


def calculate_path_distance(path1, path2):
    """
    Calculate a distance metric between two paths.
    
    Args:
        path1: First path string
        path2: Second path string
        
    Returns:
        int: Distance metric (0 = identical paths)
    """
    # Split paths into components
    components1 = path1.strip('/').split('/')
    components2 = path2.strip('/').split('/')
    
    # Remove item_id from calculation
    if len(components1) > 0:
        components1 = components1[:-1]
    if len(components2) > 0:
        components2 = components2[:-1]
    
    # Calculate common prefix length
    common_prefix = 0
    for i in range(min(len(components1), len(components2))):
        if components1[i] == components2[i]:
            common_prefix += 1
        else:
            break
    
    # Distance = unique components in both paths
    distance = (len(components1) - common_prefix) + (len(components2) - common_prefix)
    return distance


def analyze_category_distribution(items_by_category):
    """
    Analyze the distribution of items across categories.
    
    Args:
        items_by_category: Dictionary mapping categories to lists of items
        
    Returns:
        dict: Statistics about category distribution
    """
    stats = {
        "total_items": sum(len(items) for items in items_by_category.values()),
        "total_categories": len(items_by_category),
        "category_counts": {cat: len(items) for cat, items in items_by_category.items()},
        "avg_items_per_category": sum(len(items) for items in items_by_category.values()) / max(1, len(items_by_category)),
        "min_category_size": min((len(items) for items in items_by_category.values()), default=0),
        "max_category_size": max((len(items) for items in items_by_category.values()), default=0)
    }
    return stats


def suggest_subcategory_merges(subcategories, items_by_subcategory, min_size=5):
    """
    Suggest subcategories to merge based on size and similarity.
    
    Args:
        subcategories: List of subcategory names
        items_by_subcategory: Dictionary mapping subcategories to lists of items
        min_size: Minimum suggested size for a subcategory
        
    Returns:
        dict: Suggested merges {small_subcategory: suggested_target}
    """
    suggested_merges = {}
    
    # Identify small subcategories
    small_subcats = [sc for sc in subcategories if len(items_by_subcategory[sc]) < min_size]
    
    # For each small subcategory, suggest a larger one to merge with
    for small_sc in small_subcats:
        best_match = None
        best_score = 0
        
        for other_sc in subcategories:
            if other_sc == small_sc or other_sc in small_subcats:
                continue
                
            # Simple similarity score based on common words
            score = calculate_subcategory_similarity(small_sc, other_sc)
            
            if score > best_score:
                best_score = score
                best_match = other_sc
        
        if best_match:
            suggested_merges[small_sc] = best_match
    
    return suggested_merges


def calculate_subcategory_similarity(subcat1, subcat2):
    """
    Calculate a simple similarity score between subcategory names.
    
    Args:
        subcat1: First subcategory name
        subcat2: Second subcategory name
        
    Returns:
        float: Similarity score (0-1)
    """
    # Convert to lowercase and split into words
    words1 = set(subcat1.lower().replace('_', ' ').split())
    words2 = set(subcat2.lower().replace('_', ' ').split())
    
    # Calculate Jaccard similarity
    if not words1 or not words2:
        return 0
        
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union