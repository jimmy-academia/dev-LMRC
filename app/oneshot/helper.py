
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
