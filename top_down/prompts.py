"""
Contains prompts for the RecursiveAgent
"""

# System prompt that establishes the role and overall task
SYSTEM_PROMPT = """You are an advanced search system designed to efficiently navigate through hierarchical categories to find the most relevant products.
Your task is to analyze categories, make strategic decisions about which paths to explore, and organize items based on relevance to the user's query.

You should reason carefully about each decision, considering:
1. Which categories are most likely to contain items matching the query
2. When to create new categories to better organize items
3. How to efficiently use resources, focusing on the most promising search paths

Think step-by-step and be decisive in your recommendations.
"""

# Prompt for deciding whether to create a new subcategory, select an existing one, or pass
DECIDE_CREATE_SELECT_PASS_PROMPT = """
Based on the user's query: "{query}"

You need to decide the best action for the current path: {path}

Available actions:
1. SELECT - Choose one of the existing subcategories to explore next
2. CREATE - Create a new subcategory from the direct items at this path
3. PASS - Skip further exploration at this level as it's unlikely to contain relevant items

Current state:
- Direct items at this path: {direct_item_count} items
- Available subcategories: {subcategories_info}
- Relevance information: {relevance_info}

Analyze the situation and recommend ONE action (SELECT, CREATE, or PASS) with a brief explanation.
Format your response as JSON:
{{
  "reasoning": "<your detailed reasoning process>",
  "decision": "<SELECT|CREATE|PASS>"
}}
"""

# Prompt for selecting the most relevant subcategory to explore next
SELECT_SUBCATEGORY_PROMPT = """
Based on the user's query: "{query}"

You need to select the most promising subcategory to explore from the current path: {path}

Available subcategories:
{subcategories_detailed}

Your task is to analyze each subcategory and determine which one is most likely to contain items relevant to the query.
Consider:
- How directly the subcategory name relates to the query terms
- The number of items in each subcategory (more items may mean higher chance of relevance)
- The specificity of the subcategory (more specific categories may be more accurate)

Format your response as JSON:
{{
  "reasoning": "<your detailed analysis of each subcategory's relevance>",
  "selected_subcategory": "<name of the most relevant subcategory>"
}}
"""

# Prompt for creating a new subcategory from direct items
CREATE_SUBCATEGORY_PROMPT = """
Based on the user's query: "{query}"

You need to create a new subcategory at the current path: {path}

There are {direct_item_count} direct items at this path. Here are some sample items:
{sample_items}

Your task is to:
1. Suggest an appropriate name for the new subcategory that relates to the query
2. Identify which items should belong in this subcategory based on the query
3. Explain your reasoning for this organization

Format your response as JSON:
{{
  "reasoning": "<your detailed thought process>",
  "subcategory_name": "<descriptive name for the new subcategory>",
  "item_selection_criteria": "<criteria for selecting items for this subcategory>"
}}
"""

# Helper function to format subcategory information
def format_subcategories_info(subcategories):
    """Format subcategories dict into a readable string."""
    if not subcategories:
        return "No subcategories available"
    
    info = []
    for name, count in subcategories.items():
        info.append(f"'{name}': {count} items")
    
    return ", ".join(info)

# Helper function to format detailed subcategory information
def format_subcategories_detailed(subcategories):
    """Format subcategories with more detailed information for selection."""
    if not subcategories:
        return "No subcategories available"
    
    details = []
    for name, count in subcategories.items():
        details.append(f"- '{name}': Contains {count} items")
    
    return "\n".join(details)

# Helper function to format sample items
def format_sample_items(items, fs, max_samples=5):
    """Format a sample of items with their metadata."""
    if not items:
        return "No items available"
    
    samples = []
    for i, item_id in enumerate(list(items)[:max_samples]):
        item = fs.get_item_by_id(item_id)
        if item:
            # Truncate metadata if it's too long
            metadata = item.get('metadata', '')
            if len(metadata) > 100:
                metadata = metadata[:100] + "..."
            samples.append(f"{i+1}. {metadata} (ID: {item_id})")
    
    return "\n".join(samples)

# Helper function to get relevance information for the current path
def get_relevance_info(query, path, fs):
    """Get information about the relevance of the current path to the query."""
    # This could be enhanced with more sophisticated relevance metrics
    keywords = [kw.strip() for kw in query.split() if len(kw.strip()) > 3]
    path_components = [p for p in path.split('/') if p]
    
    relevance = []
    
    # Check if path components match query keywords
    matching_components = [p for p in path_components if any(kw.lower() in p.lower() for kw in keywords)]
    if matching_components:
        relevance.append(f"Path contains {len(matching_components)} components related to the query")
    
    # You could add additional relevance signals here
    
    return ", ".join(relevance) if relevance else "No direct relevance signals detected"