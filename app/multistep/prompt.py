sys_expert = "You are an expert in consumer products, able to categorize items, understand requests, and organize items in a hierarchical file path structure."

# Prompt for generating subcategory and tags for an item
category_gen_prompt = """
You are organizing items in a hierarchical directory structure. For an item in the '{category}' category, suggest an appropriate subcategory name.

- Subcategory names should be generic and broadly applicable (not brand-specific)
- Use underscores instead of spaces in subcategory names
- Reuse existing subcategories when appropriate rather than creating too many unique ones
- Provide 2-4 relevant tags for the item

Existing subcategories:
{existing_subcats}

Item to categorize:
- Metadata: {metadata}
- Summary: {summary}

Return your response in the following JSON format:
{{
    "SubcategoryName": "Subcategory_Name",
    "Tags": ["tag1", "tag2", "tag3"],
    "Reasoning": "Brief explanation of why this subcategory is appropriate"
}}
"""

# Prompt for clustering subcategories
category_cluster_prompt = """
You are organizing items in a hierarchical directory structure under the '{category}' category. 
We have generated subcategory assignments for items, but there are too many unique subcategories.

Your task is to consolidate these subcategories into {target_range} final subcategories:
1. Analyze the generated subcategories and their associated items
2. Identify logical groupings or clusters
3. Create a mapping from existing subcategories to your new consolidated subcategories
4. Ensure the new subcategory names are clear, generic, and follow the same format (using underscores instead of spaces)

Subcategory candidates (with item details):
{subcategory_candidates}

Return your response as a JSON object mapping each original subcategory to your new consolidated subcategory name:
{{
    "Original_Subcategory1": "New_Consolidated_Category",
    "Original_Subcategory2": "New_Consolidated_Category",
    "Original_Subcategory3": "Different_Consolidated_Category",
    ...
}}
"""

# Prompt for finding the correct path for a user request
request_prompt = """You are an expert in finding the most appropriate location in an existing hierarchical directory.

Existing directory structure:
{tree_dict}

A user has described a product with this query:
"{query}"

Your task is to:
1. Analyze the query carefully to understand what the product is
2. Examine the EXISTING directory structure
3. Find the BEST MATCHING path within the existing structure (don't create new paths)
4. Choose the most appropriate 3-level deep location (/level1/level2/level3/) where this item belongs

The product has ID: {item_id}

Return your answer in the following format:
{{
    "Reasoning": "Your step-by-step analysis of why this path is appropriate",
    "Path": "/existing-level1/existing-level2/existing-level3/{item_id}"
}}
"""