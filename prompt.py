sys_expert = "You are an expert in categorizing items into a hiearchical file path."
cat_prompt = """Start with root, and use category as the first path. 

- In level names, replace space with underscores.
- You must use all three levels
- Device the most generic names for each level subcategory; ensure that the 2nd and 3rd level subcategories are inclusive to other items; Do not use special brand names as subcategory.
- Use the existing directory if possible, or branch off any level if the category should be different.

Existing directory:
%s

What is the three level path of the following item?

%s


Return in the following format:
{
    "Reasoning": "....",
    "Path": "/%s/2nd-level/3rd-level/%s"
}"""

request_prompt = f"""You are an expert in finding the most appropriate location in an existing hierarchical directory.

Existing directory structure:
{{0}}

A user has described a product with this query:
"{{1}}"

Your task is to:
1. Analyze the query carefully to understand what the product is
2. Examine the EXISTING directory structure
3. Find the BEST MATCHING path within the existing structure (don't create new paths)
4. Choose the most appropriate 3-level deep location (/level1/level2/level3/) where this item belongs

The product has ID: {{2}}

Return your answer in the following format:
{{
    "Reasoning": "Your step-by-step analysis of why this path is appropriate",
    "Path": "/existing-level1/existing-level2/existing-level3/{{2}}"
}}
"""  
# .format(json.dumps(tree_dict, indent=2), request['query'], request['item_id'], request['item_id'])