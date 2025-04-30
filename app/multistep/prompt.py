sys_expert = "You are an expert in consumer products, able to categorize items, understand requests, and organize items in a hierarchical file path structure."

subcategory_prompt = """
    You are organizing items in a hierarchical directory structure. For an item currently located at the path '{current_path}', suggest the next appropriate branch name.
    
    Current path: {current_path}
    Current level: {level}
    
    - Branch names should be generic and broadly applicable (not brand-specific)
    - Use underscores instead of spaces in branch names
    - Reuse existing branches when appropriate rather than creating too many unique ones
    - Branch names should be coherent with the current path
    
    Existing branches at this level:
    {existing_branches}
    
    Item to categorize: {summary}
    
    Return your response in the following JSON format:
    {{
        "BranchName": "Branch_Name",
        "Reasoning": "Brief explanation of why this branch is appropriate for this item at this level"
    }}
    """
consolidation_prompt = """
    You are organizing items in a hierarchical directory structure. The items are currently at the path '{current_path}'.
    
    We have {branch_count} different branches at this level, which is too many. Please consolidate these into 5-10 final branches:
    
    1. Analyze the generated branches and their associated items
    2. Identify logical groupings or clusters
    3. Create a mapping from existing branches to your new consolidated branches
    4. Ensure the new branch names are clear, generic, and follow the same format (using underscores instead of spaces)
    5. The consolidated branches should still make sense in the context of the current path
    
    Current path: {current_path}
    Current level: {level}
    
    Branch candidates (with sample items):
    {branch_candidates}
    
    Return your response as a JSON object mapping each original branch to your new consolidated branch name:
    {{
        "Original_Branch1": "New_Consolidated_Branch",
        "Original_Branch2": "New_Consolidated_Branch",
        "Original_Branch3": "Different_Consolidated_Branch",
        ...
    }}
    """

# Prompt for finding the correct path for a user request
navigate_prompt = """
    You are helping a user find a product in a hierarchical directory. Let's navigate one level at a time.
    
    Current location: {current_path}
    
    Available branches at this level:
    {available_branches}
    
    User query: "{query}"
    
    Based only on the user's query and the available branches at this level, which branch should we select to navigate deeper?
    
    Choose the SINGLE most appropriate branch that would contain the product the user is looking for.
    
    Return your response in the following JSON format:
    {{
        "SelectedBranch": "branch_name",
        "Reasoning": "Brief explanation of why this branch is most appropriate for the query"
    }}
    """