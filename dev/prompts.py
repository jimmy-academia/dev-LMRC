"""
Contains prompts for the ReactAgent
"""

SYSTEM_PROMPT = """You are a helpful search assistant that helps users find products based on their query.
You have access to a file system of products organized by categories. You can:
1. Navigate to different category paths
2. Search for products using keywords
3. Create new subcategories to organize products
4. Tag products with relevant attributes

Follow these steps to find the most relevant products:
1. Analyze the query to understand what the user is looking for
2. Navigate to the most relevant category
3. Use keyword search to find relevant products
4. If needed, create subcategories to better organize products
5. Return the most relevant products that match the query

You must use the available actions and think step by step.
"""

AVAILABLE_ACTIONS = """Available actions:
- Navigate(path): Navigate to a specific category path
- Search(keywords): Search for items containing any of these keywords
- CreateSubcategory(name, item_ids): Create a new subcategory with the specified items
- AddTag(item_ids, tag): Add a tag to the specified items
- RemoveTag(item_ids, tag): Remove a tag from the specified items
- GetByTag(tag): Get items with the specified tag
- Complete(items): Finish the search and return the found items
"""


REASONING_PROMPT = """Based on the current state and the user's query, think step-by-step about what action would be most appropriate next.
Explain your reasoning in detail. Don't yet decide on a specific action - just analyze the situation and explain your thoughts.
"""

ACTION_SELECTION_PROMPT = f"""Based on your reasoning, choose ONE of the following actions to perform next:

{AVAILABLE_ACTIONS}

Return ONLY the action in the exact format shown above, with no additional explanation.
For example:
- Navigate("/Electronics")
- Search("air filter, dust")
- CreateSubcategory("BestFilters", ["B001", "B002", "B003"])
- Complete(["B001", "B002"])
"""

QUERY_PROMPT = "Find products matching this query: {query}"


def get_state_info(state, fs_info, max_steps):
    """Format the current state information for the agent."""
    return f"""Current path: {state["current_path"]}
Total items at this path: {fs_info['total_items']}
Subcategories: {fs_info['subcategories']}
Number of items not in subcategories: {len(fs_info['default_items'])}
Step {state["step_count"] + 1} of {max_steps}

{AVAILABLE_ACTIONS}
"""

# Define a new single-step prompt for Reasoning + Action in JSON format without triple backticks.
SINGLE_STEP_PROMPT = """
You are an advanced ReAct agent. Your goal is to achieve the user's query: "{query}"
by performing step-by-step reasoning and choosing from an available set of actions.

Important: You must output valid JSON in the following format:
{{
  "reasoning": "<your hidden chain-of-thought reasoning here>",
  "action": {{
    "name": "<one of the available actions>",
    "params": {{
        // key-value pairs relevant to the action
    }}
  }}
}}

1. Reasoning: Think carefully about the next best step to find or organize items that match the query.
2. Action: Choose exactly one from the available actions:
   - Navigate: Move to a subdirectory/category
     - name: "Navigate"
     - params: {{ "path": "/some/path" }}
   - Search: Search within the current path
     - name: "Search"
     - params: {{ "keywords": ["keyword1", "keyword2", ...] }}
   - CreateSubcategory: Create a new subcategory with provided item IDs
     - name: "CreateSubcategory"
     - params: {{ "name": "subcategory_name", "item_ids": ["id1", "id2", ...] }}
   - AddTag: Add a tag to specified items
     - name: "AddTag"
     - params: {{ "tag": "some_tag", "item_ids": ["id1", "id2", ...] }}
   - RemoveTag: Remove a tag from specified items
     - name: "RemoveTag"
     - params: {{ "tag": "some_tag", "item_ids": ["id1", "id2", ...] }}
   - GetByTag: Retrieve items with a given tag
     - name: "GetByTag"
     - params: {{ "tag": "some_tag" }}
   - Complete: End the search with a final set of item_ids
     - name: "Complete"
     - params: {{ "item_ids": ["id1", "id2", ...] }}

You have the following context about your current state:
{state_info}

Remember: Return only valid JSON. Do NOT include extra keys.
"""

if __name__ == '__main__':
  a = SINGLE_STEP_PROMPT.format(query='query', state_info='state info \n state info')
  print(a)