import os
import json
import re
import random
from collections import defaultdict
import openai

##############################
# Data Structures
##############################

def load_items():
    """
    Hypothetical function returning a list of items, each with:
      - item_id
      - category
      - metadata
    We have top-level categories but no subcategories yet.
    """
    # Example stubs
    return [
        {
            "item_id": "B0778XR2QM",
            "category": "Beauty",
            "metadata": "Supergoop! Super Power Sunscreen Mousse SPF 50..."
        },
        {
            "item_id": "12345ABC",
            "category": "Beauty",
            "metadata": "A water-based face cream with hyaluronic acid..."
        },
        {
            "item_id": "XYZ999",
            "category": "Electronics",
            "metadata": "Wireless earbuds with Bluetooth 5.0, noise cancellation..."
        },
        # ...
    ]

##############################
# File Management
##############################

def ensure_path_exists(path):
    """
    Create a directory path if it doesn't already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_item_as_json(item, base_dir="catalog"):
    """
    Saves the item in a JSON file. 
    The item has 'category_path' = e.g. ["Beauty", "Sun Care", "Sunscreen Mousse"].
    We'll convert that into a path like:
      base_dir/Beauty/Sun Care/Sunscreen Mousse/<item_id>.json
    """
    if "category_path" not in item or not item["category_path"]:
        # fallback to top-level category
        cat = item.get("category", "Unknown")
        item["category_path"] = [cat]

    dir_path = os.path.join(base_dir, *item["category_path"])
    ensure_path_exists(dir_path)
    
    # item file path
    file_path = os.path.join(dir_path, f"{item['item_id']}.json")
    
    # Save as JSON
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(item, f, indent=2)

def move_item(item, new_path_list, base_dir="catalog"):
    """
    Moves an item to a new subcategory path (a list of strings).
    Also updates the item['category_path'].
    """
    item["category_path"] = new_path_list
    save_item_as_json(item, base_dir=base_dir)
    # Optionally, remove old file if it existed. 
    # For simplicity, we don't track the old path here.

##############################
# Tagging & LLM Tools
##############################

def call_openai(messages, model="gpt-4", max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens
    )
    return response.choices[0].message["content"]

def propose_subcategories_tool(category_name, sample_items):
    """
    Example 'tool' that calls GPT to propose a multi-level subcategory structure 
    (like a nested tree) given a sample of item metadata.
    Returns a Python data structure describing subcategories (paths).
    """
    # We build a prompt:
    item_snippets = "\n".join(
        f"- ID:{it['item_id']} snippet:{it['metadata'][:200].replace('\n',' ')}"
        for it in sample_items
    )
    prompt = f"""
We have {len(sample_items)} items in the '{category_name}' category. 
Here are short snippets:
{item_snippets}

Please propose a multi-level subcategory structure (like a tree) for these items. 
Use a JSON format. For example:

{{
  "subcategories": [
    {{
      "name": "Sun Care",
      "children": [
        {{ "name": "Sunscreen Mousse" }},
        {{ "name": "Sunscreen Spray" }}
      ]
    }},
    {{
      "name": "Skincare",
      "children": [
        {{ "name": "Moisturizers" }},
        ...
      ]
    }}
  ]
}}

Keep it short and relevant.
"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that proposes subcategory structures."},
        {"role": "user", "content": prompt}
    ]
    raw_json = call_openai(messages)
    return raw_json

def extract_tags_tool(item):
    """
    Calls GPT to extract a set of attribute tags from item metadata.
    Returns a list of {name, value, confidence, source} or an empty list.
    """
    snippet = item["metadata"][:300].replace("\n"," ")
    prompt = f"""
Extract short attribute tags from the following text. 
Return a JSON list with elements like: 
[{{"name":"...", "value":"...", "confidence":..., "source":"llm-extract"}}].

Text: 
{snippet}
"""
    messages = [
        {"role": "system", "content": "You are a tagging assistant."},
        {"role": "user", "content": prompt}
    ]
    response = call_openai(messages)
    # We expect JSON, but it might not be perfectly valid. 
    # For simplicity, let's do a naive parse or use e.g. `json.loads`.
    try:
        tags = json.loads(response)
        # Add/ensure "source":"llm-extract"
        for t in tags:
            t["source"] = "llm-extract"
        return tags
    except:
        return []

##############################
# ReAct Agent
##############################

class FileSystemReActAgent:
    """
    A simplistic ReAct-style agent that:
    - sees top-level categories
    - for each category, proposes subcategories
    - moves items into them
    - extracts tags
    - can adjust items if it decides a deeper path is needed
    """
    
    def __init__(self, items, base_dir="catalog"):
        self.items = items
        self.base_dir = base_dir
        self.conversation = []
        
        # map from category -> list of items
        self.items_by_cat = defaultdict(list)
        for it in items:
            self.items_by_cat[it["category"]].append(it)
        
        # We'll store a record of subcategory structures once proposed
        self.subcategory_trees = {}
        
        # ReAct steps:
        # 1) Summarize categories
        # 2) For each category, propose subcategories
        # 3) Place items
        # 4) Extract tags
        # 5) Possibly refine
        # ...
    
    def run(self):
        """
        Main driver method. 
        In real usage, we might do ReAct step by step with Observations.
        Here we do it in a single pass for demonstration.
        """
        # Step 1: We have categories:
        categories = list(self.items_by_cat.keys())
        print(f"Top-level categories found: {categories}")
        
        for cat in categories:
            print(f"\n--- Handling category '{cat}' ---")
            
            # 2) Propose subcategories using some sample items
            sample_items = random.sample(self.items_by_cat[cat], 
                                         min(len(self.items_by_cat[cat]), 5))
            raw_json = propose_subcategories_tool(cat, sample_items)
            print(f"Subcategory proposal for '{cat}':\n{raw_json}\n")
            
            # Parse the proposed structure
            try:
                structure = json.loads(raw_json)
            except:
                structure = {"subcategories": []}  # fallback
            self.subcategory_trees[cat] = structure
            
            # 3) Place items
            # We'll do a naive approach: for each item, guess the single best path from the structure:
            for it in self.items_by_cat[cat]:
                # (In a real system, you'd call GPT again or do pattern matching 
                #  to decide which subcategory path best fits the item.)
                # For demonstration, let's put them all in the first child or 'misc' if none.
                path = [cat]
                subcats = structure.get("subcategories", [])
                if subcats:
                    first_subcat = subcats[0]
                    path.append(first_subcat["name"])
                    # If that subcat has children, pick the first or "misc"
                    if "children" in first_subcat and first_subcat["children"]:
                        path.append(first_subcat["children"][0]["name"])
                else:
                    path.append("misc")
                
                it["category_path"] = path
                
                # 4) Extract tags
                extracted_tags = extract_tags_tool(it)
                it["tags"] = extracted_tags
                
                # Save in file system
                move_item(it, it["category_path"], base_dir=self.base_dir)
        
        print("\nDone assigning items to subcategories and extracting tags.")

##############################
# Main
##############################

def main():
    items = load_items()
    # Clean out or ensure the base directory
    base_dir = "catalog"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    agent = FileSystemReActAgent(items, base_dir=base_dir)
    agent.run()
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
