import json
from cat_sample import load_category_items
from utils import set_verbose, create_llm_client, loadj


sys_expert = "You are an expert in categorizing items into a hiearchical file path."
cat_prompt = """Start with root, and use category as the first path. 

- In level names, replace space with underscores.
- Use the existing directory if possible, or branch off any level if the category should different.

Existing directory:
%s

What is the three level path of the following item?

%s


Return in the following format:
{
    "Reasoning": "....",
    "Path": "/%s/2nd-level/3rd-level/%s"
}"""



def main():
    call_llm = create_llm_client()
    item_pool = load_category_items('Food')
    file_tree = {}

    for item in item_pool:

        prompt_dict = create_prompt_dict(file_tree)

        formatted_prompt = cat_prompt % (
            json.dumps(prompt_dict, indent=2),
            str(item),
            item['category'],
            item['item_id']
        )

        print(formatted_prompt)
        input()
        
        llm_response = call_llm([sys_struct(sys_expert), user_struct(formatted_prompt)])
        parsed_response = json.loads(llm_response)
        path = parsed_response["Path"]
        
        # Split the path into components
        path_components = path.strip('/').split('/')
        
        # Update the directory structure
        current_level = file_tree
        for i in range(len(path_components) - 1):  # Skip the last component which is the item ID
            component = path_components[i]
            if component not in current_level:
                current_level[component] = {}
            current_level = current_level[component]

        # Add the item_id at the lowest level
        item_id = item['item_id']
        if item_id != path_components[-1]:
            logging.warning(f'LLM hallucinated {path_components[-1]} for {item_id}!')
        if "item_ids" not in current_level:
            current_level["item_ids"] = []
        current_level["item_ids"].append(item_id)

        print(f"Added item {item_id} to path: {path}")

    print("\nFinal Directory Structure:")
    print(json.dumps(file_tree, indent=2))
    dumpj(file_tree, "file_tree.json")

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
        
if __name__ == '__main__':
    set_verbose(1)
    main()