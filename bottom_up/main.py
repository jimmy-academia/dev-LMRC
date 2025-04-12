import json
from cat_sample import load_category_items
from utils import set_verbose


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
    item_pool = load_category_items('Food')
    directory_structure = {}

    for item in item_pool:
        formatted_prompt = cat_prompt % (
            json.dumps(directory_structure, indent=2),
            str(item),
            item['category'],
            item['item_id']
        )
        
        print(formatted_prompt)
        input()
        

        # llm_response = call_llm([sys_struct(sys_expert), user_struct(formatted_prompt)])
        llm_response = """{ "Reasoning": "The item is a natural food coloring used specifically for baking and cake decorating. Since it belongs to the 'Food' category and is primarily a baking ingredient, the second level should be 'Baking_Ingredients'. The third level should reflect its specific function, which is 'Food_Coloring'.", "Path": "/Food/Baking_Ingredients/Food_Coloring/B07T5WY5T9" }"""

        parsed_response = json.loads(llm_response)
        path = parsed_response["Path"]
        
        # Split the path into components
        path_components = path.strip('/').split('/')
        
        # Update the directory structure
        current_level = directory_structure
        for i in range(len(path_components) - 1):  # Skip the last component which is the item ID
            component = path_components[i]
            if component not in current_level:
                current_level[component] = {}
            current_level = current_level[component]

        print(f"Added item {item['item_id']} to path: {path}")
        print("\nFinal Directory Structure:")
        print(json.dumps(directory_structure, indent=2))
        
if __name__ == '__main__':
    set_verbose(1)
    main()