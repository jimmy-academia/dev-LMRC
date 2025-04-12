from cat_sample import load_category_items
# from sample import load_sampled_data

from utils import set_verbose
    

def quick_check():
    output_pkl='cache/sample_query_item.pkl'
    item_pool, requests = load_sampled_data(output_pkl)
    print(f"Sample size: {len(item_pool)} items, {len(requests)} requests")
    print('==== first request ====')
    print(requests[0])
    print('==== first item ====')
    print(item_pool[0])


prompt = '''
You are an expert in categorizing items into a hiearchical file path. start with root, and use category as the first path, what is the three level path of the following item

%s

- In level names, replace space with underscores

Return in the following format:
{
    "Reasoning": "....",
    "Path": "/%s/2nd-level/3rd-level/%s"
}
'''

def main():
    # output_pkl='cache/sample_query_item.pkl'
    # item_pool, requests = load_sampled_data(output_pkl)
    item_pool = load_category_items('Food')
    
    count = 0
    for item in item_pool:
        print('========')
        print(prompt%(str(item), item['category'], item['item_id']))
        count += 1
        if count > 2:
            break

if __name__ == '__main__':
    set_verbose(1)

    # quick_check()
    main()