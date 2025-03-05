# https://github.com/hyp1231/AmazonReviews2023/blob/main/amazon-c4/README.md
import json
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from collections import defaultdict

from debug import check


queries = load_dataset('McAuley-Lab/Amazon-C4')['test']

'''
{'qid': 0, 'query': "I need filters that effectively trap dust and improve the air quality in my home. It's surprising how much dust they can collect in just a few months.", 'item_id': 'B0C5QYYHTJ', 'user_id': 'AGREO2G3GTRNYOJK4CIQV2DTZLSQ', 'ori_rating': 5, 'ori_review': 'These filters work I could not believe the amount of dust in the old filter when it was replaced after 3 months.  These really trap the dust and make my home a much healthier place.'}
'''



filepath = hf_hub_download(
        repo_id='McAuley-Lab/Amazon-C4',
        filename='sampled_item_metadata_1M.jsonl',
        repo_type='dataset'
    )
item_pool = []
with open(filepath, 'r') as f:
    for line in f:
        item_pool.append(json.loads(line.strip()))

b = [item for item in item_pool if item['category'] == 'Care']
# for item in item_pool:
    # a

'''
{'item_id': 'B0778XR2QM', 'category': 'Care', 'metadata': 'Supergoop! Super Power Sunscreen Mousse SPF 50, 7.1 Fl Oz. Product Description Kids, moms, and savvy sun-seekers will flip for this whip! Formulated with nourishing Shea butter and antioxidant packed Blue Sea Kale, this one-of-a kind mousse formula is making sunscreen super FUN! The refreshing light essence of cucumber and citrus has become an instant hit at Super goop! HQ where we’ve been known to apply gobs of it just for the uplifting scent. Water resistant for up to 80 minutes too! Brand Story Supergoop! is the first and only prestige skincare brand completely dedicated to sun protection. Supergoop! has Super Broad Spectrum protection, which means it protects skin from UVA rays, UVB rays and IRA rays.'}

'''

check()

id2item = {}
for item in item_pool:
    id2item[item['item_id']] = item

for query in queries:
    print('===')
    print(query['query'])
    print('===')
    print(query['ori_review'])
    print('===')
    print(id2item[query['item_id']])
    input()



category_itemids = defaultdict(list)
for item in item_pool:
    category_itemids[item['category']].append(item['item_id'])

print(category_itemids.keys())
length = []
for key in category_itemids:
    length.append(len(category_itemids[key]))

print(length)

categories_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023", 
    filename="all_categories.txt",
    repo_type='dataset'
)

# Download asin2category.json
asin2category_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023", 
    filename="asin2category.json",
    repo_type='dataset'
)

with open(categories_path, "r") as f:
    all_categories = [line.strip() for line in f if line.strip()]
print(all_categories)

# check()

with open(asin2category_path, "r") as f:
    asin_to_category = json.load(f)
print(list(asin_to_category.items())[:5])



reviews_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                               "raw_review_All_Beauty", 
                               split="full", 
                               trust_remote_code=True, 
                               streaming=True)


# dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
# print(dataset["full"][0])

print(reviews_dataset)
check()

'''
{'rating': 5.0, 'title': 'Such a lovely scent but not overpowering.', 'text': "This spray is really nice. It smells really good, goes on really fine, and does the trick. I will say it feels like you need a lot of it though to get the texture I want. I have a lot of hair, medium thickness. I am comparing to other brands with yucky chemicals so I'm gonna stick with this. Try it!", 'images': [], 'asin': 'B00YQ6X8EO', 'parent_asin': 'B00YQ6X8EO', 'user_id': 'AGKHLEW2SOWHNMFQIJGBECAF7INQ', 'timestamp': 1588687728923, 'helpful_vote': 0, 'verified_purchase': True}
'''

metadata_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                                "raw_meta_All_Beauty", 
                                split="full", 
                                trust_remote_code=True)

next(iter(metadata_dataset))

'''
{'main_category': 'All Beauty', 'title': 'Howard LC0008 Leather Conditioner, 8-Ounce (4-Pack)', 'average_rating': 4.8, 'rating_number': 10, 'features': [], 'description': [], 'price': 'None', 'images': {'hi_res': [None, 'https://m.media-amazon.com/images/I/71i77AuI9xL._SL1500_.jpg'], 'large': ['https://m.media-amazon.com/images/I/41qfjSfqNyL.jpg', 'https://m.media-amazon.com/images/I/41w2yznfuZL.jpg'], 'thumb': ['https://m.media-amazon.com/images/I/41qfjSfqNyL._SS40_.jpg', 'https://m.media-amazon.com/images/I/41w2yznfuZL._SS40_.jpg'], 'variant': ['MAIN', 'PT01']}, 'videos': {'title': [], 'url': [], 'user_id': []}, 'store': 'Howard Products', 'categories': [], 'details': '{"Package Dimensions": "7.1 x 5.5 x 3 inches; 2.38 Pounds", "UPC": "617390882781"}', 'parent_asin': 'B01CUPMQZE', 'bought_together': None, 'subtitle': None, 'author': None}
'''



# check()