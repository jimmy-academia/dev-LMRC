import logging
import re
import openai
from huggingface_hub import hf_hub_download
from collections import defaultdict
from .base import BaseScheme

from debug import check

class TaxonomyGPT4Scheme(BaseScheme):
    """
    Implements a basic GPT-4 approach:
      1) GPT-4 for comma-separated keyword generation
      2) Simple lexical search (any keyword match)
    """
    def prep_task_specifics(self):
        logging.info("[TaxonomyGPT4Scheme] Using basic GPT-4 keyword generation.")
        # e.g. load your openai.api_key if not already set in main.py

        categories_path = hf_hub_download(
            repo_id="McAuley-Lab/Amazon-Reviews-2023", 
            filename="all_categories.txt",
            repo_type='dataset'
        )
        with open(categories_path, "r") as f:
            self.all_categories = [line.strip() for line in f if line.strip()]

        # prepare all category id
        self.category_ids = defaultdict(list)
        self.id2cat = {}
        for item in self.item_pool:
            self.category_ids[item['category']].append(item['item_id'])
            self.id2cat[item['item_id']] = item['category']

    def _determine_category(self, query):
        """
        Calls GPT-4 to produce a comma-separated list of keywords from the user query.
        """
        prompt = (
            f"Determine the category of the product required based on the following query: '{query}'. "
            f"All categories include: {self.all_categories}."
            "Select one category from the list."
        )

        logging.info(f'asking gpt4 with prompt {prompt}')
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for product search."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=50,
            temperature=0.0,
        )

        # Extract the GPT-4 output
        category_text = response.choices[0].message.content
        # Parse comma-separated keywords
        # keywords = [word.strip() for word in keywords_text.split(",") if word.strip()]
        category = category_text.strip()
        logging.info(f'located category: {category}')
        return category

    def _in_category_ids(self, category):

        ## return the ids in each category
        if category in self.category_ids:
            return self.category_ids[category]
        else:
            return []

    def _lexical_search(self, corpus, keywords):
        """
        Returns item_ids whose metadata contains ANY of the keywords (case-insensitive).
        """
        results = []
        for idx, doc in enumerate(corpus):
            # If doc matches ANY of the keywords, add it
            if any(re.search(r'\b' + re.escape(kw) + r'\b', doc, re.IGNORECASE) for kw in keywords):
                results.append(self.item_pool[idx]['item_id'])
        return results

    def _get_final_candidates(self, query_text):
        # 1) Generate basic keywords
        category = self._determine_category(query_text)

        logging.info(f'gpt suggest category: {category}')
        logging.info(f'ground truth: {self.id2cat[self.ground_truth]}')

        if category != self.id2cat[self.ground_truth]:
            return []
            
        final_item_ids = self._in_category_ids(category)

        # 2) Run lexical search
        # corpus = [item['metadata'] for item in self.item_pool]
        # final_item_ids = self._lexical_search(corpus, keywords)
        return final_item_ids
