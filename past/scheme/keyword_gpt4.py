import logging
import re
import openai
from tqdm import tqdm

from .base import BaseScheme

class KeywordGPT4BasicScheme(BaseScheme):
    """
    Implements a basic GPT-4 approach:
      1) GPT-4 for comma-separated keyword generation
      2) Simple lexical search (any keyword match)
    """
    def prep_task_specifics(self):
        logging.info("[KeywordGPT4BasicScheme] Using basic GPT-4 keyword generation.")
        # e.g. load your openai.api_key if not already set in main.py
    
    def _generate_keywords(self, query):
        """
        Calls GPT-4 to produce a comma-separated list of keywords from the user query.
        """
        prompt = (
            f"Extract the most important keywords from the following query for a lexical search: '{query}'. "
            "Return a comma-separated list of keywords."
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant for generating search keywords."
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
        keywords_text = response.choices[0].message.content
        # Parse comma-separated keywords
        keywords = [word.strip() for word in keywords_text.split(",") if word.strip()]
        return keywords

    def _lexical_search(self, corpus, keywords):
        """
        Returns item_ids whose metadata contains ANY of the keywords (case-insensitive).
        """
        results = []
        for idx, doc in enumerate(tqdm(corpus, desc='lexical search enumerate corpus', ncols=88)):
            # If doc matches ANY of the keywords, add it
            if any(re.search(r'\b' + re.escape(kw) + r'\b', doc, re.IGNORECASE) for kw in keywords):
                results.append(self.item_pool[idx]['item_id'])
        return results

    def _get_final_candidates(self, query_text):
        # 1) Generate basic keywords
        keywords = self._generate_keywords(query_text)
        logging.info(f'keywords: {keywords}')
        # 2) Run lexical search
        corpus = [item['metadata'] for item in self.item_pool]
        final_item_ids = self._lexical_search(corpus, keywords)
        logging.info(f'length of final_item_ids: {len(final_item_ids)}')
        return final_item_ids
