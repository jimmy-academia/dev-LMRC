import logging
import os
import json

from utils import dumpj
from debug import check
from .base import BaseScheme

class KeywordGPT4Scheme(BaseScheme):
    """
    Scheme that uses GPT-4 to generate keywords, then does a naive keyword-based search.
    """
    def __init__(self, args, task_data):
        super().__init__(args, task_data)
        # Precompute tokenized metadata for naive matching
        self.prepared_metadata = []
        for item in self.item_pool:
            tokens = item['metadata'].lower().split()
            self.prepared_metadata.append(tokens)

    def prep_task_specifics(self):
        logging.info("[KeywordGPT4Scheme] Ready to generate keywords & search items.")

    def _generate_keywords(self, query_text):
        """
        Child-specific GPT call to get keywords. 
        Placeholder logic: Just take first 3 words as 'keywords'.
        """
        return query_text.lower().split()[:3]

    def _keyword_search(self, keywords, top_k=50):
        """
        A naive approach: Score each item by the number of keyword matches in metadata.
        Return the top_k item indices (sorted by match_count desc).
        """
        scores = []
        for idx, tokens in enumerate(self.prepared_metadata):
            match_count = sum(kw in tokens for kw in keywords)
            scores.append((match_count, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scores[:top_k]]
        return top_indices

    def _get_final_candidates(self, query_text):
        """
        Generates keywords, then does naive search, returns final item_ids.
        """
        keywords = self._generate_keywords(query_text)
        top_indices = self._keyword_search(keywords, top_k=50)
        # Convert indices -> item_ids
        final_item_ids = [self.item_pool[i]['item_id'] for i in top_indices]
        return final_item_ids
