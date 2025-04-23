import logging
import re
import openai

from .base import BaseScheme

class KeywordGPT4RefinedScheme(BaseScheme):
    """
    Implements Script 2:
      1) GPT-4 for initial keyword generation.
      2) Lexical search.
      3) Summarize or re-check the search results.
      4) GPT-4 to refine keywords based on those results.
      5) Final lexical search with refined keywords.
    """
    def prep_task_specifics(self):
        logging.info("[KeywordGPT4RefinedScheme] Using GPT-4 with refinement steps.")
    
    def _generate_keywords(self, query):
        prompt = (
            f"Extract the most important keywords from the following query for a lexical search: '{query}'. "
            "Return a comma-separated list of keywords."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert assistant for generating search keywords."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.0,
        )
        keywords_text = response["choices"][0]["message"]["content"]
        return [word.strip() for word in keywords_text.split(",") if word.strip()]

    def _adjust_keywords(self, query, initial_keywords, search_results):
        # Summarize search results
        # (Here, we simply join all doc strings, but you can do something more advanced.)
        results_summary = " ".join(doc for _, doc in search_results)
        prompt = (
            f"Original query: '{query}'\n"
            f"Initial keywords: {', '.join(initial_keywords)}\n"
            f"Initial search results: '{results_summary}'\n"
            "Please refine or adjust the keywords to better capture the query intent and relevant content. "
            "Return a comma-separated list of improved keywords."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert search assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.0
        )
        refined_text = response["choices"][0]["message"]["content"]
        return [word.strip() for word in refined_text.split(",") if word.strip()]

    def _lexical_search_docs(self, corpus, keywords):
        """
        Returns list of (idx, doc) for matching docs. We keep them in (idx, doc) form
        so we can feed them back into GPT for summarization if needed.
        """
        results = []
        for idx, doc in enumerate(corpus):
            if any(re.search(r'\b' + re.escape(kw) + r'\b', doc, re.IGNORECASE) for kw in keywords):
                results.append((idx, doc))
        return results

    def _lexical_search_ids(self, corpus, keywords):
        """
        Returns item_ids for matching docs.
        """
        results = []
        for idx, doc in enumerate(corpus):
            if any(re.search(r'\b' + re.escape(kw) + r'\b', doc, re.IGNORECASE) for kw in keywords):
                results.append(self.item_pool[idx]['item_id'])
        return results

    def _get_final_candidates(self, query_text):
        # 1) Generate initial keywords
        initial_keywords = self._generate_keywords(query_text)

        # 2) Perform initial search
        corpus = [item['metadata'] for item in self.item_pool]
        initial_results = self._lexical_search_docs(corpus, initial_keywords)

        # 3) Refine keywords based on initial results
        refined_keywords = self._adjust_keywords(query_text, initial_keywords, initial_results)

        # 4) Final search with refined keywords
        final_item_ids = self._lexical_search_ids(corpus, refined_keywords)
        return final_item_ids
