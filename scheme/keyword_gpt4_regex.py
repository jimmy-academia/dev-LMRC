import logging
import re
import openai

from .base import BaseScheme

class KeywordGPT4RegexScheme(BaseScheme):
    """
    Implements Script 3:
      1) GPT-4 generates a Python regex pattern capturing relevant documents.
      2) Applies that regex to the corpus.
    """
    def prep_task_specifics(self):
        logging.info("[KeywordGPT4RegexScheme] Using GPT-4 to generate a regex pattern.")
    
    def _generate_regex_pattern(self, query):
        prompt = (
            f"Given the search query: '{query}', produce a Python regular expression pattern "
            "that would match documents relevant to this query. Return only the regex pattern as a string."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at creating regular expressions for search."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.0
        )
        pattern = response["choices"][0]["message"]["content"].strip()
        # Optional: remove leading/trailing slashes if present
        if pattern.startswith("/") and pattern.endswith("/"):
            pattern = pattern[1:-1]
        return pattern

    def _regex_search(self, corpus, pattern):
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logging.error(f"Regex compilation error: {e}")
            return []
        
        matches = []
        for idx, doc in enumerate(corpus):
            if compiled.search(doc):
                matches.append(self.item_pool[idx]['item_id'])
        return matches

    def _get_final_candidates(self, query_text):
        pattern = self._generate_regex_pattern(query_text)
        corpus = [item['metadata'] for item in self.item_pool]
        return self._regex_search(corpus, pattern)
