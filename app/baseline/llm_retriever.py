"""
LLM-enhanced similarity retriever class.
"""

import time
import logging
from utils import system_struct, user_struct, create_llm_client
from .base_retriever import SimilarityRetriever
from .prompt import llm_enhance_prompt, system_enhance


class LLMEnhancedSimilarityRetriever(SimilarityRetriever):
    """
    LLM-enhanced query similarity retriever.
    
    This retriever uses an LLM to enhance the query before embedding,
    adding relevant keywords, clarifying ambiguous terms, and
    including likely synonyms to improve retrieval performance.
    """
    
    def __init__(self, args, output_dir="app/baseline/output"):
        super().__init__(args, output_dir)
        # Initialize LLM client for query enhancement
        self.llm_client = create_llm_client(model=args.model)
    
    def enhance_query(self, query):
        """Enhance query using LLM."""
        prompt = llm_enhance_prompt.format(query=query)
        
        try:
            enhanced = self.llm_client([
                system_struct(system_enhance),
                user_struct(prompt)
            ])
            
            # Log the enhancement
            log_entry = {
                'original_query': query,
                'enhanced_query': enhanced,
                'timestamp': time.time()
            }
            self.query_enhancement_log.append(log_entry)
            
            return enhanced
        except Exception as e:
            logging.error(f"Error enhancing query: {e}")
            return query