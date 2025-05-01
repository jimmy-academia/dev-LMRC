"""
Chain-of-thought enhanced similarity retriever class.
"""

import time
import logging
from utils import system_struct, user_struct, create_llm_client
from .base_retriever import SimilarityRetriever
from .prompt import cot_enhance_prompt, system_enhance, query_extraction_prompt, system_extract


class CoTEnhancedSimilarityRetriever(SimilarityRetriever):
    """
    Chain-of-thought enhanced query similarity retriever.
    
    This retriever uses a more elaborate LLM approach with step-by-step reasoning
    to analyze and enhance the query before embedding, leading to a more 
    comprehensive understanding of the user's intent.
    """
    
    def __init__(self, args, output_dir="app/baseline/output"):
        super().__init__(args, output_dir)
        # Initialize LLM client for query enhancement
        self.llm_client = create_llm_client(model=args.model)
        
        # Initialize a second LLM client for query extraction (could be a simpler/faster model)
        # For simplicity, we'll use the same model, but this could be changed
        self.extraction_client = create_llm_client(model=args.model)
    
    def enhance_query(self, query):
        """Enhance query using chain-of-thought reasoning with a two-step process."""
        # Step 1: Generate the CoT reasoning and enhanced query
        cot_prompt = cot_enhance_prompt.format(query=query)
        
        try:
            full_response = self.llm_client([
                system_struct(system_enhance),
                user_struct(cot_prompt)
            ])
            
            # Step 2: Extract just the enhanced query using the extraction prompt
            extraction_prompt = query_extraction_prompt.format(cot_response=full_response)
            
            extracted_query = self.extraction_client([
                system_struct(system_extract),
                user_struct(extraction_prompt)
            ]).strip()
            
            # If extraction failed or returned empty, fall back to original query
            if not extracted_query:
                logging.warning("Query extraction failed, falling back to original query")
                extracted_query = query
            
            # Log the enhancement process
            log_entry = {
                'original_query': query,
                'cot_reasoning': full_response,
                'extracted_query': extracted_query,
                'timestamp': time.time()
            }
            self.query_enhancement_log.append(log_entry)
            
            return extracted_query
            
        except Exception as e:
            logging.error(f"Error enhancing query with CoT: {e}")
            return query