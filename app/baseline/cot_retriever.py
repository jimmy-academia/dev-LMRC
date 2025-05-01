"""
Chain-of-thought enhanced similarity retriever class.
"""

import time
import logging
from utils import system_struct, user_struct, create_llm_client
from .base_retriever import SimilarityRetriever
from .prompt import cot_enhance_prompt, system_enhance


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
    
    def enhance_query(self, query):
        """Enhance query using chain-of-thought reasoning."""
        prompt = cot_enhance_prompt.format(query=query)
        
        try:
            response = self.llm_client([
                system_struct(system_enhance),
                user_struct(prompt)
            ])
            
            # Extract the enhanced query from the CoT response
            lines = response.strip().split('\n')
            enhanced = None
            
            # Look for lines that appear to be the enhanced query
            for i, line in enumerate(lines):
                if line.startswith("Enhanced query:") or "enhanced query" in line.lower():
                    if i + 1 < len(lines):
                        enhanced = lines[i + 1].strip()
                        break
            
            # If we didn't find an explicit label, take the last non-empty line
            if not enhanced:
                for line in reversed(lines):
                    if line.strip() and not line.lower().startswith(("based on", "think", "analysis")):
                        enhanced = line.strip()
                        break
            
            # Fallback to original query if parsing fails
            if not enhanced:
                enhanced = query
            
            # Log the enhancement
            log_entry = {
                'original_query': query,
                'cot_response': response,
                'enhanced_query': enhanced,
                'timestamp': time.time()
            }
            self.query_enhancement_log.append(log_entry)
            
            return enhanced
        except Exception as e:
            logging.error(f"Error enhancing query with CoT: {e}")
            return query