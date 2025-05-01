"""
Similarity-based retrieval baseline methods for LMRC.

This module serves as an entry point for three similarity search approaches:
1. Direct similarity (sim_query_item): Compare query directly to item metadata
2. LLM enhanced similarity (sim_llm_item): Enhance query with LLM first, then compare
3. Chain-of-thought enhanced similarity (sim_cot_item): Use CoT to enhance query before comparison
"""

import logging

# Import specific retrievers
from .base_retriever import SimilarityRetriever
from .llm_retriever import LLMEnhancedSimilarityRetriever
from .cot_retriever import CoTEnhancedSimilarityRetriever


def run(args, item_pool, requests):
    """
    Run the appropriate similarity-based retrieval method based on the specified app type.
    
    Args:
        args: Command-line arguments containing app type and model
        item_pool: List of items to search through
        requests: List of search requests to process
        
    Returns:
        dict: Statistics about the retrieval performance
    """
    # Select retriever class based on app type
    if args.app == 'sim_query_item':
        retriever = SimilarityRetriever(args)
        logging.info("Using direct similarity retrieval (no query enhancement)")
    elif args.app == 'sim_llm_item':
        retriever = LLMEnhancedSimilarityRetriever(args)
        logging.info("Using LLM-enhanced similarity retrieval")
    elif args.app == 'sim_cot_item':
        retriever = CoTEnhancedSimilarityRetriever(args)
        logging.info("Using chain-of-thought enhanced similarity retrieval")
    else:
        raise ValueError(f"Unknown similarity method: {args.app}")
    
    # Run the selected retriever and return stats
    stats = retriever.run(item_pool, requests)
    return stats