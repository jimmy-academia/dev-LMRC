"""
Direct similarity retriever class with no query enhancement.
"""

from .base_retriever import SimilarityRetriever


class DirectSimilarityRetriever(SimilarityRetriever):
    """
    Direct similarity between query and items (no enhancement).
    
    This retriever simply compares the original query text directly to the item 
    metadata and summary using vector embeddings, without any query enhancement.
    """
    
    def __init__(self, args, output_dir="app/baseline/output"):
        super().__init__(args, output_dir)
        # No additional initialization needed for direct similarity