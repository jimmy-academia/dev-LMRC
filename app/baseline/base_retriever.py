"""
Base retriever class for similarity-based retrieval methods.
"""

import time
import logging
import json
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from utils import ensure_dir, loadj, dumpj


class SimilarityRetriever:
    """Base class for similarity-based retrieval methods."""
    
    def __init__(self, args, output_dir="app/baseline/output"):
        self.args = args
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.output_dir / "log")
        ensure_dir(self.output_dir / "embeddings")
        
        # Initialize OpenAI client for embeddings
        api_key = self._read_api_key()
        self.openai_client = OpenAI(api_key=api_key)
        
        # Usage tracking
        self.embedding_count = 0
        self.embedding_tokens = 0
        self.embedding_cost = 0
        self.query_enhancement_log = []
        
        # Method name - used for cache file naming
        self.method_name = args.app
    
    def _read_api_key(self, keypath=".openaikey"):
        """Read OpenAI API key from file."""
        with open(keypath, 'r') as f:
            return f.read().strip()
    
    def get_embeddings(self, texts):
        """Get embeddings using OpenAI's text-embedding-3-large model."""
        if not isinstance(texts, list):
            texts = [texts]
            
        self.embedding_count += len(texts)
        
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model="text-embedding-3-large"
            )
            # Extract embedding vectors
            embeddings = [item.embedding for item in response.data]
            
            # Track token usage
            self.embedding_tokens += response.usage.total_tokens
            # Cost calculation: $0.13 per 1M tokens
            self.embedding_cost += (response.usage.total_tokens / 1_000_000) * 0.13
            
            return embeddings[0] if len(texts) == 1 else embeddings
        except Exception as e:
            logging.error(f"Error getting OpenAI embeddings: {e}")
            # Return zero vector as fallback
            return [0.0] * 3072  # Dimension for text-embedding-3-large
    
    def enhance_query(self, query):
        """Base method for query enhancement - no enhancement by default."""
        return query
    
    def get_item_embedding_path(self, item_count):
        """Get path for cached item embeddings based on method name."""
        return self.output_dir / "embeddings" / f"item_embeddings_{self.method_name}_{item_count}.pkl"
    
    def get_embedding_cost_path(self, item_count):
        """Get path for embedding cost record."""
        return self.output_dir / "embeddings" / f"embedding_cost_{self.method_name}_{item_count}.json"
    
    def compute_item_embeddings(self, item_pool):
        """Compute embeddings for all items in the pool, with caching."""
        item_count = len(item_pool)
        embedding_path = self.get_item_embedding_path(item_count)
        cost_path = self.get_embedding_cost_path(item_count)
        
        # Check if embeddings already exist
        if embedding_path.exists():
            logging.info(f"Loading cached item embeddings from {embedding_path}")
            with open(embedding_path, 'rb') as f:
                all_embeddings = pickle.load(f)
            
            # Load previous cost if available
            if cost_path.exists():
                with open(cost_path, 'r') as f:
                    cost_data = json.load(f)
                    self.embedding_tokens = cost_data.get('embedding_tokens', 0)
                    self.embedding_cost = cost_data.get('embedding_cost', 0)
                    logging.info(f"Loaded previous embedding cost: ${self.embedding_cost:.4f} for {self.embedding_tokens} tokens")
            
            return all_embeddings
        
        # If not cached, compute embeddings
        logging.info(f"Computing embeddings for {item_count} items")
        item_texts = []
        for item in item_pool:
            # Combine metadata and summary if available
            text = item.get('metadata', '')
            if 'summary' in item:
                text = f"{text} {item['summary']}"
            item_texts.append(text)
        
        # Track initial cost
        initial_tokens = self.embedding_tokens
        initial_cost = self.embedding_cost
        
        # Process in batches to avoid rate limits
        batch_size = 50
        all_embeddings = []
        
        for i in tqdm(range(0, len(item_texts), batch_size), desc="Computing item embeddings"):
            batch = item_texts[i:i+batch_size]
            batch_embeddings = self.get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)
            time.sleep(0.1)  # Brief pause to avoid rate limits
        
        # Save embeddings and cost data
        with open(embedding_path, 'wb') as f:
            pickle.dump(all_embeddings, f)
        
        # Calculate and save the embedding cost
        embedding_tokens_used = self.embedding_tokens - initial_tokens
        embedding_cost_incurred = self.embedding_cost - initial_cost
        
        cost_data = {
            'embedding_tokens': embedding_tokens_used,
            'embedding_cost': embedding_cost_incurred,
            'timestamp': time.time(),
            'model': 'text-embedding-3-large',
            'item_count': item_count
        }
        
        with open(cost_path, 'w') as f:
            json.dump(cost_data, f, indent=2)
        
        logging.info(f"Saved {item_count} item embeddings and cost data (${embedding_cost_incurred:.4f})")
        return all_embeddings
    
    def compute_similarities(self, query_embedding, item_embeddings):
        """Compute cosine similarity between query and all items."""
        # Convert to numpy arrays for efficient computation
        query_np = np.array(query_embedding)
        items_np = np.array(item_embeddings)
        
        # Normalize vectors
        query_norm = query_np / np.linalg.norm(query_np)
        items_norm = items_np / np.linalg.norm(items_np, axis=1, keepdims=True)
        
        # Compute cosine similarities
        similarities = np.dot(items_norm, query_norm)
        
        return similarities
    
    def retrieve(self, item_pool, query, item_embeddings):
        """
        Retrieve all items ranked by similarity to a query.
        
        Returns all items with their similarity scores, sorted from highest to lowest.
        """
        # Enhance query if needed (implemented in subclasses)
        enhanced_query = self.enhance_query(query)
        
        # Get query embedding
        query_embedding = self.get_embeddings(enhanced_query)
        
        # Compute similarities
        similarities = self.compute_similarities(query_embedding, item_embeddings)
        
        # Get all indices, sorted by similarity
        sorted_indices = np.argsort(-similarities)
        
        # Return all items with scores
        results = []
        for idx in sorted_indices:
            results.append({
                'item_id': item_pool[idx]['item_id'],
                'score': float(similarities[idx]),
                'metadata': item_pool[idx].get('metadata', ''),
                'summary': item_pool[idx].get('summary', '')
            })
        
        return results, enhanced_query
    
    def process_requests(self, item_pool, requests, item_embeddings):
        """Process all requests and evaluate results."""
        results = []
        ranks = []
        
        for request in tqdm(requests, desc="Processing requests"):
            query = request['query']
            target_item_id = request['item_id']
            
            # Retrieve all items, sorted by similarity
            retrieved_items, enhanced_query = self.retrieve(item_pool, query, item_embeddings)
            
            # Find rank of target item
            target_rank = next((i for i, item in enumerate(retrieved_items) 
                               if item['item_id'] == target_item_id), -1)
            ranks.append(target_rank)
            
            # Log the result
            result = {
                'query': query,
                'enhanced_query': enhanced_query if enhanced_query != query else None,
                'target_item_id': target_item_id,
                'target_rank': target_rank,
                'top_retrieved': retrieved_items[:3]  # Log only top 3 for brevity
            }
            results.append(result)
            
            # Print progress
            print(f"Query: {query}")
            if enhanced_query != query:
                print(f"Enhanced: {enhanced_query}")
            print(f"Target: {target_item_id}")
            print(f"Rank: {target_rank if target_rank >= 0 else 'Not found'}")
            print("-" * 50)
        
        # Calculate hit rates for different k values
        hit_rates = {}
        for k in [1, 3, 5, 10, 20, 50, 100]:
            hits = sum(1 for rank in ranks if rank >= 0 and rank < k)
            hit_rates[f"hit_rate@{k}"] = hits / len(requests) if requests else 0
        
        return results, ranks, hit_rates
    
    def run(self, item_pool, requests):
        """Run the retrieval method on the given item pool and requests."""
        start_time = time.time()
        
        # Compute embeddings for all items (with caching)
        item_embeddings = self.compute_item_embeddings(item_pool)
        
        # Process all requests
        results, ranks, hit_rates = self.process_requests(item_pool, requests, item_embeddings)
        
        # Calculate stats
        elapsed_time = time.time() - start_time
        stats = {
            'method': self.method_name,
            'total_requests': len(requests),
            'hit_rates': hit_rates,
            'embedding_count': self.embedding_count,
            'embedding_tokens': self.embedding_tokens,
            'embedding_cost': self.embedding_cost,
            'elapsed_time': elapsed_time,
            'rank_distribution': {
                'mean': np.mean(ranks) if ranks else float('nan'),
                'median': np.median(ranks) if ranks else float('nan'),
                'min': min(ranks) if ranks else float('nan'),
                'max': max(ranks) if ranks else float('nan')
            }
        }
        
        # Add LLM usage if applicable
        if hasattr(self, 'llm_client') and self.llm_client:
            llm_usage = self.llm_client.get_usage()
            stats['llm_usage'] = llm_usage
        
        # Save results
        base_name = f"{self.method_name}_{len(item_pool)}"
        results_path = self.output_dir / f"{base_name}_results.json"
        stats_path = self.output_dir / f"{base_name}_stats.json"
        log_path = self.output_dir / "log" / f"{base_name}_log.json"
        ranks_path = self.output_dir / f"{base_name}_ranks.json"
        
        dumpj(results, results_path)
        dumpj(stats, stats_path)
        dumpj(ranks, ranks_path)
        
        if self.query_enhancement_log:
            dumpj(self.query_enhancement_log, log_path)
        
        # Log summary
        logging.info(f"Method: {self.method_name}")
        for k, rate in hit_rates.items():
            logging.info(f"{k}: {rate:.4f}")
        logging.info(f"Embedding tokens: {self.embedding_tokens}, cost: ${self.embedding_cost:.4f}")
        logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        logging.info(f"Results saved to {results_path}")
        
        return stats