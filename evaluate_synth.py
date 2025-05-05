#!/usr/bin/env python3
"""
Evaluation Script for Baseline Methods on Synthetic Data

Runs all three baseline similarity retrieval methods on synthetic product data
and compares their performance with complex ranking metrics.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import numpy as np
import time
from tqdm import tqdm

from utils import set_seeds, set_verbose, loadj, dumpj, ensure_dir, create_llm_client
from app.baseline.base_retriever import SimilarityRetriever
from app.baseline.llm_retriever import LLMEnhancedSimilarityRetriever
from app.baseline.cot_retriever import CoTEnhancedSimilarityRetriever

# Configure logging
set_verbose(1)

class SyntheticDataRetriever:
    """Base class for synthetic data retrieval methods."""
    
    def __init__(self, method_name, output_dir="app/baseline/output/synth"):
        self.method_name = f"synth_{method_name}"
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)
        ensure_dir(self.output_dir / "log")
        
    def load_synthetic_data(self, products_path, requests_path):
        """Load synthetic products and requests."""
        with open(products_path, 'r') as f:
            self.products = json.load(f)
            
        with open(requests_path, 'r') as f:
            self.requests = json.load(f)
            
        # Create item_pool in the format expected by the base retriever
        self.item_pool = []
        for product in self.products:
            item = {
                "item_id": product["id"],
                "metadata": product["description"],  # Use only the description for embedding
                "summary": f"{product['tier']} {product['product_type']}",
                "product_type": product["product_type"],
                "tier": product["tier"]
            }
            self.item_pool.append(item)
            
        logging.info(f"Loaded {len(self.item_pool)} products and {len(self.requests)} requests")
        return self.item_pool, self.requests
        
    def calculate_ranking_metrics(self, retrieved_ids, ground_truth_ranking, target_id):
        """Calculate ranking metrics for a single request."""
        metrics = {}
        
        # Create relevance dict (1 for items in ground truth, 0 otherwise)
        relevance = {item_id: 1 for item_id in ground_truth_ranking}
        
        # Calculate Precision@k and NDCG@k
        for k in [5, 10, 20]:
            if k > len(retrieved_ids):
                continue
                
            # Precision@k
            relevant_in_topk = sum(1 for item_id in retrieved_ids[:k] if item_id in relevance)
            metrics[f"precision@{k}"] = relevant_in_topk / k
            
            # NDCG@k
            dcg = sum((1 if item_id in relevance else 0) / np.log2(i + 2) 
                      for i, item_id in enumerate(retrieved_ids[:k]))
            
            # Ideal ranking would have all relevant items first
            idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(ground_truth_ranking))))
            
            metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0
        
        # MRR for target item
        if target_id in retrieved_ids:
            rank = retrieved_ids.index(target_id) + 1
            metrics["mrr"] = 1.0 / rank
        else:
            metrics["mrr"] = 0.0
            
        # Recall@k
        ground_truth_set = set(ground_truth_ranking)
        for k in [10, 20, 50]:
            if k > len(retrieved_ids):
                continue
                
            retrieved_set = set(retrieved_ids[:k])
            
            if ground_truth_set:
                recall = len(retrieved_set.intersection(ground_truth_set)) / len(ground_truth_set)
                metrics[f"recall@{k}"] = recall
            else:
                metrics[f"recall@{k}"] = 0.0
        
        return metrics


class SynthBasicRetriever(SimilarityRetriever, SyntheticDataRetriever):
    """Basic similarity retriever for synthetic data."""
    
    def __init__(self, args, output_dir="app/baseline/output/synth"):
        SimilarityRetriever.__init__(self, args, output_dir)
        SyntheticDataRetriever.__init__(self, args.app, output_dir)
        
        # Override the method to get embedding path to use a consistent name
        self.get_item_embedding_path_orig = self.get_item_embedding_path
        self.get_item_embedding_path = lambda item_count: Path(self.output_dir) / "embeddings" / f"synth_embeddings_{item_count}.pkl"
    
    def process_requests(self, item_pool, requests, item_embeddings):
        """Process all requests and evaluate results with different metrics."""
        results = []
        
        for request in tqdm(requests, desc=f"Processing requests for {self.method_name}"):
            query = request['query']
            target_item_id = request['target_product_id']
            ground_truth_ranking = request['ranked_list']
            
            # Retrieve all items, sorted by similarity
            retrieved_items, enhanced_query = self.retrieve(item_pool, query, item_embeddings)
            
            # Get the retrieved IDs in order
            retrieved_ids = [item['item_id'] for item in retrieved_items]
            
            # Calculate rank of target item
            target_rank = next((i for i, item_id in enumerate(retrieved_ids) 
                               if item_id == target_item_id), -1)
            
            # Calculate metrics
            metrics = self.calculate_ranking_metrics(retrieved_ids, ground_truth_ranking, target_item_id)
            
            # Log the result
            result = {
                'query': query,
                'enhanced_query': enhanced_query if enhanced_query != query else None,
                'target_item_id': target_item_id,
                'target_rank': target_rank,
                'metrics': metrics,
                'top_retrieved': retrieved_ids[:10]
            }
            results.append(result)
            
        # Calculate average metrics
        avg_metrics = {}
        all_metric_keys = set()
        for r in results:
            all_metric_keys.update(r['metrics'].keys())
            
        for key in all_metric_keys:
            values = [r['metrics'].get(key, 0) for r in results if key in r['metrics']]
            avg_metrics[key] = np.mean(values) if values else 0
            
        # Calculate hit rates for target item
        for k in [1, 5, 10, 20]:
            hits = sum(1 for r in results if r['target_rank'] >= 0 and r['target_rank'] < k)
            avg_metrics[f"target_hit_rate@{k}"] = hits / len(results) if results else 0
        
        return results, avg_metrics
    
    def run(self, products_path, requests_path):
        """Run the retrieval method on synthetic data."""
        start_time = time.time()
        
        # Load synthetic data
        item_pool, requests = self.load_synthetic_data(products_path, requests_path)
        
        # Compute embeddings for all items (with caching)
        item_embeddings = self.compute_item_embeddings(item_pool)
        
        # Process all requests
        results, avg_metrics = self.process_requests(item_pool, requests, item_embeddings)
        
        # Calculate stats
        elapsed_time = time.time() - start_time
        stats = {
            'method': self.method_name,
            'total_requests': len(requests),
            'avg_metrics': avg_metrics,
            'embedding_count': self.embedding_count,
            'embedding_tokens': self.embedding_tokens,
            'embedding_cost': self.embedding_cost,
            'elapsed_time': elapsed_time
        }
        
        # Save results
        results_path = self.output_dir / f"{self.method_name}_results.json"
        stats_path = self.output_dir / f"{self.method_name}_stats.json"
        
        dumpj(results, results_path)
        dumpj(stats, stats_path)
        
        return stats, avg_metrics


class SynthLLMRetriever(LLMEnhancedSimilarityRetriever, SyntheticDataRetriever):
    """LLM-enhanced retriever for synthetic data."""
    
    def __init__(self, args, output_dir="app/baseline/output/synth"):
        LLMEnhancedSimilarityRetriever.__init__(self, args, output_dir)
        SyntheticDataRetriever.__init__(self, args.app, output_dir)
        
        # Override the method to get embedding path to use a consistent name
        self.get_item_embedding_path_orig = self.get_item_embedding_path
        self.get_item_embedding_path = lambda item_count: Path(self.output_dir) / "embeddings" / f"synth_embeddings_{item_count}.pkl"
    
    process_requests = SynthBasicRetriever.process_requests
    run = SynthBasicRetriever.run


class SynthCoTRetriever(CoTEnhancedSimilarityRetriever, SyntheticDataRetriever):
    """Chain-of-thought enhanced retriever for synthetic data."""
    
    def __init__(self, args, output_dir="app/baseline/output/synth"):
        CoTEnhancedSimilarityRetriever.__init__(self, args, output_dir)
        SyntheticDataRetriever.__init__(self, args.app, output_dir)
        
        # Override the method to get embedding path to use a consistent name
        self.get_item_embedding_path_orig = self.get_item_embedding_path
        self.get_item_embedding_path = lambda item_count: Path(self.output_dir) / "embeddings" / f"synth_embeddings_{item_count}.pkl"
    
    process_requests = SynthBasicRetriever.process_requests
    run = SynthBasicRetriever.run


def print_metrics_table(all_metrics):
    """Print a table comparing metrics from all methods."""
    method_names = []
    metrics_data = []
    costs = []
    
    # Extract data from all_metrics correctly
    for method, metrics, cost in all_metrics:
        method_names.append(method)
        metrics_data.append(metrics)
        costs.append(cost)
    
    metric_keys = ['target_hit_rate@1', 'target_hit_rate@5', 'target_hit_rate@10', 
                  'precision@10', 'ndcg@10', 'recall@10', 'mrr']
    
    header_names = ['Hit@1', 'Hit@5', 'Hit@10', 'Prec@10', 'NDCG@10', 'Rec@10', 'MRR']
    
    # Print header
    print("\n" + "="*80)
    print("COMPARISON OF BASELINE METHODS ON SYNTHETIC DATA")
    print("="*80)
    
    # Method names row
    print(f"{'Metric':<10}", end="")
    for name in method_names:
        print(f" | {name:<15}", end="")
    print()
    
    # Divider
    print("-"*80)
    
    # Metrics rows
    for i, key in enumerate(metric_keys):
        print(f"{header_names[i]:<10}", end="")
        for metrics in metrics_data:
            value = metrics.get(key, 0)
            print(f" | {value:.4f}           ", end="")
        print()
    
    # Print computation costs
    print("\nComputation Costs:")
    for i, method in enumerate(method_names):
        print(f"  {method}: ${costs[i]:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline methods on synthetic data')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--data-dir', type=str, default='cache/synth_products')
    
    args = parser.parse_args()
    set_seeds(args.seed)
    
    # Paths to synthetic data
    products_path = Path(args.data_dir) / "products.json"
    requests_path = Path(args.data_dir) / "requests.json"
    
    # Check if files exist
    if not products_path.exists() or not requests_path.exists():
        print(f"Synthetic data not found at {args.data_dir}")
        print("Please run 'python -m data.synth' first to generate synthetic data.")
        return
    
    # Run all three methods
    methods = ['sim_query_item', 'sim_llm_item', 'sim_cot_item']
    all_metrics = []
    
    print(f"Evaluating {len(methods)} baseline methods on synthetic data...")
    
    for method in methods:
        print(f"\n=== Running {method} ===")
        
        # Set the method in args
        args.app = method
        
        # Create the appropriate retriever
        if method == 'sim_query_item':
            retriever = SynthBasicRetriever(args)
        elif method == 'sim_llm_item':
            retriever = SynthLLMRetriever(args)
        elif method == 'sim_cot_item':
            retriever = SynthCoTRetriever(args)
        
        stats, avg_metrics = retriever.run(products_path, requests_path)
        all_metrics.append((method, avg_metrics, stats['embedding_cost']))
    
    # Print comparison table
    print_metrics_table(all_metrics)
    
    print("\nEvaluation complete. Results saved to: app/baseline/output/synth/")


if __name__ == "__main__":
    main()