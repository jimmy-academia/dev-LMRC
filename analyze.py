#!/usr/bin/env python3
"""
Baseline Results Analysis Script

This script analyzes the results from the baseline similarity retrieval methods, printing:
1. Ground truth item rank for each request
2. Enhanced query (if applicable)
3. Ground truth item descriptions (metadata and summary)

Usage: python analyze_baseline_results.py [--method METHOD] [--item_count ITEM_COUNT]
"""

import argparse
import json
import os
from pathlib import Path


def load_json(filepath):
    """Load a JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {filepath}: {e}")
        return None


def get_item_by_id(item_pool, item_id):
    """Find an item in the pool by its ID."""
    for item in item_pool:
        if item['item_id'] == item_id:
            return item
    return None


def analyze_results(method, item_count):
    """Analyze and print results for a specific baseline method."""
    # Define paths
    base_dir = Path('app/baseline/output')
    results_file = base_dir / f"{method}_{item_count}_results.json"
    stats_file = base_dir / f"{method}_{item_count}_stats.json"
    
    # Load results and stats
    results = load_json(results_file)
    stats = load_json(stats_file)
    
    if not results or not stats:
        print(f"Could not find results for {method} with {item_count} items")
        return
    
    # Load item pool from cached file if possible (or load from original source)
    item_pool_file = Path(f'cache/subsample_{item_count}.pkl')
    if item_pool_file.exists():
        import pickle
        with open(item_pool_file, 'rb') as f:
            item_pool, _ = pickle.load(f)
    else:
        print(f"Warning: Couldn't load item pool directly. Need to specify item file path.")
        return
    
    # Print stats summary
    print(f"\n{'='*80}")
    print(f"ANALYSIS FOR: {method.upper()} ({item_count} items)")
    print(f"{'='*80}")
    
    print("\nPerformance Summary:")
    for k, v in stats['hit_rates'].items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\nRank Distribution:")
    for k, v in stats['rank_distribution'].items():
        if isinstance(v, float) and not isinstance(v, int):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    # Print detailed results for each request
    print(f"\n{'-'*80}")
    print(f"DETAILED RESULTS (Total: {len(results)} requests)")
    print(f"{'-'*80}")
    
    for i, result in enumerate(results):
        query = result['query']
        enhanced_query = result.get('enhanced_query', None)
        target_id = result['target_item_id']
        target_rank = result['target_rank']
        
        # Get ground truth item details
        ground_truth_item = get_item_by_id(item_pool, target_id)
        if not ground_truth_item:
            print(f"WARNING: Could not find item with ID {target_id} in item pool")
            continue
        
        metadata = ground_truth_item.get('metadata', 'N/A')
        summary = ground_truth_item.get('summary', 'N/A')
        category = ground_truth_item.get('category', 'N/A')
        
        # Print result details
        print(f"\nRequest #{i+1}:")
        print(f"  Query: {query}")
        if enhanced_query and enhanced_query != query:
            print(f"  Enhanced Query: {enhanced_query}")
        print(f"  Target Rank: {target_rank if target_rank >= 0 else 'Not found'}")
        print(f"  Category: {category}")
        
        print("\n  Ground Truth Item:")
        print(f"    ID: {target_id}")
        print(f"    Metadata: {metadata[:200]}..." if len(metadata) > 200 else f"    Metadata: {metadata}")
        print(f"    Summary: {summary}")
        
        # Print some of the top retrieved items if available
        if 'top_retrieved' in result and result['top_retrieved']:
            print("\n  Top Retrieved Items:")
            for j, item in enumerate(result['top_retrieved'][:3]):  # Show top 3
                print(f"    #{j+1}: {item['item_id']} (Score: {item['score']:.4f})")
                print(f"       Summary: {item.get('summary', 'N/A')[:100]}...")
        
        print(f"{'-'*50}")


def main():
    parser = argparse.ArgumentParser(description='Analyze baseline retrieval results')
    parser.add_argument('--method', type=str, default='sim_query_item',
                        choices=['sim_query_item', 'sim_llm_item', 'sim_cot_item'],
                        help='Baseline method to analyze')
    parser.add_argument('--item_count', type=int, default=500,
                        help='Number of items in the dataset')
    
    args = parser.parse_args()
    
    analyze_results(args.method, args.item_count)
    
    # Offer to compare methods
    print("\nWould you like to compare the hit rates across all methods? (y/n)")
    if input().strip().lower() == 'y':
        methods = ['sim_query_item', 'sim_llm_item', 'sim_cot_item']
        
        print(f"\n{'='*80}")
        print(f"COMPARISON OF HIT RATES")
        print(f"{'='*80}")
        
        print(f"\n{'Method':<20} | {'HR@1':<8} | {'HR@3':<8} | {'HR@5':<8} | {'HR@10':<8} | {'HR@20':<8}")
        print(f"{'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        
        for method in methods:
            stats_file = Path('app/baseline/output') / f"{method}_{args.item_count}_stats.json"
            stats = load_json(stats_file)
            
            if not stats:
                continue
                
            hit_rates = stats['hit_rates']
            print(f"{method:<20} | {hit_rates.get('hit_rate@1', 0):<8.4f} | "
                  f"{hit_rates.get('hit_rate@3', 0):<8.4f} | "
                  f"{hit_rates.get('hit_rate@5', 0):<8.4f} | "
                  f"{hit_rates.get('hit_rate@10', 0):<8.4f} | "
                  f"{hit_rates.get('hit_rate@20', 0):<8.4f}")


if __name__ == '__main__':
    main()