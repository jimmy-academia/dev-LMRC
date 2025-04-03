import logging
import argparse
from pathlib import Path

from utils import set_seeds, set_verbose, ensure_dir, create_llm_client
from data import prepare_item_requests, prepare_file_system
from agent import get_agent  # Import the factory function
from helper import create_output_report

from debug import check

def main():
    """Main function to run the improved search agent."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run product search with different agent strategies')
    parser.add_argument('--agent', type=str, default='recursive', choices=['base', 'recursive', 'bag'],
                      help='Agent implementation to use (default: base)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2], 
                      help='Verbosity level (0=minimal, 1=info, 2=debug)')
    # parser.add_argument('--steps', type=int, default=5, help='Maximum steps per query')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    args = parser.parse_args()
    
    # Setup environment
    set_seeds(args.seed)
    set_verbose(args.verbose)
    ensure_dir('cache')
    ensure_dir(args.output)
    
    # Load data
    data_pkl = Path('cache/queries_item_pool.pkl')
    item_pool, requests = prepare_item_requests(data_pkl)
    
    # Load or initialize file system
    fs_pkl = Path('cache/category_file_system.pkl')
    fs = prepare_file_system(fs_pkl, item_pool)
    fs.verify_categories()
    
    # Create LLM client
    llm_client = create_llm_client()
    
    # Get the requested agent implementation
    ReactAgent = get_agent(args.agent)
    logging.info(f"Using {args.agent} agent implementation")
    
    # Create agent with cached search index
    agent = ReactAgent(fs, llm_client)
    
    for i, request in enumerate(requests):
        query = request['query']
        gt = request['item_id']
        gt_cat = fs.id_to_item[gt]['category']
        gt_meta = fs.id_to_item[gt]['metadata']
    
        logging.info(f"\n Running search {i+1}/{len(requests)} for: \"{query}\"\n Ground Truth ID: {gt}, category {gt_cat}, \n metadata: {gt_meta[:90]}...")
    
        results = agent.search(query, gt, gt_cat)
        
        if results["success"]:
            print("\nTop 5 Matches:")
            for i, item in enumerate(results["items"][:5]):
                print(f"{i+1}. {item['metadata']}")
                if "path" in item:
                    print(f"   Path: {item['path']}")
                if "tags" in item:
                    print(f"   Tags: {', '.join(item['tags'])}")
                print()
            
            print(f"\nSearch summary: {results['summary']}")
            
            # Create output report
            output_path = Path(args.output) / f"results_{args.agent}_{i+1}.html"
            create_output_report(results, output_path)
        else:
            print(f"\nSearch failed: {results.get('message', 'Unknown error')}")
        
        # Save file system state
        agent.save_file_system(fs_pkl)
    
    logging.info("\nSearch complete.")

if __name__ == "__main__":
    main()