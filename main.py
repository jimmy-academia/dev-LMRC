import logging
from pathlib import Path

from utils import set_seeds, set_verbose, ensure_dir, create_llm_client
from data import prepare_item_requests, prepare_file_system
from agent import ReactAgent

from debug import check

def main():
    """Main function to run the improved search agent."""
    set_seeds(1)
    set_verbose(1)
    ensure_dir('cache')
    
    # Load data
    data_pkl = Path('cache/queries_item_pool.pkl')
    item_pool, requests = prepare_item_requests(data_pkl)
    
    # Load or initialize file system
    fs_pkl = Path('cache/category_file_system.pkl')
    fs = prepare_file_system(fs_pkl, item_pool)
    fs.verify_categories() # print(f"  {category}: {len(items)} items")
    
    # Create LLM client
    llm_client = create_llm_client()
    
    # Create agent with cached search index
    agent = ReactAgent(fs, llm_client)
    
    for request in requests:
        query = request['query']
        gt = request['item_id']
        gt_cat = fs.id_to_item[gt]['category']
        gt_meta = fs.id_to_item[gt]['metadata']
    
        logging.info(f"\n Running search for: \"{query}\"\n Ground Truth ID: {gt}, category {gt_cat}, \n metadata: {gt_meta[:90]}....")
    
        results = agent.search(query, gt, gt_cat, max_steps=5)
        
        if results["success"]:
            print("\nTop 5 Matches:")
            for i, item in enumerate(results["items"][:5]):
                print(f"{i+1}. {item['metadata']}")  # Print full metadata
                if "path" in item:
                    print(f"   Path: {item['path']}")
                if "tags" in item:
                    print(f"   Tags: {', '.join(item['tags'])}")
                print()
            
            print(f"\nSearch summary: {results['summary']}")
        else:
            print(f"\nSearch failed: {results['message']}")
        # future task: agent.save_file_system() qid....
        
    # future task: Create an output report create_output_report(results, resultpath...)
    
    logging.info("\nSearch complete.")
    
    

if __name__ == "__main__":
    main()