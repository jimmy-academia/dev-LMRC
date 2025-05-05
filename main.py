#!/usr/bin/env python3
"""
Main entry point for running different LMRC implementations.
"""
import argparse
import logging
from utils import set_seeds, set_verbose
from data import load_subsample, load_sample

from debug import check

def main():
    parser = argparse.ArgumentParser(description='Run LMRC implementations')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', '-v', type=int, default=1,
                        help='Verbosity level (0-2)')
    parser.add_argument('--app', '-a', type=str, default='multistep',
                        choices=['oneshot', 'multistep', 'sim_query_item', 'sim_llm_item', 'sim_cot_item'],
                        help='Choose the method approach to tackle LMRC')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='Choose Openai API model.')
    # gpt-4.1-nano does not work...
    
    args = parser.parse_args()
    set_seeds(args.seed)
    set_verbose(args.verbose)

    ## prepare environment: item_pool, requests
    # item_count = 500
    # test_count = 100
    # item_pool, requests = load_subsample(item_count, test_count=test_count)
    item_pool, requests = load_sample()
    requests = requests[:100]

    logging.info(f"loaded {len(item_pool)} items and {len(requests)} requests.")

    if args.app == 'oneshot':
        from app.oneshot.run import run
    elif args.app == 'multistep':
        from app.multistep.run import run 
    ## the other baselines ##
    elif args.app in ['sim_query_item', 'sim_llm_item', 'sim_cot_item']:
        from app.baseline.similarity import run
    elif args.app == "generative_retrieval":
        from app.generative.retrieval import run
    else:
        logging.error(f"Unknown approach: {args.app}")
        exit(1)


    run(args, item_pool, requests)
    
if __name__ == '__main__':
    main()