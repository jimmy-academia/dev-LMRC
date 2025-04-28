#!/usr/bin/env python3
"""
Main entry point for running different LMRC implementations.
"""
import argparse
import logging
from utils import set_verbose
from data import load_subsample

def main():
    parser = argparse.ArgumentParser(description='Run LMRC implementations')
    parser.add_argument('--app', '-a', type=str, default='multistep',
                        choices=['oneshot', 'multistep'],
                        help='Choose the method approach to tackle LMRC')
    parser.add_argument('--verbose', '-v', type=int, default=1,
                        help='Verbosity level (0-2)')
    
    args = parser.parse_args()
    set_verbose(args.verbose)

    ## prepare environment: item_pool, requests
    item_count = 1000
    test_count = 20
    item_pool, requests = load_subsample(item_count)

    input('good')

    if args.app == 'oneshot':
        from app.oneshot.run import run
    elif args.app == 'multistep':
        from app.multistep.run import run 
    else:
        logging.error(f"Unknown approach: {args.app}")
        exit(1)

    run()
    
if __name__ == '__main__':
    main()