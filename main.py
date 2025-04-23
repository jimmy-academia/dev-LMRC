#!/usr/bin/env python3
"""
Main entry point for running different LMRC implementations.
"""
import argparse
import logging
from utils import set_verbose

def main():
    parser = argparse.ArgumentParser(description='Run LMRC implementations')
    parser.add_argument('--app', '-a', type=str, default='oneshot',
                        choices=['oneshot', 'multistep'],
                        help='Choose the method approach to tackle LMRC')
    parser.add_argument('--verbose', '-v', type=int, default=1,
                        help='Verbosity level (0-2)')
    
    args = parser.parse_args()
    set_verbose(args.verbose)
    
    if args.approach == 'oneshot':
        from app.oneshot.run import run
    elif args.approach == 'multistep':
        from app.multistep.run import run 
    else:
        logging.error(f"Unknown approach: {args.approach}")
        exit(1)

    run()
    
if __name__ == '__main__':
    main()