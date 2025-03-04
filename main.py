import os
import logging
import argparse
from pathlib import Path

from loader import get_task_loader
# main.py
from scheme import (
    EmbeddingRAGScheme,
    KeywordGPT4BasicScheme,
    TaxonomyGPT4Scheme
)

from utils import set_seeds, set_verbose, readf

import openai

def setup_scheme(args, task_data):
    if args.scheme == 'embedding_rag':
        return EmbeddingRAGScheme(args, task_data)
    elif args.scheme == 'keyword_gpt4':
        return KeywordGPT4BasicScheme(args, task_data)
    elif args.scheme == 'taxonomy':
        return TaxonomyGPT4Scheme(args, task_data)
    else:
        raise ValueError(f"Unknown scheme: {args.scheme}")

def set_arguments():
    parser = argparse.ArgumentParser(description='Run retrieval-based approaches on Amazon-C4.')
    
    # environment
    
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='verbose level')
    # logging decisions
    parser.add_argument('--ckpt', type=str, default='ckpt', help='checkpoint directory')

    # Task / Scheme
    parser.add_argument('--task', type=str, default='retrieval_task', help='Task name for logging/output files.')
    parser.add_argument('--scheme', type=str, default='taxonomy', 
                        help='Which retrieval scheme to run.')
    parser.add_argument('--embedding_model', type=str, default="all-mpnet-base-v2",
                        help='Transformer model to use for embeddings')
    '''
    Possible embedding models:
    all-MiniLM-L6-v2, all-mpnet-base-v2, multi-qa-mpnet-base-cos-v1,
    all-roberta-large-v1, all-distilroberta-v1, paraphrase-mpnet-base-v2,
    sentence-t5-large (or sentence-t5-base)
    '''
    
    parser.add_argument('--top_k', type=int, default=105841,
                        help='Max number of retrieved documents to consider in final results.')
    #1058417

    # Whether to overwrite existing embeddings/index files
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute item embeddings and Faiss index, overwriting cached files.')

    args = parser.parse_args()

    # For dev mode only
    args.dev = True
    return args

def main():
    args = set_arguments()

    key_path = ".openaikey"
    if os.path.exists(key_path):
        openai_api_key = readf(key_path).strip()
        openai.api_key = openai_api_key
        logging.info("OpenAI API key loaded from 'cache/openaikey'.")
    else:
        logging.warning(
            "No 'cache/openaikey' file found. "
            "Make sure to set openai.api_key manually or via environment variables."
        )
        input('check openai key')

    set_seeds(args.seed)
    set_verbose(args.verbose)

    Path('output').mkdir(exist_ok=True)
    args.record_path = Path(f'output/{args.task}_{args.scheme}.json')

    # if args.record_path.exists() and not args.overwrite:
        # logging.info(f'{args.record_path} exists, skipping.')
        # return

    logging.info(f'Running scheme: {args.scheme}')

    # 1) Load data
    task_data = get_task_loader(args)

    # 2) Set up scheme
    scheme = setup_scheme(args, task_data)

    # 3) Prepare scheme specifics (index building, data structure creation, etc.)
    scheme.prep_task_specifics()

    # 4) Retrieve & Evaluate
    scheme.operate()

if __name__ == '__main__':
    main()
