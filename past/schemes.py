import logging
import os
import json

import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import compute_ndcg_at_50, dumpj

from debug import check

class EmbeddingRAGScheme:
    """
    RAG scheme using embeddings + Faiss vector search + optional GPT-4 for final check.
    Caches embeddings/index on disk so we can reuse them without re-encoding every run.
    """
    def __init__(self, args, task_data):
        self.args = args
        self.queries = task_data['queries']
        self.item_pool = task_data['item_pool']
        self.item2idx = {item['item_id']: idx for idx, item in enumerate(self.item_pool)}

        self.index_ready = False
        self.model = None
        self.faiss_index = None

        # We'll store item_ids in a known order so we can map from Faiss row -> item.
        self.item_ids_order = [item['item_id'] for item in self.item_pool]

        # Define the embedding model name.
        self.embedding_model_name = getattr(self.args, "embedding_model")

        # Ensure the cache folder exists
        self.cache_dir = "cache"
        os.makedirs(self.cache_dir, exist_ok=True)

        # Use the model name in cache file names, stored under cache/
        self.emb_path = os.path.join(self.cache_dir, f'cached_embeddings_{self.embedding_model_name}.npy')
        self.ids_path = os.path.join(self.cache_dir, f'cached_item_ids_{self.embedding_model_name}.json')
        self.faiss_path = os.path.join(self.cache_dir, f'faiss_index_{self.embedding_model_name}.bin')

    def prep_task_specifics(self):
        """Either load cached embeddings + index, or encode items + build new index."""
        logging.info("Loading sentence-transformer model...")
        self.model = SentenceTransformer(self.embedding_model_name)
        
        # Check if we already have cached embeddings and we do NOT want to overwrite
        if (os.path.exists(self.emb_path) and
            os.path.exists(self.ids_path) and
            os.path.exists(self.faiss_path) and
            not self.args.overwrite_cache):

            logging.info("Cached embeddings/index found. Loading from disk...")

            # 1) Load item_ids and verify same order
            with open(self.ids_path, 'r') as f:
                cached_ids = json.load(f)
            if cached_ids == self.item_ids_order:
                # 2) Load embeddings
                embeddings = np.load(self.emb_path)
                # 3) Load Faiss index
                self.faiss_index = faiss.read_index(self.faiss_path)
                self.index_ready = True
                logging.info("Successfully loaded Faiss index and embeddings from cache.")
                return
            else:
                logging.warning("Cached item_ids.json doesn't match current item pool order. Rebuilding...")

        # Otherwise, encode items from scratch
        logging.info("Encoding item metadata with sentence-transformer. This may take a while...")
        item_texts = [it['metadata'] for it in self.item_pool]
        embeddings = self.model.encode(item_texts, batch_size=64, show_progress_bar=True)
        embeddings = embeddings.astype('float32')  # Faiss expects float32

        logging.info("Normalizing embeddings for cosine similarity...")
        faiss.normalize_L2(embeddings)

        # Build a FlatIP index
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        self.index_ready = True

        # Save to disk for future runs
        logging.info("Saving item_ids, embeddings, and Faiss index to disk.")
        with open(self.ids_path, 'w') as f:
            json.dump(self.item_ids_order, f, indent=2)
        np.save(self.emb_path, embeddings)
        faiss.write_index(self.faiss_index, self.faiss_path)

        logging.info("Faiss index is ready and cached.")

    def retrieve_top_k(self, query_text, top_k=50):
        if not self.index_ready:
            logging.error("Faiss index not built yet.")
            return []

        # Encode the query
        q_emb = self.model.encode([query_text], show_progress_bar=False).astype('float32')
        faiss.normalize_L2(q_emb)

        # Search the index
        D, I = self.faiss_index.search(q_emb, top_k)
        top_indices = I[0]
        return top_indices

    def gpt4_rerank(self, query_text, candidate_items):
        # A placeholder GPT-4 check or re-rank
        out = []
        for it in candidate_items:
            # Real code would call openai.ChatCompletion...
            fit = "Yes"  # dummy
            out.append((it['item_id'], fit))
        return out

    def operate(self):
        all_ndcgs = []
        result_records = []

        total = 0
        in_count = 0
        for q in self.queries:
            qid = q['qid']
            query_text = q['query']
            ground_truth = q['item_id']

            logging.debug(f"Working on query: \"{query_text}\"")
            logging.debug("Ground truth item: %s", json.dumps(self.item_pool[self.item2idx[ground_truth]], indent=2))

            # 1) Retrieve
            top_indices = self.retrieve_top_k(query_text, top_k=200)
            candidate_items = [self.item_pool[idx] for idx in top_indices]

            # Check if ground truth is in top retrieved
            in_candidate = self.item2idx[ground_truth] in top_indices
            logging.info(f'Ground truth in top_indices: {in_candidate}')
            total += 1
            if in_candidate:
                in_count += 1

            # 2) GPT-4 re-check (dummy)
            gpt_results = self.gpt4_rerank(query_text, candidate_items)
            filtered_item_ids = [iid for (iid, fit) in gpt_results if fit == "Yes"]

            # 3) Evaluate
            ndcg_val = compute_ndcg_at_50(filtered_item_ids, ground_truth)
            all_ndcgs.append(ndcg_val)

            record = {
                'query_id': qid,
                'query': query_text,
                'ground_truth_item': ground_truth,
                'retrieved_items': filtered_item_ids,
                'ndcg_at_50': ndcg_val
            }
            result_records.append(record)

        avg_ndcg = sum(all_ndcgs) / len(all_ndcgs) if all_ndcgs else 0.0
        logging.info(f"Average NDCG@50 on {len(self.queries)} queries: {avg_ndcg:.4f}")

        output = {
            'scheme': 'embedding_rag',
            'average_ndcg@50': avg_ndcg,
            'results': result_records
        }
        dumpj(output, os.path.join("output", f'{self.args.task}_{self.args.scheme}.json'))
        logging.info("Saved results JSON.")
        logging.info(f'in_count: {in_count} out of total: {total}')


class KeywordGPT4Scheme:
    """
    Scheme that uses GPT-4 to generate keywords, then does a naive keyword-based search
    over the item metadata.
    """
    def __init__(self, args, task_data):
        self.args = args
        self.queries = task_data['queries']
        self.item_pool = task_data['item_pool']
        self.item2idx = {item['item_id']: idx for idx, item in enumerate(self.item_pool)}

        # For faster naive keyword search, pre-tokenize metadata
        self.prepared_metadata = []
        for item in self.item_pool:
            # Basic cleanup / tokenization
            tokens = item['metadata'].lower().split()
            self.prepared_metadata.append(tokens)

    def prep_task_specifics(self):
        # Here you might load an LLM API key or do other setup.
        # For demonstration, we do nothing special.
        logging.info("KeywordGPT4Scheme: Ready to generate keywords and search items.")

    def generate_keywords_with_gpt(self, query_text):
        """
        Calls GPT to generate a list of relevant keywords for the given query.
        You can customize the prompt, parse the result, etc.
        Below is just a placeholder that slices query into tokens.
        """
        # Placeholder: The real code might look like:
        """
        response = openai.ChatCompletion.create(
            model=self.args.planner_llm,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates good search keywords."
                },
                {
                    "role": "user",
                    "content": f"Given the query: '{query_text}', list 5 relevant keywords (only the keywords)."
                }
            ],
            temperature=0.2
        )
        # parse the response to get the keywords as a list
        # ...
        """
        # For now, we simply return the first three words:
        tokens = query_text.lower().split()
        return tokens[:3]

    def keyword_search(self, keywords, top_k=50):
        """
        A naive approach: Score each item by the number of keyword matches in metadata.
        Return the top_k items' indices.
        """
        scores = []
        for idx, tokens in enumerate(self.prepared_metadata):
            # count how many keywords match (set intersection or repeated counts)
            match_count = sum(kw in tokens for kw in keywords)
            scores.append((match_count, idx))

        # Sort by match_count (descending), then take top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scores[:top_k]]
        return top_indices

    def operate(self):
        all_ndcgs = []
        result_records = []

        total = 0
        in_count = 0
        for q in self.queries:
            qid = q['qid']
            query_text = q['query']
            ground_truth = q['item_id']

            # 1) Generate keywords
            keywords = self.generate_keywords_with_gpt(query_text)
            logging.debug(f"Generated keywords from GPT: {keywords}")

            # 2) Perform naive keyword search
            top_indices = self.keyword_search(keywords, top_k=50)
            candidate_items = [self.item_pool[idx] for idx in top_indices]

            # Check if ground truth is in the top retrieved items
            in_candidate = (self.item2idx[ground_truth] in top_indices)
            total += 1
            if in_candidate:
                in_count += 1
            logging.info(f'Ground truth in top_indices: {in_candidate}')

            # 3) Evaluate using NDCG@50
            retrieved_item_ids = [it['item_id'] for it in candidate_items]
            ndcg_val = compute_ndcg_at_50(retrieved_item_ids, ground_truth)
            all_ndcgs.append(ndcg_val)

            record = {
                'query_id': qid,
                'query': query_text,
                'ground_truth_item': ground_truth,
                'generated_keywords': keywords,
                'retrieved_items': retrieved_item_ids,
                'ndcg_at_50': ndcg_val
            }
            result_records.append(record)

        avg_ndcg = sum(all_ndcgs) / len(all_ndcgs) if all_ndcgs else 0.0
        logging.info(f"Average NDCG@50 on {len(self.queries)} queries: {avg_ndcg:.4f}")

        output = {
            'scheme': 'keyword_gpt4',
            'average_ndcg@50': avg_ndcg,
            'results': result_records
        }
        out_path = os.path.join("output", f'{self.args.task}_{self.args.scheme}.json')
        dumpj(output, out_path)
        logging.info(f"Saved results JSON to {out_path}")
        logging.info(f"in_count: {in_count} out of total: {total}")
