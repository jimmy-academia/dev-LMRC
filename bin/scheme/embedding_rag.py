import logging
import os
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils import dumpj
from debug import check
from .base import BaseScheme

class EmbeddingRAGScheme(BaseScheme):
    """
    RAG scheme using embeddings + Faiss vector search + optional GPT-4 re-check.
    Caches embeddings/index on disk so we can reuse them without re-encoding every run.
    """
    def __init__(self, args, task_data):
        super().__init__(args, task_data)
        self.index_ready = False
        self.model = None
        self.faiss_index = None

        # We'll store item_ids in a known order so we can map from Faiss row -> item.
        self.item_ids_order = [item['item_id'] for item in self.item_pool]
        self.num_item = len(self.item_pool)

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
        logging.info("[EmbeddingRAGScheme] Loading or building Faiss index.")
        self.model = SentenceTransformer(self.embedding_model_name)
        
        # Check if we already have cached embeddings and we do NOT want to overwrite
        if (os.path.exists(self.emb_path) and
            os.path.exists(self.ids_path) and
            os.path.exists(self.faiss_path) and
            not self.args.overwrite):

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
                logging.warning("Cached item_ids doesn't match current item pool order. Rebuilding...")

        # Otherwise, encode items from scratch
        logging.info("Encoding item metadata. This may take a while...")
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

        logging.info("Faiss index built and cached.")

    def _retrieve_candidates(self, query_text, top_k):
        """
        Encode query and retrieve top_k items from Faiss index (by similarity).
        Return a list of item indices.
        """
        if not self.index_ready:
            logging.error("Faiss index not ready. Did you call prep_task_specifics()?")
            return []

        q_emb = self.model.encode([query_text], show_progress_bar=False).astype('float32')
        faiss.normalize_L2(q_emb)

        # Search
        D, I = self.faiss_index.search(q_emb, top_k)
        return I[0]  # array of indices

    def _gpt4_rerank(self, query_text, candidate_indices):
        """
        Optional GPT-4 logic. Here we just do a dummy pass-through.
        Return a list of item_ids that pass the 'Yes' filter.
        """
        # Convert indices -> item objects
        candidate_items = [self.item_pool[idx] for idx in candidate_indices]

        # For demonstration, pretend GPT always says "Yes":
        filtered_items = []
        for it in candidate_items:
            filtered_items.append(it['item_id'])
        return filtered_items

    def _get_final_candidates(self, query_text):
        """
        The child-specific method that retrieves + reranks and returns final item_ids.
        """
        print(self.num_item)
        candidate_indices = self._retrieve_candidates(query_text, top_k=self.top_k)
        final_item_ids = self._gpt4_rerank(query_text, candidate_indices)
        return final_item_ids
