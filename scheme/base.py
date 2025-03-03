import logging
import os
import json

from utils import compute_ndcg_at_50, dumpj

class BaseScheme:
    """
    Base scheme class: 
      - Houses common initialization (args, queries, item_pool).
      - Implements the main operate() loop. 
      - Calls child-specific _get_final_candidates() for retrieval logic.
    """
    def __init__(self, args, task_data):
        self.args = args
        self.queries = task_data['queries']
        self.item_pool = task_data['item_pool']
        # Map item_id -> index
        self.item2idx = {item['item_id']: idx for idx, item in enumerate(self.item_pool)}

        # If we want to be consistent with embedding_rag, we can store top_k=200 as default
        # Or let the user pass it in from the command line
        self.top_k = getattr(self.args, "top_k", 200)

    def prep_task_specifics(self):
        """
        Called before operate(), for building indices or any heavy setup.
        Child classes should override if needed.
        """
        pass  # Default: no-op

    def _get_final_candidates(self, query_text):
        """
        Child classes must override to provide the final list of item_ids for this query.
        """
        raise NotImplementedError("Child classes must implement '_get_final_candidates'.")

    def operate(self):
        """
        The main pipeline for processing each query:
          1) For each query, call _get_final_candidates(...)
          2) Evaluate using NDCG@50
          3) Save or log results
        """
        all_ndcgs = []
        result_records = []

        total = 0
        in_count = 0
        rank_list = []

        for q in self.queries:
            qid = q['qid']
            query_text = q['query']
            ground_truth = q['item_id']
            
            # Retrieve final candidate item_ids from child scheme
            final_candidates = self._get_final_candidates(query_text)
            # Make sure it's a list of item_ids
            if not isinstance(final_candidates, list):
                raise ValueError("Child scheme must return a list of item_ids.")

            # If we want to enforce top_k here too, just in case:
            final_candidates = final_candidates[: self.top_k]

            # Check if ground truth is in the retrieved set
            in_candidate = (ground_truth in final_candidates)
            total += 1
            if in_candidate:
                in_count += 1
                rank_list.append(final_candidates.index(ground_truth))
            else:
                rank_list.append(-1)


            logging.info(f"Ground truth in final candidates: {in_candidate}")

            # Evaluate NDCG@50
            ndcg_val = compute_ndcg_at_50(final_candidates, ground_truth)
            all_ndcgs.append(ndcg_val)

            record = {
                'query_id': qid,
                'query': query_text,
                'ground_truth_item': ground_truth,
                'retrieved_items': final_candidates,
                'ndcg_at_50': ndcg_val
            }
            result_records.append(record)

        # Summaries
        avg_ndcg = sum(all_ndcgs) / len(all_ndcgs) if all_ndcgs else 0.0
        logging.info(f"Average NDCG@50 on {len(self.queries)} queries: {avg_ndcg:.4f}")

        output = {
            'scheme': self.args.scheme,
            'average_ndcg@50': avg_ndcg,
            'results': result_records
        }
        out_file = os.path.join("output", f"{self.args.task}_{self.args.scheme}.json")
        dumpj(output, out_file)
        logging.info(f"Saved results JSON to {out_file}")
        logging.info(f"in_count: {in_count} out of total: {total}")
        logging.info(f"rank_list {rank_list}")
