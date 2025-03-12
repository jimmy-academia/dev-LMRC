import re
import time
import torch
import pickle
import json
import random

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# (Imports from the HF datasets or your local caching logic)
from datasets import load_dataset
from huggingface_hub import hf_hub_download

############################################
# 1. Data Loading (same as your existing code)
############################################

def prepare_item_query(data_pkl):
    if not data_pkl.exists():
        queries = load_dataset('McAuley-Lab/Amazon-C4')['test']
        filepath = hf_hub_download(
                repo_id='McAuley-Lab/Amazon-C4',
                filename='sampled_item_metadata_1M.jsonl',
                repo_type='dataset'
            )
        item_pool = []
        with open(filepath, 'r') as f:
            for line in f:
                item_pool.append(json.loads(line.strip()))
        with open(data_pkl, 'wb') as f:
            pickle.dump((item_pool, queries), f)
    else:
        with open(data_pkl, 'rb') as f:
            item_pool, queries = pickle.load(f)

    return item_pool, queries

############################################
# 2. Build an inverted index for fast lookups
############################################

def prepare_inverted_index(inverted_index_pkl):
    # Build or load the inverted index
    if not inverted_index_pkl.exists():
        inverted_index = build_inverted_index(item_pool)
        with open(inverted_index_pkl, 'wb') as f:
            pickle.dump(inverted_index, f)
    else:
        with open(inverted_index_pkl, 'rb') as f:
            inverted_index = pickle.load(f)

    return inverted_index


def build_inverted_index(items):
    inverted_index = defaultdict(set)
    for idx, item in enumerate(tqdm(items, desc='building inverted index...', ncols=88)):
        # Get metadata text; default to empty string if missing.
        metadata = item.get('metadata', '')
        # Tokenize: extract words and convert to lowercase.
        tokens = re.findall(r'\w+', metadata.lower())
        for token in tokens:
            inverted_index[token].add(idx)
    return inverted_index

def query_index(inverted_index, query):
    """
    Given an inverted index and a query string, return the set of item indices 
    that contain all tokens in the query.
    """
    tokens = re.findall(r'\w+', query.lower())
    if not tokens:
        return set()
    # Retrieve the set of indices for each token found in the query.
    result_sets = [inverted_index[tok] for tok in tokens if tok in inverted_index]
    if not result_sets:
        return set()
    # Intersect sets to ensure all tokens are present.
    return set.intersection(*result_sets)

def get_items_from_indices(items, indices):
    """
    Retrieve the full items given a list (or set) of indices.
    """
    return [items[i] for i in indices]


############################################
# 3. Structures for conversation / GPT calls
############################################

import openai
from utils import readf
# You might have your own keys and configurations here.
openai.api_key = readf(".openaikey").strip()

def user_struct(x): 
    return {"role": "user", "content": x}

def system_struct(x): 
    return {"role": "system", "content": x}

def assistant_struct(x): 
    return {"role": "assistant", "content": x}

def gpt_call(messages, model="gpt-4"):
    """
    Example helper to call the OpenAI chat API. 
    You might want to adjust parameters or handle errors / streaming, etc.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=0.0,
    )
    return response.choices[0].message.content

############################################
# 4. Agent State & Actions
############################################

class AgentState:
    """
    Keeps track of various sets of item indices:
    - `positive_set`: items that have been deemed relevant so far
    - `negative_set`: items that have been excluded or marked irrelevant
    - `candidate_set`: everything that remains a candidate (not yet placed into positive or negative sets)
    
    You can expand this as needed, for example:
    - to track “currently viewed” items,
    - to track items that are “in question,” 
    - or any other specialized subsets.
    """
    def __init__(self, total_size):
        self.positive_set = set()
        self.negative_set = set()
        self.candidate_set = set(range(total_size))
    
    def __repr__(self):
        return (f"AgentState(\n"
                f"  positive_set_size={len(self.positive_set)},\n"
                f"  negative_set_size={len(self.negative_set)},\n"
                f"  candidate_set_size={len(self.candidate_set)}\n"
                f")")

def union_with_positive(state: AgentState, new_indices: set):
    """
    Union the new indices into the positive set, and remove them from 
    the candidate set if they are there.
    """
    state.positive_set |= new_indices
    state.candidate_set -= new_indices

def subtract_with_negative(state: AgentState, new_indices: set):
    """
    Put these new indices in the negative set, and remove them from 
    both candidate and positive sets if present.
    """
    state.negative_set |= new_indices
    state.positive_set -= new_indices
    state.candidate_set -= new_indices

def sample_from_set(the_set: set, k=5):
    """
    Randomly sample k indices from the given set (if fewer than k, return all).
    """
    if len(the_set) <= k:
        return list(the_set)
    return random.sample(list(the_set), k)

############################################
# 5. Orchestrating the Iterative Loop
############################################

def run_agent_iteration(item_pool, inverted_index, initial_query, steps=10):
    """
    Example loop to show how you might:
      1. Start by sampling from the entire dataset and put it in `positive_set`.
      2. Repeatedly sample from sets or run lexical searches,
         then let GPT decide the next action.
      3. Apply those actions to refine your `AgentState`.
    """
    # 5.1 Initialize the AgentState
    state = AgentState(total_size=len(item_pool))
    
    # Create a function to show the LLM or user some items
    def summarize_random_items(state, n=10, from_positive=True):
        """
        Sample from either the positive set or negative set 
        (or candidate set) and return a text snippet describing them.
        """
        source = state.positive_set if from_positive else state.negative_set
        if not source:
            return "No items in this set."
        sampled_indices = sample_from_set(source, n)
        items = get_items_from_indices(item_pool, sampled_indices)
        # Build a simple summary string
        summary = []
        for idx, itm in zip(sampled_indices, items):
            meta = itm.get('metadata', '')
            summary.append(f"Index: {idx} | Metadata: {meta[:100]}...")
        return "\n".join(summary)
    
    messages = [
        system_struct("You are an expert assistant for refining search results."),
        user_struct(query)
        system_struct(f"Initially, we have an entire dataset in the candidate set. We want to iteratively put search result into positive set or negative set."
                    "We can perform the following actions:\n"
                    "1) 'Action: lexical_search <keyword>' to get a set of items.\n"
                    "2) 'Action: union_positive': union the last searched set with the current positive set.\n"
                    "3) 'Action: subtract_negative': move the last searched set into the negative set.\n"
                    "4) 'Action: sample <positive|negative|candidate> <number>' see a random sample from any set (positive, negative, or candidate) to decide, e.g., 'Action: sample positive 5.\n"
                    "Please propose the next action (and an optional reason).")
    ]
    
    last_search_result = set()
    
    for step in range(steps):
        # Call the model to get the next action
        assistant_reply = gpt_call(messages)
        
        # For demonstration, we’ll just print out what the LLM “says,”
        # then parse it in some naive manner. In practice, you’d need 
        # a robust parser for the LLM’s instructions.
        print(f"\n=== Step {step+1} ===")
        print(f"\n input prompt: \n")
        for message in messages:
            print(message)
        print(f"LLM says:\n{assistant_reply}\n")

        messages.append(assistant_struct(assistant_reply))

        input()
        # 1) Identify an action in the LLM text:
        #    For example, look for lines like "Action: lexical_search cats"
        #    or "Action: union_positive"
        
        # Very naive pattern matching, as an example
        lines = assistant_reply.lower().split("\n")
        action_line = None
        for l in lines:
            if "action:" in l:
                action_line = l
                break
        
        if not action_line:
            # If no recognized action, we can proceed or break
            messages.append(system_struct("No clear action found. Please specify an action, e.g. 'Action: lexical_search cats'."))
            continue
        
        # parse out something like "lexical_search cats" after "Action:"
        action_line = action_line.split("action:")[-1].strip()  # get the text after 'Action:'
        
        print('+++++ processed lines +++++')
        print(lines)
        print('---action line---')
        print(action_line)
        input()

        # Very naive approach to parse:
        # handle "lexical_search <keyword>"
        if action_line.startswith("lexical_search"):
            # e.g. "lexical_search cats"
            parts = action_line.split()
            if len(parts) >= 2:
                keyword = " ".join(parts[1:])
                # run lexical search
                last_search_result = query_index(inverted_index, keyword)
                # Add a user message summarizing the search
                messages.append(system_struct(
                    f"Performed lexical search for '{keyword}' -> found {len(last_search_result)} items."
                ))
            else:
                messages.append(system_struct("No keyword found after lexical_search."))
        
        elif action_line.startswith("union_positive"):
            # union the last_search_result with the positive set
            union_with_positive(state, last_search_result)
            messages.append(system_struct(f"Union done. Positive set now has {len(state.positive_set)} items."))
        
        elif action_line.startswith("subtract_negative"):
            # subtract the last_search_result to the negative set
            subtract_with_negative(state, last_search_result)
            messages.append(system_struct(
                f"Subtraction done. Negative set now has {len(state.negative_set)} items."
            ))
        
        elif action_line.startswith("sample"):
            # e.g. "sample positive 5" or "sample negative 3"
            # parse out target set and number
            parts = action_line.split()
            if len(parts) == 3:
                target_set = parts[1]
                sample_count = int(parts[2])
                if target_set == "positive":
                    text_summary = summarize_random_items(state, sample_count, from_positive=True)
                elif target_set == "negative":
                    text_summary = summarize_random_items(state, sample_count, from_positive=False)
                else:
                    # assume "candidate"
                    sampled_indices = sample_from_set(state.candidate_set, sample_count)
                    items = get_items_from_indices(item_pool, sampled_indices)
                    text_summary = "\n".join([f"Index: {idx} | Metadata: {itm.get('metadata','')[:100]}..." 
                                              for idx, itm in zip(sampled_indices, items)])
                messages.append(system_struct(f"Sample from {target_set}:\n{text_summary}"))
            else:
                messages.append(system_struct("Invalid sample command. Use: 'sample <positive|negative|candidate> <number>'."))
        
        else:
            messages.append(system_struct(f"Unrecognized action: {action_line}"))
        
        # Optionally, you can add a message about the current state
        messages.append(system_struct(str(state)))
        
    print("\nFinal state:", state)
    return state


############################################
# 6. Run the Agent
############################################

def main():
    data_pkl = Path('cache/queries_item_pool.pkl')
    item_pool, queries = prepare_item_query(data_pkl)
    inverted_index_pkl = Path('cache/item_pool_inverted_index.pkl')
    inverted_index = prepare_inverted_index(inverted_index_pkl)
    query = queries[0]['query']
    ground_truth = queries[0]['item_id']

    final_state = run_agent_iteration(item_pool, inverted_index, query, steps=10)
    print("Done.")



if __name__ == "__main__":
    main()
    