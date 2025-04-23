import re
import time
import torch
import pickle
import json
import random

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# If you rely on HuggingFace’s datasets and caching logic:
from datasets import load_dataset
from huggingface_hub import hf_hub_download

############################################
# 1. Data Loading
############################################

def prepare_item_query(data_pkl: Path):
    """
    Loads (or downloads and caches) the item_pool and queries.
    item_pool is a list of dicts, e.g.:
      {
        'item_id': 'B0778XR2QM', 
        'category': 'Care', 
        'metadata': 'Supergoop! Super Power Sunscreen Mousse ...'
      }
    queries is a HF dataset (example).
    """
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
# 2. Category Index & Inverted Index
############################################

def build_category_index(items):
    """
    Build a dictionary mapping category -> set of item indices.
    E.g. category_index['Care'] = {0, 12, 103, ...}
    """
    category_index = defaultdict(set)
    for idx, item in enumerate(items):
        cat = item.get('category', 'Unknown')
        category_index[cat].add(idx)
    return category_index

def build_inverted_index(items):
    """
    Build a naive inverted index that maps token -> set of item indices,
    across the *entire* dataset. We can refine usage by intersecting
    with categories or subcategories as needed.
    """
    inverted_index = defaultdict(set)
    for idx, item in enumerate(tqdm(items, desc='building inverted index...', ncols=88)):
        metadata = item.get('metadata', '')
        tokens = re.findall(r'\w+', metadata.lower())
        for token in tokens:
            inverted_index[token].add(idx)
    return inverted_index

def query_index(inverted_index, query, restrict_to=None):
    """
    Perform a lexical search. 
      - 'query' is a string.
      - 'restrict_to' is an optional set of item indices to limit the search to 
        (e.g., all items from a certain category or subcategory).
    Return the set of item indices that contain *all tokens* in the query.
    """
    tokens = re.findall(r'\w+', query.lower())
    if not tokens:
        return set()
    # Retrieve sets for each token:
    if restrict_to is None:
        # Search entire space
        sets_found = [inverted_index.get(tok, set()) for tok in tokens]
    else:
        # Restrict to a subset
        sets_found = [inverted_index.get(tok, set()).intersection(restrict_to) 
                      for tok in tokens]
    if not sets_found or any(len(s) == 0 for s in sets_found):
        return set()
    # Intersect to ensure items contain all tokens.
    return set.intersection(*sets_found)

############################################
# 3. Subcategory Extraction (Optional)
############################################

def discover_subcategories(items, category_index):
    """
    Example approach:
      For each category, parse the metadata of items, look for common keywords,
      or cluster items, or (optionally) use GPT to guess a subcategory.
    
    Here we do a naive example: 
      - We'll look for 'sunscreen', 'toy', 'electronics', etc. in the metadata
      - Then group accordingly. 
    In real usage, you'd want a more robust approach, or a GPT-based classifier.
    """
    # Pre-define some naive subcategory keywords by domain:
    subcat_keywords = {
        'sunscreen': 'sunscreen|spf|sun block',
        'toy': 'toy|toyset|lego',
        'skincare': 'cream|serum|lotion|skincare',
        'electronics': 'lcd|headphones|usb|electronics|bluetooth',
        'clothing': 'shirt|pants|dress|apparel',
        # add more as needed, or do dynamic clustering
    }
    
    # subcategories_map[category][subcat_name] = set of indices
    subcategories_map = defaultdict(lambda: defaultdict(set))
    
    for cat, indices in category_index.items():
        for idx in indices:
            metadata = items[idx].get('metadata', '').lower()
            # Check each subcategory pattern
            matched_any = False
            for subcat_name, pattern in subcat_keywords.items():
                if re.search(pattern, metadata):
                    subcategories_map[cat][subcat_name].add(idx)
                    matched_any = True
            if not matched_any:
                # Put into a "misc" bucket
                subcategories_map[cat]['misc'].add(idx)
    
    return subcategories_map

############################################
# 4. Agent State & Utility
############################################

class AgentState:
    """
    This improved state tracks:
      - category_filter: which category (or categories) we are currently focusing on
      - subcategory_filter: which subcategory we are focusing on
      - candidate_set: items still under consideration
      - positive_set: items we have deemed relevant
      - negative_set: items we have deemed irrelevant
      - last_search_result: the most recent set of retrieved items
    """
    def __init__(self, total_size):
        # Start out with everything as a candidate.
        self.candidate_set = set(range(total_size))
        self.positive_set = set()
        self.negative_set = set()
        self.category_filter = None
        self.subcategory_filter = None
        self.last_search_result = set()
    
    def __repr__(self):
        return (f"AgentState(\n"
                f"  category_filter={self.category_filter},\n"
                f"  subcategory_filter={self.subcategory_filter},\n"
                f"  positive_set_size={len(self.positive_set)},\n"
                f"  negative_set_size={len(self.negative_set)},\n"
                f"  candidate_set_size={len(self.candidate_set)},\n"
                f"  last_search_result_size={len(self.last_search_result)}\n"
                f")")

def get_items_from_indices(items, indices):
    return [items[i] for i in indices]

def sample_from_set(the_set: set, k=5):
    if len(the_set) <= k:
        return list(the_set)
    return random.sample(list(the_set), k)

############################################
# 5. GPT-based Relevance Checking
############################################

import openai
def user_struct(x): 
    return {"role": "user", "content": x}

def system_struct(x): 
    return {"role": "system", "content": x}

def assistant_struct(x): 
    return {"role": "assistant", "content": x}

def gpt_call(messages, model="gpt-4", temperature=0.0, max_tokens=300):
    """
    Basic helper to call the OpenAI chat API. Adjust as needed.
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content

def review_items_with_gpt(items, user_request, model="gpt-4"):
    """
    Given a list of items (dicts) and the user’s request, call GPT to check relevance.
    Returns a list of booleans (True if relevant, False otherwise).
    
    Warning: If 'items' is large, you'd need a more sophisticated approach
    (chunking, summarizing, or partial reviews). This is just a demonstration.
    """
    results = []
    for itm in items:
        snippet = itm.get('metadata', '')[:300]  # short snippet
        messages = [
            system_struct("You are a helpful assistant that checks item relevance to a user request."),
            user_struct(f"User request: {user_request}\n\nItem metadata snippet: {snippet}\n\n"
                        f"Question: Does this item satisfy the user's request? "
                        f"Answer only Yes or No, then a short reason.")
        ]
        response = gpt_call(messages, model=model)
        # Very naive parse:
        # We'll look for "Yes" or "No" in response; refine as needed.
        lower_resp = response.lower()
        if "yes" in lower_resp and "no" not in lower_resp:
            results.append(True)
        elif "no" in lower_resp and "yes" not in lower_resp:
            results.append(False)
        else:
            # fallback or ambiguous
            results.append(False)
    return results

############################################
# 6. The Main Iterative Loop
############################################

def run_agent_iteration(items,
                       category_index,
                       subcategories_map,
                       inverted_index,
                       user_request,
                       steps=10):
    """
    An example loop showing how the agent can refine the search.
    
    The agent can:
      - pick a category
      - pick a subcategory
      - do a lexical search
      - do a GPT-based relevance check
      - move items to positive/negative sets
      - sample from sets to inspect items
      - readjust actions after seeing a sample
    """
    state = AgentState(total_size=len(items))
    
    # For prompting, we keep track of the conversation messages:
    messages = [
        system_struct("You are an expert assistant for refining a user’s e-commerce search."),
        user_struct(f"User request: {user_request}")
    ]
    
    # Provide an explicit system guideline to the assistant:
    system_instructions = (
        "You have the following possible actions:\n"
        "1) Action: pick_category <category_name>\n"
        "2) Action: pick_subcategory <subcat_name>\n"
        "3) Action: lexical_search <keyword or phrase>\n"
        "4) Action: relevance_check\n"
        "5) Action: union_positive\n"
        "6) Action: subtract_negative\n"
        "7) Action: sample <positive|negative|candidate> <number>\n"
        "8) Action: stop\n"
        "\n"
        "After each step, we (the orchestrating code) will parse your action and respond. "
        "You may see short item snippets if you request a sample. "
        "Aim to produce a final positive_set of items that truly match the user’s request."
    )
    messages.append(system_struct(system_instructions))
    
    for step in range(steps):
        
        # Show the agent the current state (could be truncated in real usage):
        messages.append(system_struct(str(state)))
        
        # Now ask the assistant what to do next:
        assistant_reply = gpt_call(messages, model="gpt-4")
        print(f"\n=== Step {step+1} ===")
        print("Assistant says:")
        print(assistant_reply)
        
        # We store that into the conversation:
        messages.append(assistant_struct(assistant_reply))
        
        # Parse out the action line. We do naive pattern matching:
        lines = assistant_reply.lower().split("\n")
        action_line = None
        for l in lines:
            if "action:" in l:
                action_line = l
                break
        
        if not action_line:
            # No action found, ask for clarification:
            clarification = "No action found. Please specify an action, e.g. 'Action: pick_category Beauty'."
            messages.append(system_struct(clarification))
            continue
        
        # Extract text after "Action:"
        action_text = action_line.split("action:")[-1].strip()
        
        # Handle each possible action
        if action_text.startswith("pick_category"):
            parts = action_text.split()
            if len(parts) < 2:
                messages.append(system_struct("No category name found after 'pick_category'."))
                continue
            cat_name = " ".join(parts[1:])
            # Check if cat_name is valid:
            if cat_name not in category_index:
                messages.append(system_struct(
                    f"Category '{cat_name}' not found in known categories. Known: {list(category_index.keys())}"
                ))
                continue
            # Filter candidate_set to items within that category:
            cat_indices = category_index[cat_name]
            # Also remove from candidate_set anything not in cat_indices:
            state.candidate_set = state.candidate_set.intersection(cat_indices)
            state.category_filter = cat_name
            # Reset subcategory if switching category
            state.subcategory_filter = None
            messages.append(system_struct(f"Picked category '{cat_name}'. Candidate set now has {len(state.candidate_set)} items."))
        
        elif action_text.startswith("pick_subcategory"):
            parts = action_text.split()
            if len(parts) < 2:
                messages.append(system_struct("No subcategory name found after 'pick_subcategory'."))
                continue
            subcat_name = " ".join(parts[1:])
            cat = state.category_filter
            if not cat:
                messages.append(system_struct("No category chosen yet. Please pick_category first."))
                continue
            subcats_for_cat = subcategories_map[cat]
            if subcat_name not in subcats_for_cat:
                messages.append(system_struct(
                    f"Subcategory '{subcat_name}' not found under '{cat}'. Known subcats: {list(subcats_for_cat.keys())}"
                ))
                continue
            subcat_indices = subcats_for_cat[subcat_name]
            # Restrict candidate set further
            state.candidate_set = state.candidate_set.intersection(subcat_indices)
            state.subcategory_filter = subcat_name
            messages.append(system_struct(f"Picked subcategory '{subcat_name}'. Candidate set now has {len(state.candidate_set)} items."))
        
        elif action_text.startswith("lexical_search"):
            parts = action_text.split()
            if len(parts) < 2:
                messages.append(system_struct("No keyword provided after lexical_search."))
                continue
            keyword = " ".join(parts[1:])
            # Restrict search to candidate set:
            found = query_index(
                inverted_index, 
                keyword, 
                restrict_to=state.candidate_set
            )
            state.last_search_result = found
            messages.append(system_struct(f"Lexical search found {len(found)} items for '{keyword}'."))
        
        elif action_text.startswith("relevance_check"):
            # We do GPT-based checking of last_search_result (or candidate_set).
            # Usually you'd do a smaller chunk, but let's demonstrate the idea:
            to_check_indices = list(state.last_search_result)
            to_check_items = get_items_from_indices(items, to_check_indices)
            
            if not to_check_items:
                messages.append(system_struct("No items in last_search_result to check."))
                continue
            
            # In real usage, chunk the items or do them in smaller batches:
            relevant_list = review_items_with_gpt(to_check_items, user_request)
            # Move relevant items to positive set, irrelevant to negative set:
            relevant_indices = set()
            irrelevant_indices = set()
            for idx, is_rel in zip(to_check_indices, relevant_list):
                if is_rel:
                    relevant_indices.add(idx)
                else:
                    irrelevant_indices.add(idx)
            # Adjust state
            state.positive_set |= relevant_indices
            state.candidate_set -= relevant_indices
            state.negative_set |= irrelevant_indices
            state.candidate_set -= irrelevant_indices
            
            messages.append(system_struct(
                f"Relevance check done. Found {len(relevant_indices)} relevant, {len(irrelevant_indices)} irrelevant."
            ))
            
        elif action_text.startswith("union_positive"):
            # Move last_search_result to positive
            new_pos = state.last_search_result
            state.positive_set |= new_pos
            state.candidate_set -= new_pos
            messages.append(system_struct(
                f"Union with positive done. Positive set now has {len(state.positive_set)} items."
            ))
            
        elif action_text.startswith("subtract_negative"):
            # Move last_search_result to negative
            new_neg = state.last_search_result
            state.negative_set |= new_neg
            # remove them from positive/candidate
            state.positive_set -= new_neg
            state.candidate_set -= new_neg
            messages.append(system_struct(
                f"Subtract with negative done. Negative set now has {len(state.negative_set)} items."
            ))
        
        elif action_text.startswith("sample"):
            parts = action_text.split()
            if len(parts) != 3:
                messages.append(system_struct("Usage: sample <positive|negative|candidate> <number>."))
                continue
            which_set = parts[1]
            count = int(parts[2])
            if which_set not in ["positive", "negative", "candidate"]:
                messages.append(system_struct("Invalid set name. Must be 'positive', 'negative', or 'candidate'."))
                continue
            
            if which_set == "positive":
                s = state.positive_set
            elif which_set == "negative":
                s = state.negative_set
            else:
                s = state.candidate_set
            
            sample_indices = sample_from_set(s, count)
            if not sample_indices:
                messages.append(system_struct(f"No items in {which_set} to sample."))
                continue
            
            sampled_items = get_items_from_indices(items, sample_indices)
            snippet_text = []
            for idx, itm in zip(sample_indices, sampled_items):
                snippet = itm.get('metadata', '')[:100].replace("\n", " ")
                snippet_text.append(f"Index: {idx} | {snippet}...")
            
            # Provide that snippet to the assistant
            messages.append(system_struct(f"Sample from {which_set}:\n" + "\n".join(snippet_text)))
        
        elif action_text.startswith("stop"):
            messages.append(system_struct("Stopping the iteration."))
            break
        
        else:
            # Unrecognized action
            messages.append(system_struct(f"Unrecognized action: {action_text}"))
            continue
    
    # Final state
    print("\nFinal state:", state)
    return state

############################################
# 7. Putting It All Together
############################################

def main():
    data_pkl = Path('cache/queries_item_pool.pkl')
    item_pool, queries = prepare_item_query(data_pkl)
    
    # Build indexes
    category_index = build_category_index(item_pool)
    subcategories_map = discover_subcategories(item_pool, category_index)
    inverted_index = build_inverted_index(item_pool)
    
    # For demonstration, pick the first query from the dataset:
    user_request = queries[0]['query']
    ground_truth = queries[0]['item_id']
    
    # Run the iterative agent loop:
    final_state = run_agent_iteration(
        items=item_pool,
        category_index=category_index,
        subcategories_map=subcategories_map,
        inverted_index=inverted_index,
        user_request=user_request,
        steps=10
    )
    
    # Show results
    print("Done. Final positive set size:", len(final_state.positive_set))
    # Optionally, compare final_state.positive_set with ground_truth if you like.

if __name__ == "__main__":
    main()
