import re
import os
import json
import random
import logging

from openai import OpenAI


from pathlib import Path

user_struct = lambda x: {"role": "user", "content": x}
system_struct = lambda x: {"role": "system", "content": x}
assistant_struct = lambda x: {"role": "assistant", "content": x}

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_verbose(verbose):
    """
    Configure logging level and suppress OpenAI HTTP request logs.
    
    Args:
        verbose (int): 0=WARNING, 1=INFO, 2=DEBUG
    """
    import logging
    
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set appropriate level based on verbosity
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    else:
        level = logging.DEBUG  # fallback
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()],
    )
    
    # Suppress all OpenAI-related logs more thoroughly
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("openai.http_client").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)  # OpenAI uses httpx
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    
    # Disable propagation for OpenAI loggers
    logging.getLogger("openai").propagate = False
    logging.getLogger("openai.http_client").propagate = False
    logging.getLogger("httpx").propagate = False

# ====

def readf(path):
    with open(path, 'r') as f:
        return f.read()

class NamespaceEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, argparse.Namespace):
      return obj.__dict__
    else:
      return super().default(obj)

def dumpj(dictionary, filepath):
    with open(filepath, "w") as f:
        obj = json.dumps(dictionary, indent=4, cls=NamespaceEncoder)
        obj = re.sub(r'("|\d+),\s+', r'\1, ', obj)
        obj = re.sub(r'\[\n\s*("|\d+)', r'[\1', obj)
        obj = re.sub(r'("|\d+)\n\s*\]', r'\1]', obj)
        f.write(obj)

def loadj(filepath):
    with open(filepath) as f:
        return json.load(f)

def ensure_dir(_dir):
    """Ensure cache directory exists."""
    Path(_dir).mkdir(exist_ok=True)

def create_llm_client(keypath=".openaikey", model="gpt-4.1-nano", temperature=0):
    """Create a function to call the LLM with updated OpenAI API, tracking usage and costs."""
    client = OpenAI(api_key=readf(keypath).strip())
    
    # Initialize token counters
    usage_tracker = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "cost": 0.0
    }
    
    # Cost rates per 1000 tokens (as of April 2025)
    # Updated based on current OpenAI pricing
    model_costs = {
        # Standard models
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        
        # GPT-4o models
        "gpt-4o": {"prompt": 0.003, "completion": 0.01},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        
        # GPT-4.1 models (newest)
        "gpt-4.1": {"prompt": 0.002, "completion": 0.008},
        "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
        "gpt-4.1-nano": {"prompt": 0.0001, "completion": 0.0004}
    }
    
    def calculate_cost(model_name, prompt_tokens, completion_tokens):
        """Calculate cost based on model and token usage."""
        if model_name not in model_costs:
            print(f"Warning: Cost data not available for {model_name}, using gpt-4o rates")
            model_name = "gpt-4o"  # Default to gpt-4o rates if model not found
            
        costs = model_costs[model_name]
        prompt_cost = (prompt_tokens / 1000) * costs["prompt"]
        completion_cost = (completion_tokens / 1000) * costs["completion"]
        return prompt_cost + completion_cost
    
    def call_llm(messages, model=model, temperature=temperature, max_tokens=500):
        """Call the LLM with a prompt and return the response, tracking usage."""
        nonlocal usage_tracker
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Update token usage
            p_tokens = response.usage.prompt_tokens
            c_tokens = response.usage.completion_tokens
            t_tokens = response.usage.total_tokens
            
            usage_tracker["prompt_tokens"] += p_tokens
            usage_tracker["completion_tokens"] += c_tokens
            usage_tracker["total_tokens"] += t_tokens
            usage_tracker["calls"] += 1
            
            # Calculate and update cost
            call_cost = calculate_cost(model, p_tokens, c_tokens)
            usage_tracker["cost"] += call_cost
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            print(f"prompt messages: {messages}")
            return "Error processing request."
    
    # Add method to get usage statistics
    def get_usage():
        """Return current usage statistics."""
        return usage_tracker.copy()
    
    # Add method to reset usage statistics
    def reset_usage():
        """Reset all usage statistics to zero."""
        nonlocal usage_tracker
        usage_tracker = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "cost": 0.0
        }
        return {"message": "Usage statistics have been reset"}
    
    # Attach methods to the call_llm function
    call_llm.get_usage = get_usage
    call_llm.reset_usage = reset_usage
    
    return call_llm

def parse_llm_output(self, llm_text):
        """Parse the LLM output as JSON."""
        try:
            text = llm_text.strip()
            # Remove Markdown code fences if present
            if text.startswith("```"):
                # Remove the first line if it starts with ```
                text = "\n".join(text.split("\n")[1:])
                # Remove the last line if it ends with ```
                if text.endswith("```"):
                    text = "\n".join(text.split("\n")[:-1])
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            return None
