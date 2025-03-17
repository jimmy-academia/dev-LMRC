import os
import random
import logging

import openai

from pathlib import Path

user_struct = lambda x: {"role": "user", "content": x}
system_struct = lambda x: {"role": "system", "content": x}
assistant_struct = lambda x: {"role": "assistant", "content": x}

def flatten_messages(messages):
    flat = ''
    for msg in messages:
        flat += f"{msg['role']} => {msg['content']}\n"
    return flat

def readf(path):
    with open(path, 'r') as f:
        return f.read()

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def set_verbose(verbose):
    # usage: logging.warning; logging.error; logging.info; logging.debug
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    elif verbose == 2:
        level = logging.DEBUG
    else:
        level = logging.DEBUG  # fallback
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler()],
    )

def ensure_dir(_dir):
    """Ensure cache directory exists."""
    Path(_dir).mkdir(exist_ok=True)

def create_llm_client(keypath=".openaikey", model="gpt-4o", temperature=0):
    """Create a function to call the LLM."""
    openai.api_key = readf(keypath).strip()
    
    def call_llm(messages, model=model, temperature=temperature, max_tokens=500):
        """Call the LLM with a prompt and return the response."""
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            print(f"prompt messages: {messages}")
            return "Error processing request."
    
    return call_llm