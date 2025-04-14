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

def create_llm_client(keypath=".openaikey", model="gpt-4o-mini", temperature=0):
    """Create a function to call the LLM with updated OpenAI API."""
    client = OpenAI(api_key=readf(keypath).strip())
    
    def call_llm(messages, model=model, temperature=temperature, max_tokens=500):
        """Call the LLM with a prompt and return the response."""
        try:
            response = client.chat.completions.create(
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