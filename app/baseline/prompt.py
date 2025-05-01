"""
Prompts for enhancing queries in the similarity-based baseline methods.
"""

# Prompt for embedding enhancement with LLM
llm_enhance_prompt = """You are a search expert helping to improve product search.
Given a user query, rewrite it to better match product descriptions by:
1. Adding relevant keywords and attributes
2. Clarifying ambiguous terms
3. Expanding abbreviations
4. Including likely synonyms

Original query: "{query}"

Return the enhanced search query only, no explanations or formatting.
"""

# Prompt for embedding enhancement with chain-of-thought
cot_enhance_prompt = """You are a search expert helping to improve product search.
Think step by step to understand what the user is looking for, then rewrite the query to better match product descriptions.

Original query: "{query}"

First, analyze the query:
1. What is the core product type being requested?
2. What specific attributes or features are mentioned or implied?
3. What are the likely use cases or user needs?
4. What relevant details might be missing?

Based on this analysis, construct an enhanced query that includes all relevant information.

Your enhanced query: 
"""

# System prompt for all query enhancement
system_enhance = "You are an expert in product search query enhancement."