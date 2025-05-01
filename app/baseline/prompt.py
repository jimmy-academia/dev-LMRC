"""
Prompts for enhancing queries in the similarity-based baseline methods.
"""

# System prompt for query enhancement
system_enhance = "You are an expert in product search query enhancement."

# System prompt for query extraction
system_extract = "You extract specific information from text."

# Prompt for simple LLM query enhancement
llm_enhance_prompt = """Rewrite this product search query to better match product descriptions by adding relevant keywords, clarifying terms, and including likely synonyms.

Original query: "{query}"

Return only the enhanced query without explanations.
"""

# Prompt for chain-of-thought query enhancement
cot_enhance_prompt = """Analyze this product search query step-by-step, then rewrite it to better match product descriptions.

Original query: "{query}"

Follow these steps:
1. Identify the core product type being requested
2. List the attributes or features mentioned or implied
3. Consider the likely use case or intent
4. Note any relevant details that might be missing
5. Based on your analysis, provide an enhanced search query

Format your response like this:
Core Product: [what the user is looking for]
Attributes: [list of relevant attributes]
Use Case: [likely intended use]
Missing Details: [what could make the search better]

Enhanced Query: [your final enhanced query]

Make sure your enhanced query is comprehensive and includes all relevant terms to improve search accuracy.
"""

# Prompt for extracting the enhanced query from CoT output
query_extraction_prompt = """
The following is a chain-of-thought analysis of a product search query, followed by an enhanced query.
Extract ONLY the final enhanced query from this text.

Text to extract from:
{cot_response}

Return only the enhanced query, nothing else.
"""