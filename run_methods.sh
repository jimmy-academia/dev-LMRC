#!/bin/bash

# Set common parameters
VERBOSITY=1
MODEL="gpt-4o-mini"

# Run the baseline methods
echo "===== Running baseline methods ====="

echo "Running direct similarity baseline (sim_query_item)..."
python main.py --app sim_query_item --verbose $VERBOSITY --model $MODEL

echo "Running LLM-enhanced similarity baseline (sim_llm_item)..."
python main.py --app sim_llm_item --verbose $VERBOSITY --model $MODEL

echo "Running Chain-of-Thought enhanced similarity baseline (sim_cot_item)..."
python main.py --app sim_cot_item --verbose $VERBOSITY --model $MODEL

# Run the original methods
# echo "===== Running main methods ====="

# echo "Running oneshot method..."
# python main.py --app oneshot --verbose $VERBOSITY --model $MODEL

# echo "Running multistep method..."
# python main.py --app multistep --verbose $VERBOSITY --model $MODEL

echo "All methods completed."

# Print summary of results
echo "===== Results Summary ====="
echo "Check the output directories for detailed results:"
echo "- app/baseline/output/ for baseline methods"
echo "- app/oneshot/output/ for oneshot method"
echo "- app/multistep/output/ for multistep method"