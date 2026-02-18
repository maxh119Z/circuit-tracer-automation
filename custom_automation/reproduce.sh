#!/bin/bash
# Usage: HF_TOKEN="your_key_here" ./reproduce.sh 0.4
# Reproduces the add descriptions section of our paper.

set -e
cd "$(dirname "$0")"
export PRUNING_THRESHOLD="${1:-0.40}"

# 1. Setup virtual environment and install dependencies
python3 -m venv venv_auto-circuits
source venv_auto-circuits/bin/activate

# 1_A. Login to Huggingface for gated models.
echo "Logging in to Huggingface"
python huggingface_login.py
cd custom_automation


# 2. Descriptions for each feature.
echo "------------------------------------------------"

echo "2_A. Fetching Features with influence <= PRUNING_THRESHOLD"
echo "------------------------------------------------"
# This script downloads all feature information from HF based on your graph at specific pruning level.
python fetch_all_activating_text.py

echo "------------------------------------------------"
echo "2_B. Generating Descriptions - Transluce Llama"
echo "------------------------------------------------"
# This script runs the local LLM to describe features. A decent GPU is required to run the 8B model.
python add_description.py

echo "------------------------------------------------"
echo "2_C. Merging to Graph (Update Website)"
echo "------------------------------------------------"
# This script injects the descriptions into test-run.json where website features are stored. 
python push_to_website.py

echo "------------------------------------------------"
echo "SUCCESS! Step 2 finished."
echo "Refresh your localhost and clear browser cache to see updates."
