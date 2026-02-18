#!/bin/bash
# Usage: ./reproduce.sh 0.4
# Reproduces the add descriptions section of our paper.

set -e
cd "$(dirname "$0")"

export PRUNING_THRESHOLD="${1:-0.40}"

# 2. Descriptions for each feature.
echo "2_A. Fetching Features with influence <= PRUNING_THRESHOLD"
echo "------------------------------------------------"
python fetch_all_activating_text.py

echo "------------------------------------------------"
echo "2_B. Generating Descriptions - Transluce Llama"
echo "------------------------------------------------"
python add_description.py

echo "------------------------------------------------"
echo "2_C. Merging to Graph (Update Website)"
echo "------------------------------------------------"
python push_to_website.py

echo "------------------------------------------------"
echo "SUCCESS! Step 2 finished."
echo "Refresh your localhost and clear browser cache to see updates."

# 3. Grouping features
