#!/bin/bash
# Usage: ./reproduce.sh
# Reproduces the add descriptions section of our paper.

set -e
cd "$(dirname "$0")"

python3 -m venv venv_auto-circuits
source venv_auto-circuits/bin/activate
pip install .

cd custom_automation
