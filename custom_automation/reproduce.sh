#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Run the full custom_automation pipeline.
#
# Usage:
#   ./reproduce.sh [PRUNING_THRESHOLD]   (default: 0.40)
#
# Steps:
#   1. Fetch pruned feature activations from HuggingFace.
#   2. Generate descriptions with the Transluce Llama explainer.
#   3. Merge descriptions into the attribution graph for the viewer.
#   4. To be added: Grouping features.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

export PRUNING_THRESHOLD="${1:-0.40}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
timestamp() { date +"%H:%M:%S"; }

step_banner() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "  $(timestamp)"
    echo "========================================"
}

elapsed() {
    local start=$1
    local end
    end=$(date +%s)
    echo "  ↳ Completed in $(( end - start ))s"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
step_banner "Pre-flight checks"

# Python
if ! command -v python &>/dev/null; then
    echo "ERROR: python not found on PATH." >&2
    exit 1
fi

# Key packages (fail fast rather than mid-pipeline)
python -c "import requests, torch, transformers" 2>/dev/null || {
    echo "ERROR: Missing Python dependencies (requests, torch, transformers)." >&2
    echo "       Install with:  pip install requests torch transformers" >&2
    exit 1
}

# Ensure artifacts directory exists
mkdir -p artifacts
echo "  ✓ Python and dependencies OK"
echo "  ✓ artifacts/ directory ready"
echo "  ✓ PRUNING_THRESHOLD=${PRUNING_THRESHOLD}"

# ---------------------------------------------------------------------------
# Step 1 — Fetch activations
# ---------------------------------------------------------------------------
step_banner "Step 1/3 — Fetching pruned feature activations"
t=$(date +%s)
python fetch_all_activating_text.py
elapsed "$t"

# ---------------------------------------------------------------------------
# Step 2 — Generate descriptions
# ---------------------------------------------------------------------------
step_banner "Step 2/3 — Generating descriptions (Transluce Llama)"
t=$(date +%s)
python add_description.py
elapsed "$t"

# ---------------------------------------------------------------------------
# Step 3 — Merge into graph
# ---------------------------------------------------------------------------
step_banner "Step 3/3 — Merging descriptions into graph"
t=$(date +%s)
python push_to_website.py
elapsed "$t"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step_banner "SUCCESS — Pipeline complete"
echo "  Refresh localhost:8041 and clear browser cache to see updates."
echo ""
