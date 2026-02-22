#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Run the full custom_automation pipeline.
#
# Usage:
#   ./reproduce.sh [PRUNING_THRESHOLD]   (default: 0.40)
#
# Environment variables:
#   OPENAI_API_KEY   — Required for Step 3 (grouping).
#   VIEWER_URL       — Optional. Base URL for the viewer (default: localhost:8041).
#   SKIP_GROUPING    — Set to 1 to skip Step 3 (if no OpenAI key available).
#
# Steps:
#   1. Fetch pruned feature activations from HuggingFace.
#   2. Generate descriptions with the Transluce Llama explainer.
#   3. Group features into supernodes via OpenAI.
#   4. Push descriptions + groups into the graph and generate viewer URL.
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

if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "  ✓ OPENAI_API_KEY is set"
elif [ "${SKIP_GROUPING:-0}" = "1" ]; then
    echo "  ⚠ OPENAI_API_KEY not set — Step 3 (grouping) will be skipped"
else
    echo "  ⚠ OPENAI_API_KEY not set — Step 3 will fail unless you set SKIP_GROUPING=1"
fi

# ---------------------------------------------------------------------------
# Step 1 — Fetch activations
# ---------------------------------------------------------------------------
step_banner "Step 1/4 — Fetching pruned feature activations"
t=$(date +%s)
python fetch_all_activation_text.py
elapsed "$t"

# ---------------------------------------------------------------------------
# Step 2 — Generate descriptions
# ---------------------------------------------------------------------------
step_banner "Step 2/4 — Generating descriptions (Transluce Llama)"
t=$(date +%s)
python generate_description.py
elapsed "$t"

# ---------------------------------------------------------------------------
# Step 3 — Group features (requires OpenAI)
# ---------------------------------------------------------------------------
if [ "${SKIP_GROUPING:-0}" = "1" ]; then
    step_banner "Step 3/4 — Grouping SKIPPED (SKIP_GROUPING=1)"
else
    step_banner "Step 3/4 — Grouping features into supernodes (OpenAI)"
    t=$(date +%s)
    python generate_supernodes.py
    elapsed "$t"
fi

# ---------------------------------------------------------------------------
# Step 4 — Push to graph & generate viewer URL
# ---------------------------------------------------------------------------
step_banner "Step 4/4 — Pushing to graph & generating viewer URL"
t=$(date +%s)
python push_to_website.py
elapsed "$t"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step_banner "SUCCESS — Pipeline complete"
echo "  Check artifacts/viewer_url.txt for the full viewer URL."
echo "  Or copy the URL printed above into your browser."
echo ""