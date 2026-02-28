#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Run the full custom_automation pipeline.
#
# Usage:
#   ./reproduce.sh [PRUNING_THRESHOLD]   (default: 0.40)
#
# Environment variables:
#   OPENAI_API_KEY   — Required for Step 2 (descriptions) and Step 3 (grouping).
#                      (If not set, the script will prompt you for it).
#   VIEWER_URL       — Optional. Base URL for the viewer (default: localhost:8041).
#   SKIP_GROUPING    — Set to 1 to skip Step 3.
#
# Steps:
#   0. Apply frontend patch (idempotent — safe to run multiple times).
#   1. Fetch pruned feature activations from HuggingFace.
#   2. Generate descriptions with OpenAI GPT-5-mini (Asyncio).
#   3. Group features into supernodes via OpenAI.
#   4. Push descriptions + groups into the graph JSON (qParams).
#
# After running: just refresh the viewer page. Groups load automatically.
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

# Key packages
python -c "import requests, openai" 2>/dev/null || {
    echo "ERROR: Missing Python dependencies (requests, openai)." >&2
    echo "       Install with:  pip install requests openai" >&2
    exit 1
}

# Ensure artifacts directory exists
mkdir -p artifacts

echo "  ✓ Python and dependencies OK"
echo "  ✓ artifacts/ directory ready"
echo "  ✓ PRUNING_THRESHOLD=${PRUNING_THRESHOLD}"

# Enforce the API key: Prompt if not set, then export
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo -n "🔑 OPENAI_API_KEY not found. Please enter it now (input will be hidden): "
    read -s USER_API_KEY
    echo "" # Add a newline after silent input
    
    if [ -z "$USER_API_KEY" ]; then
        echo "  ERROR: API key cannot be empty. Exiting." >&2
        exit 1
    fi
    
    export OPENAI_API_KEY="$USER_API_KEY"
    echo "  ✓ OPENAI_API_KEY exported for this run."
else
    echo "  ✓ OPENAI_API_KEY is already set in the environment."
fi

if [ "${SKIP_GROUPING:-0}" = "1" ]; then
    echo "  ⚠ SKIP_GROUPING=1 — Step 3 will be skipped"
fi

# ---------------------------------------------------------------------------
# Step 0 — Apply frontend patch (idempotent)
# ---------------------------------------------------------------------------
step_banner "Step 0 — Applying frontend patch (if needed)"
python apply_frontend_patch.py

# ---------------------------------------------------------------------------
# Step 1 — Fetch activations
# ---------------------------------------------------------------------------
step_banner "Step 1/5 — Fetching pruned feature activations"
t=$(date +%s)
python fetch_all_activation_text.py
elapsed "$t"

# ---------------------------------------------------------------------------
# Step 2 — Generate descriptions
# ---------------------------------------------------------------------------
step_banner "Step 2/5 — Generating descriptions (OpenAI GPT-5-mini)"
t=$(date +%s)
python generate_description.py
elapsed "$t"

# ---------------------------------------------------------------------------
# Step 3 — Group features (requires OpenAI)
# ---------------------------------------------------------------------------
if [ "${SKIP_GROUPING:-0}" = "1" ]; then
    step_banner "Step 3/5 — Grouping SKIPPED (SKIP_GROUPING=1)"
else
    step_banner "Step 3/5 — Grouping features into supernodes (OpenAI)"
    t=$(date +%s)
    python generate_supernodes.py
    elapsed "$t"
fi

# ---------------------------------------------------------------------------
# Step 4 — Push to graph qParams
# ---------------------------------------------------------------------------
step_banner "Step 4/5 — Writing groups into graph JSON"
t=$(date +%s)
python push_to_website.py
elapsed "$t"

# ---------------------------------------------------------------------------
# Step 5 — Register in viewer dropdown
# ---------------------------------------------------------------------------
step_banner "Step 5/5 — Registering graph in viewer dropdown"
t=$(date +%s)
python update_metadata.py
elapsed "$t"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step_banner "SUCCESS — Pipeline complete"
echo ""
echo "  Your groups are saved in the graph JSON (qParams)."
echo "  The graph is registered in the viewer dropdown."
echo "  👉 Just refresh the viewer page — select your graph, groups load automatically."
echo ""
echo "  No URL copying needed!"
echo ""