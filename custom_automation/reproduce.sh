#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Run the custom_automation pipeline.
#
# Usage:
#   ./reproduce.sh              Run full pipeline (steps 0-5, no validation)
#   ./reproduce.sh core         Same as above — generate groups + push to viewer
#   ./reproduce.sh all          Core + validation (step 6)
#   ./reproduce.sh validate     Validation only (step 6, assumes core already ran)
#   ./reproduce.sh PRUNING_THRESHOLD      Core with custom pruning threshold
#
#   Ex: ./reproduce.sh all 0.35   All with custom pruning threshold
#
# Environment variables:
#   OPENAI_API_KEY   — Required for descriptions, grouping, and validation.
#   VIEWER_URL       — Optional. Base URL for the viewer (default: localhost:8041).
#
# After running: just refresh the viewer page. Groups load automatically.
# You might have to Clear Browser Cache via Inspect Element
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Parse subcommand
# ---------------------------------------------------------------------------
SUBCOMMAND="${1:-core}"

# If the first arg looks like a number, treat it as pruning threshold (backward compat)
if [[ "$SUBCOMMAND" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    export PRUNING_THRESHOLD="$SUBCOMMAND"
    SUBCOMMAND="core"
    shift || true
else
    shift || true
    export PRUNING_THRESHOLD="${1:-0.40}"
    shift || true
fi

RUN_CORE=false
RUN_VALIDATE=false

case "$SUBCOMMAND" in
    core)
        RUN_CORE=true
        ;;
    all)
        RUN_CORE=true
        RUN_VALIDATE=true
        ;;
    validate)
        RUN_VALIDATE=true
        ;;
    *)
        echo "Unknown subcommand: $SUBCOMMAND"
        echo "Usage: ./reproduce.sh [core|all|validate] [PRUNING_THRESHOLD]"
        exit 1
        ;;
esac

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

if ! command -v python &>/dev/null; then
    echo "ERROR: python not found on PATH." >&2
    exit 1
fi

python -c "import requests, openai" 2>/dev/null || {
    echo "ERROR: Missing Python dependencies (requests, openai)." >&2
    echo "       Install with:  pip install requests openai" >&2
    exit 1
}

mkdir -p artifacts

echo "  ✓ Python and dependencies OK"
echo "  ✓ artifacts/ directory ready"
echo "  ✓ PRUNING_THRESHOLD=${PRUNING_THRESHOLD}"
echo "  ✓ Mode: $SUBCOMMAND"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo -n "🔑 OPENAI_API_KEY not found. Please enter it now (input will be hidden): "
    read -s USER_API_KEY
    echo ""
    
    if [ -z "$USER_API_KEY" ]; then
        echo "  ERROR: API key cannot be empty. Exiting." >&2
        exit 1
    fi
    
    export OPENAI_API_KEY="$USER_API_KEY"
    echo "  ✓ OPENAI_API_KEY exported for this run."
else
    echo "  ✓ OPENAI_API_KEY is already set in the environment."
fi

# ---------------------------------------------------------------------------
# Core pipeline (steps 0-5)
# ---------------------------------------------------------------------------
if [ "$RUN_CORE" = true ]; then

    # Step 0
    step_banner "Step 0 — Applying frontend patch (if needed)"
    python apply_frontend_patch.py

    # Step 1
    step_banner "Step 1/5 — Fetching pruned feature activations"
    t=$(date +%s)
    python fetch_all_activation_text.py
    elapsed "$t"

    # Step 2
    step_banner "Step 2/5 — Generating descriptions (OpenAI GPT-5-mini)"
    t=$(date +%s)
    python generate_description.py
    elapsed "$t"

    # Step 3
    if [ "${SKIP_GROUPING:-0}" = "1" ]; then
        step_banner "Step 3/5 — Grouping SKIPPED (SKIP_GROUPING=1)"
    else
        step_banner "Step 3/5 — Grouping features into supernodes (OpenAI)"
        t=$(date +%s)
        python generate_supernodes.py
        elapsed "$t"
    fi

    # Step 4
    step_banner "Step 4/5 — Writing groups into graph JSON"
    t=$(date +%s)
    python push_to_website.py
    elapsed "$t"

    # Step 5
    step_banner "Step 5/5 — Registering graph in viewer dropdown"
    t=$(date +%s)
    python update_metadata.py
    elapsed "$t"

fi

# ---------------------------------------------------------------------------
# Validation (step 6)
# ---------------------------------------------------------------------------
if [ "$RUN_VALIDATE" = true ]; then
    step_banner "Step 6 — Validating groups (M1: feature ID, M2: text match)"
    t=$(date +%s)
    python validate_groups.py
    elapsed "$t"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step_banner "SUCCESS — Pipeline complete ($SUBCOMMAND)"
echo ""
if [ "$RUN_CORE" = true ]; then
    echo "  Your groups are saved in the graph JSON."
    echo "  The graph is registered in the viewer dropdown."
    echo "  👉 Just refresh the viewer page — select your graph, groups load automatically."
    echo "  Clear browser cache if regenerating a same prompt graph again. Unnecessary if first time."
    echo ""
fi
if [ "$RUN_VALIDATE" = true ]; then
    echo "  Validation report: artifacts/validation_report.json"
    echo ""
fi