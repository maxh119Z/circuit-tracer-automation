#!/usr/bin/env bash
# =============================================================================
# batch_reproduce.sh — Run the custom_automation pipeline for every row in a CSV.
#
# Usage:
#   ./batch_reproduce.sh <prompts.csv> [core|all|validate] [PRUNING_THRESHOLD]
#
# CSV format (header row required, transcoder_set column optional):
#   slug,prompt[,transcoder_set]
#
# Examples:
#   ./batch_reproduce.sh prompts.csv
#   ./batch_reproduce.sh prompts.csv all
#   ./batch_reproduce.sh prompts.csv all 0.40
#
# Environment variables:
#   OPENAI_API_KEY   — Required for descriptions, grouping, and validation.
#   VIEWER_URL       — Optional. Base URL for the viewer (default: localhost:8041).
#
# After running: refresh the viewer page and select a graph from the dropdown.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
CSV_FILE="${1:?Usage: ./batch_reproduce.sh <prompts.csv> [core|all|validate] [PRUNING_THRESHOLD]}"
SUBCOMMAND="${2:-core}"
export PRUNING_THRESHOLD="${3:-0.40}"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: CSV file not found: $CSV_FILE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if ! command -v python &>/dev/null; then
    echo "ERROR: python not found on PATH." >&2
    exit 1
fi

python -c "import requests, openai" 2>/dev/null || {
    echo "ERROR: Missing Python dependencies (requests, openai)." >&2
    echo "       Install with:  pip install requests openai" >&2
    exit 1
}

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo -n "OPENAI_API_KEY not found. Please enter it now (input will be hidden): "
    read -rs USER_API_KEY
    echo ""
    if [ -z "$USER_API_KEY" ]; then
        echo "ERROR: API key cannot be empty." >&2
        exit 1
    fi
    export OPENAI_API_KEY="$USER_API_KEY"
    echo "  OPENAI_API_KEY exported for this run."
else
    echo "  OPENAI_API_KEY is set."
fi

RUN_CORE=false
RUN_VALIDATE=false
case "$SUBCOMMAND" in
    core)     RUN_CORE=true ;;
    all)      RUN_CORE=true; RUN_VALIDATE=true ;;
    validate) RUN_VALIDATE=true ;;
    *)
        echo "Unknown subcommand: $SUBCOMMAND. Use core|all|validate." >&2
        exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
timestamp() { date +"%H:%M:%S"; }
step_banner() { echo ""; echo "========================================"; echo "  $1"; echo "  $(timestamp)"; echo "========================================"; }
elapsed() { local end; end=$(date +%s); echo "  Completed in $(( end - $1 ))s"; }

# ---------------------------------------------------------------------------
# Count rows (skip header)
# ---------------------------------------------------------------------------
TOTAL=$(tail -n +2 "$CSV_FILE" | grep -c '[^[:space:]]' || true)
echo ""
echo "Batch mode: $TOTAL prompt(s) from $CSV_FILE"
echo "Mode: $SUBCOMMAND | Pruning threshold: $PRUNING_THRESHOLD"
echo ""

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
ROW_NUM=0
while IFS=',' read -r slug prompt _rest || [[ -n "$slug" ]]; do
    # Skip header row
    [[ "$slug" == "slug" ]] && continue
    # Skip blank lines
    [[ -z "${slug// }" ]] && continue

    ROW_NUM=$(( ROW_NUM + 1 ))
    slug="${slug// /}"  # trim whitespace

    step_banner "[$ROW_NUM/$TOTAL] slug: $slug"
    echo "  Prompt: $prompt"

    export CURRENT_SLUG="$slug"
    mkdir -p "artifacts/$slug"

    GRAPH_JSON="../test_graphs/${slug}.json"
    if [[ ! -f "$GRAPH_JSON" ]]; then
        echo "  WARNING: Graph file not found: $GRAPH_JSON" >&2
        echo "  Run 'circuit-tracer attribute-batch' first to generate attribution graphs." >&2
        echo "  Skipping $slug." >&2
        continue
    fi

    if [ "$RUN_CORE" = true ]; then

        step_banner "[$ROW_NUM/$TOTAL] Step 0 — Frontend patch"
        python apply_frontend_patch.py

        step_banner "[$ROW_NUM/$TOTAL] Step 1 — Fetch activations"
        t=$(date +%s); python fetch_all_activation_text.py; elapsed "$t"

        step_banner "[$ROW_NUM/$TOTAL] Step 2 — Generate descriptions"
        t=$(date +%s); python generate_description.py; elapsed "$t"

        if [ "${SKIP_GROUPING:-0}" = "1" ]; then
            step_banner "[$ROW_NUM/$TOTAL] Step 3 — Grouping SKIPPED"
        else
            step_banner "[$ROW_NUM/$TOTAL] Step 3 — Generate supernodes"
            t=$(date +%s); python generate_supernodes.py; elapsed "$t"
        fi

        step_banner "[$ROW_NUM/$TOTAL] Step 4 — Push to graph JSON"
        t=$(date +%s); python push_to_website.py; elapsed "$t"

        step_banner "[$ROW_NUM/$TOTAL] Step 5 — Register in viewer dropdown"
        t=$(date +%s); python update_metadata.py; elapsed "$t"

    fi

    if [ "$RUN_VALIDATE" = true ]; then
        step_banner "[$ROW_NUM/$TOTAL] Step 6 — Validate groups"
        t=$(date +%s); python validate_groups.py; elapsed "$t"
    fi

    echo ""
    echo "  Done: $slug"

done < "$CSV_FILE"

# ---------------------------------------------------------------------------
# Aggregate validation across all prompts
# ---------------------------------------------------------------------------
if [ "$RUN_VALIDATE" = true ]; then
    step_banner "Aggregating validation scores across all prompts"
    python aggregate_batch.py
    echo "  Batch summary → artifacts/batch_summary.md"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step_banner "SUCCESS — Batch pipeline complete ($ROW_NUM slugs processed)"
echo ""
echo "  Refresh the viewer and use the dropdown to switch between graphs."
if [ "$RUN_VALIDATE" = true ]; then
    echo "  Validation summary → artifacts/batch_summary.md"
fi
echo ""
