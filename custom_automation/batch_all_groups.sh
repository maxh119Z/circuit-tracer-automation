#!/usr/bin/env bash
# =============================================================================
# batch_all_groups.sh — Run the full pipeline for multiple prompts × grouping
#                       variant range.
#
# For each prompt in the CSV, runs Steps 1-2 once (shared fetch + descriptions),
# then Steps 3-5 (supernodes, push, register) for every requested grouping
# variant. Produces one viewer entry per (prompt × grouping_variant) combo.
#
# Usage:
#   ./batch_all_groups.sh <prompts.csv> [GROUPING_VARIANTS] [core|all|validate] [PRUNING_THRESHOLD]
#
# GROUPING_VARIANTS — comma-separated list or dash-range of grouping variants:
#   a0,a1,a2,a3   explicit list  (default)
#   a0-a3         range (expands to a0 a1 a2 a3)
#   a0,a3         just those two
#
# Examples:
#   ./batch_all_groups.sh prompts.csv
#   ./batch_all_groups.sh prompts.csv a0,a3
#   ./batch_all_groups.sh prompts.csv a0-a3 all 0.40
#   DESCRIPTION_VARIANT=v2 ./batch_all_groups.sh prompts.csv a0,a2
#
# CSV format (header row required, transcoder_set column optional):
#   slug,prompt[,transcoder_set]
#
# Environment variables:
#   OPENAI_API_KEY      — Required for descriptions, grouping, and validation.
#   DESCRIPTION_VARIANT — Which description variant to use (default: v2).
#   VIEWER_URL          — Optional. Base URL for the viewer (default: localhost:8041).
#
# Output per (prompt, grouping_variant) combo:
#   artifacts/<slug>-<desc>-<gvar>/              All variant-specific artifacts
#   test_graphs/<slug>-<desc>-<gvar>.json        Graph registered in viewer dropdown
#
# Shared intermediate artifacts (retained for inspection / re-runs):
#   artifacts/<slug>/pruned_activations.json
#   artifacts/<slug>/feature_descriptions_<desc>.json
#
# After running: refresh the viewer page and use the dropdown to switch graphs.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
CSV_FILE="${1:?Usage: ./batch_all_groups.sh <prompts.csv> [GROUPING_VARIANTS] [core|all|validate] [PRUNING_THRESHOLD]}"
RAW_VARIANTS="${2:-a0,a1,a2,a3}"
SUBCOMMAND="${3:-core}"
export PRUNING_THRESHOLD="${4:-0.40}"
export DESCRIPTION_VARIANT="${DESCRIPTION_VARIANT:-v2}"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: CSV file not found: $CSV_FILE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse grouping variant spec → array
# Supports:
#   Range:          "a0-a3"     → (a0 a1 a2 a3)
#   Comma-list:     "a0,a2,a3"  → (a0 a2 a3)
# ---------------------------------------------------------------------------
parse_variants() {
    local raw="$1"
    if [[ "$raw" =~ ^a([0-9]+)-a([0-9]+)$ ]]; then
        local start="${BASH_REMATCH[1]}"
        local end="${BASH_REMATCH[2]}"
        local out=()
        for (( i=start; i<=end; i++ )); do
            out+=("a${i}")
        done
        echo "${out[@]}"
    else
        echo "${raw//,/ }"
    fi
}

read -ra GROUPING_VARIANTS <<< "$(parse_variants "$RAW_VARIANTS")"

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

mkdir -p artifacts

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
# Summary
# ---------------------------------------------------------------------------
TOTAL=$(tail -n +2 "$CSV_FILE" | grep -c '[^[:space:]]' || true)
N_VARIANTS="${#GROUPING_VARIANTS[@]}"
echo ""
echo "Batch × Groups pipeline"
echo "  Prompts:           $TOTAL (from $CSV_FILE)"
echo "  Description:       $DESCRIPTION_VARIANT"
echo "  Grouping variants: ${GROUPING_VARIANTS[*]}"
echo "  Mode:              $SUBCOMMAND | Pruning: $PRUNING_THRESHOLD"
echo "  Total graph runs:  $(( TOTAL * N_VARIANTS ))"
echo ""

# ---------------------------------------------------------------------------
# Step 0 — Frontend patch (once for the whole batch)
# ---------------------------------------------------------------------------
step_banner "Step 0 — Applying frontend patch (once)"
python apply_frontend_patch.py

# ---------------------------------------------------------------------------
# Main loop: prompts
# ---------------------------------------------------------------------------
ROW_NUM=0
while IFS=',' read -r slug prompt _rest || [[ -n "$slug" ]]; do
    # Skip header and blank lines
    [[ "$slug" == "slug" ]] && continue
    [[ -z "${slug// }" ]] && continue

    ROW_NUM=$(( ROW_NUM + 1 ))
    slug="${slug// /}"   # trim whitespace

    step_banner "[$ROW_NUM/$TOTAL] Prompt: $slug"
    echo "  Text: $prompt"

    GRAPH_JSON="../test_graphs/${slug}.json"
    if [[ ! -f "$GRAPH_JSON" ]]; then
        echo "  WARNING: Graph file not found: $GRAPH_JSON" >&2
        echo "  Run 'circuit-tracer attribute-batch' first to generate the attribution graph." >&2
        echo "  Skipping $slug." >&2
        continue
    fi

    if [ "$RUN_CORE" = true ]; then

        # ------------------------------------------------------------------
        # Steps 1–2: fetch activations + descriptions (shared across grouping
        # variants for this prompt). Artifacts land in artifacts/<slug>/.
        # ------------------------------------------------------------------
        export CURRENT_SLUG="$slug"
        mkdir -p "artifacts/$slug"

        step_banner "[$ROW_NUM/$TOTAL] Step 1 — Fetch activations (shared)"
        t=$(date +%s); python fetch_all_activation_text.py; elapsed "$t"

        step_banner "[$ROW_NUM/$TOTAL] Step 2 — Generate descriptions [desc=$DESCRIPTION_VARIANT] (shared)"
        t=$(date +%s); python generate_description.py; elapsed "$t"

        # ------------------------------------------------------------------
        # Inner loop: grouping variants
        # ------------------------------------------------------------------
        for gvar in "${GROUPING_VARIANTS[@]}"; do
            VARIANT_SLUG="${slug}-${DESCRIPTION_VARIANT}-${gvar}"
            export GROUPING_VARIANT="$gvar"
            export CURRENT_SLUG="$VARIANT_SLUG"

            mkdir -p "artifacts/$VARIANT_SLUG"

            # Copy shared artifacts into this variant's directory
            cp "artifacts/${slug}/pruned_activations.json" \
               "artifacts/${VARIANT_SLUG}/pruned_activations.json"
            cp "artifacts/${slug}/feature_descriptions_${DESCRIPTION_VARIANT}.json" \
               "artifacts/${VARIANT_SLUG}/feature_descriptions_${DESCRIPTION_VARIANT}.json"

            # Copy base graph so push_to_website.py writes to the right file
            cp "$GRAPH_JSON" "../test_graphs/${VARIANT_SLUG}.json"

            step_banner "[$ROW_NUM/$TOTAL | $DESCRIPTION_VARIANT/$gvar] Step 3 — Generate supernodes"
            t=$(date +%s); python generate_supernodes.py; elapsed "$t"

            step_banner "[$ROW_NUM/$TOTAL | $DESCRIPTION_VARIANT/$gvar] Step 4 — Push to graph JSON"
            t=$(date +%s); python push_to_website.py; elapsed "$t"

            step_banner "[$ROW_NUM/$TOTAL | $DESCRIPTION_VARIANT/$gvar] Step 5 — Register in viewer"
            t=$(date +%s); python update_metadata.py; elapsed "$t"

            if [ "$RUN_VALIDATE" = true ]; then
                step_banner "[$ROW_NUM/$TOTAL | $DESCRIPTION_VARIANT/$gvar] Step 6 — Validate groups"
                t=$(date +%s); python validate_groups.py; elapsed "$t"
            fi

            echo "  [$VARIANT_SLUG] artifacts/${VARIANT_SLUG}/"
            echo "  [$VARIANT_SLUG] test_graphs/${VARIANT_SLUG}.json"
        done

    fi

    echo ""
    echo "  [$ROW_NUM/$TOTAL] $slug complete — $N_VARIANTS variant(s) registered."

done < "$CSV_FILE"

# ---------------------------------------------------------------------------
# Aggregate validation across all prompts × variants
# ---------------------------------------------------------------------------
if [ "$RUN_VALIDATE" = true ]; then
    step_banner "Aggregating validation scores across all prompts × variants"
    python aggregate_batch.py
    echo "  Batch summary → artifacts/batch_summary.md"
fi

unset CURRENT_SLUG GROUPING_VARIANT

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step_banner "SUCCESS — Batch × Groups pipeline complete"
echo ""
echo "  $ROW_NUM prompt(s) × $N_VARIANTS grouping variant(s) processed."
echo "  Graph naming: <slug>-${DESCRIPTION_VARIANT}-<grouping-variant>"
echo "  Refresh the viewer and use the dropdown to switch between graphs."
if [ "$RUN_VALIDATE" = true ]; then
    echo "  Validation summary → artifacts/batch_summary.md"
fi
echo ""
