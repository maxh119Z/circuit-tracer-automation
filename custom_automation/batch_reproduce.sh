#!/usr/bin/env bash
# =============================================================================
# batch_reproduce.sh — Run the custom_automation pipeline for every row in a CSV.
#
# Usage:
#   ./batch_reproduce.sh <prompts.csv> [core|all|validate] [PRUNING_THRESHOLD]
#
# To run all grouping variants per prompt (replaces batch_all_groups.sh):
#   GROUPING_VARIANTS=a0,a2,a3 ./batch_reproduce.sh prompts.csv
#   GROUPING_VARIANTS=a0-a3    ./batch_reproduce.sh prompts.csv all
#
# CSV format (header row required, transcoder_set column optional):
#   slug,prompt[,transcoder_set]
#
# Examples:
#   ./batch_reproduce.sh prompts.csv
#   ./batch_reproduce.sh prompts.csv all
#   ./batch_reproduce.sh prompts.csv all 0.40
#   GROUPING_VARIANTS=a0,a2,a3 ./batch_reproduce.sh prompts.csv
#   GROUPING_VARIANTS=a0-a3 ./batch_reproduce.sh prompts.csv all
#
# Environment variables:
#   OPENAI_API_KEY      — Required for descriptions, grouping, and validation.
#   DESCRIPTION_VARIANT — Which description variant to use (default: v2).
#   GROUPING_VARIANTS   — Optional. Comma-separated or range (e.g. a0,a2,a3 or a0-a3).
#                         When set, nests a loop over grouping variants per prompt.
#                         When unset, uses GROUPING_VARIANT (default: a0) as single variant.
#   VIEWER_URL          — Optional. Base URL for the viewer (default: localhost:8041).
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
export DESCRIPTION_VARIANT="${DESCRIPTION_VARIANT:-v2}"

if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: CSV file not found: $CSV_FILE" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Parse grouping variant spec → array
# Supports:
#   Range:      "a0-a3"    → (a0 a1 a2 a3)
#   Comma-list: "a0,a2,a3" → (a0 a2 a3)
# ---------------------------------------------------------------------------
parse_variants() {
    local raw="$1"
    if [[ "$raw" =~ ^a([0-9]+)-a([0-9]+)$ ]]; then
        local start="${BASH_REMATCH[1]}" end="${BASH_REMATCH[2]}" out=()
        for (( i=start; i<=end; i++ )); do out+=("a${i}"); done
        echo "${out[@]}"
    else
        echo "${raw//,/ }"
    fi
}

# Multi-variant mode when GROUPING_VARIANTS is set; single-variant otherwise.
if [[ -n "${GROUPING_VARIANTS:-}" ]]; then
    read -ra VARIANT_LIST <<< "$(parse_variants "$GROUPING_VARIANTS")"
    MULTI_VARIANT=true
else
    VARIANT_LIST=("${GROUPING_VARIANT:-a2}")
    MULTI_VARIANT=false
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
# Summary
# ---------------------------------------------------------------------------
TOTAL=$(tail -n +2 "$CSV_FILE" | grep -c '[^[:space:]]' || true)
N_VARIANTS="${#VARIANT_LIST[@]}"
echo ""
echo "Batch pipeline"
echo "  Prompts:           $TOTAL (from $CSV_FILE)"
echo "  Description:       $DESCRIPTION_VARIANT"
if [ "$MULTI_VARIANT" = true ]; then
    echo "  Grouping variants: ${VARIANT_LIST[*]} (multi-variant mode)"
    echo "  Total graph runs:  $(( TOTAL * N_VARIANTS ))"
else
    echo "  Grouping variant:  ${VARIANT_LIST[0]}"
fi
echo "  Mode:              $SUBCOMMAND | Pruning: $PRUNING_THRESHOLD"
echo ""

# ---------------------------------------------------------------------------
# Step 0 — Frontend patch (once for the whole batch)
# ---------------------------------------------------------------------------
step_banner "Step 0 — Applying frontend patch (once)"
python pipeline/apply_frontend_patch.py

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
ROW_NUM=0
while IFS=',' read -r slug prompt _rest || [[ -n "$slug" ]]; do
    [[ "$slug" == "slug" ]] && continue
    [[ -z "${slug// }" ]] && continue

    ROW_NUM=$(( ROW_NUM + 1 ))
    slug="${slug// /}"

    step_banner "[$ROW_NUM/$TOTAL] Prompt: $slug"
    echo "  Text: $prompt"

    GRAPH_JSON="../test_graphs/${slug}.json"
    if [[ ! -f "$GRAPH_JSON" ]]; then
        echo "  WARNING: Graph file not found: $GRAPH_JSON" >&2
        echo "  Run './attribute-batch.sh' first to generate attribution graphs." >&2
        echo "  Skipping $slug." >&2
        continue
    fi

    if [ "$RUN_CORE" = true ]; then

        # ------------------------------------------------------------------
        # Steps 1–2: shared per prompt (regardless of variant count)
        # ------------------------------------------------------------------
        export CURRENT_SLUG="$slug"
        mkdir -p "artifacts/$slug"

        DESC_ARTIFACT="artifacts/${slug}/feature_descriptions_${DESCRIPTION_VARIANT}.json"
        if [[ -f "$DESC_ARTIFACT" ]]; then
            echo "  SKIP Steps 1-2 — descriptions already exist for $slug"
        else
            step_banner "[$ROW_NUM/$TOTAL] Step 1 — Fetch activations (shared)"
            t=$(date +%s); python pipeline/fetch_all_activation_text.py; elapsed "$t"

            step_banner "[$ROW_NUM/$TOTAL] Step 2 — Generate descriptions [desc=$DESCRIPTION_VARIANT] (shared)"
            t=$(date +%s); python pipeline/generate_description.py; elapsed "$t"
        fi

        # ------------------------------------------------------------------
        # Steps 3–5 (+ optional 6): once per variant
        # ------------------------------------------------------------------
        for gvar in "${VARIANT_LIST[@]}"; do
            export GROUPING_VARIANT="$gvar"

            if [ "$MULTI_VARIANT" = true ]; then
                VARIANT_SLUG="${slug}-${DESCRIPTION_VARIANT}-${gvar}"
            else
                VARIANT_SLUG="$slug"
            fi
            export CURRENT_SLUG="$VARIANT_SLUG"

            GROUPS_ARTIFACT="artifacts/${VARIANT_SLUG}/feature_groups_${DESCRIPTION_VARIANT}_${gvar}.json"
            if [[ "$MULTI_VARIANT" = true && -f "$GROUPS_ARTIFACT" ]]; then
                echo "  SKIP $VARIANT_SLUG — already processed"
                continue
            fi

            mkdir -p "artifacts/$VARIANT_SLUG"

            if [ "$MULTI_VARIANT" = true ]; then
                cp "artifacts/${slug}/pruned_activations.json" \
                   "artifacts/${VARIANT_SLUG}/pruned_activations.json"
                cp "artifacts/${slug}/feature_descriptions_${DESCRIPTION_VARIANT}.json" \
                   "artifacts/${VARIANT_SLUG}/feature_descriptions_${DESCRIPTION_VARIANT}.json"
                cp "$GRAPH_JSON" "../test_graphs/${VARIANT_SLUG}.json"
            fi

            LABEL="[$ROW_NUM/$TOTAL | desc=$DESCRIPTION_VARIANT gvar=$gvar]"

            if [ "${SKIP_GROUPING:-0}" = "1" ]; then
                step_banner "$LABEL Step 3 — Grouping SKIPPED"
            else
                step_banner "$LABEL Step 3 — Generate supernodes"
                t=$(date +%s); python pipeline/generate_supernodes.py; elapsed "$t"
            fi

            step_banner "$LABEL Step 4 — Push to graph JSON"
            t=$(date +%s); python pipeline/push_to_website.py; elapsed "$t"

            step_banner "$LABEL Step 5 — Register in viewer dropdown"
            t=$(date +%s); python pipeline/update_metadata.py; elapsed "$t"

            if [ "$RUN_VALIDATE" = true ]; then
                step_banner "$LABEL Step 6 — Validate groups"
                t=$(date +%s); python pipeline/validate_groups.py; elapsed "$t"
            fi

            echo "  [$VARIANT_SLUG] artifacts/${VARIANT_SLUG}/"
        done

    elif [ "$RUN_VALIDATE" = true ]; then
        # validate-only mode: loop over variants too
        for gvar in "${VARIANT_LIST[@]}"; do
            export GROUPING_VARIANT="$gvar"
            if [ "$MULTI_VARIANT" = true ]; then
                export CURRENT_SLUG="${slug}-${DESCRIPTION_VARIANT}-${gvar}"
            else
                export CURRENT_SLUG="$slug"
            fi
            step_banner "[$ROW_NUM/$TOTAL | $gvar] Step 6 — Validate groups"
            t=$(date +%s); python pipeline/validate_groups.py; elapsed "$t"
        done
    fi

    echo ""
    echo "  [$ROW_NUM/$TOTAL] $slug complete."

done < "$CSV_FILE"

# ---------------------------------------------------------------------------
# Aggregate validation
# ---------------------------------------------------------------------------
if [ "$RUN_VALIDATE" = true ]; then
    step_banner "Aggregating validation scores"
    python analysis/aggregate_batch.py
    echo "  Batch summary → artifacts/batch_summary.md"
fi

unset CURRENT_SLUG GROUPING_VARIANT

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step_banner "SUCCESS — Batch pipeline complete ($ROW_NUM slug(s) processed)"
echo ""
if [ "$MULTI_VARIANT" = true ]; then
    echo "  $ROW_NUM prompt(s) × $N_VARIANTS grouping variant(s) processed."
    echo "  Graph naming: <slug>-${DESCRIPTION_VARIANT}-<grouping-variant>"
else
    echo "  $ROW_NUM prompt(s) processed."
fi
echo "  Refresh the viewer and use the dropdown to switch between graphs."
if [ "$RUN_VALIDATE" = true ]; then
    echo "  Validation summary → artifacts/batch_summary.md"
fi
echo ""
