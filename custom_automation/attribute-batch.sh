#!/usr/bin/env bash
# =============================================================================
# attribute-batch.sh — Run `circuit-tracer attribute` for every row in a CSV.
#
# Usage:
#   ./attribute-batch.sh <prompts.csv> [--server]
#
# CSV format (header row required):
#   slug,prompt,transcoder_set
#
# Options:
#   --server   Start the viewer server after all attribution runs complete.
#
# Examples:
#   ./attribute-batch.sh ../prompts.csv
#   ./attribute-batch.sh ../prompts.csv --server
#
# After running, feed the same CSV into batch_reproduce.sh or batch_all_groups.sh
# to run the full automation pipeline.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
CSV_FILE="${1:?Usage: ./attribute-batch.sh <prompts.csv> [--server]}"
START_SERVER=false
[[ "${2:-}" == "--server" ]] && START_SERVER=true

if [[ ! -f "$CSV_FILE" ]]; then
    echo "ERROR: CSV file not found: $CSV_FILE" >&2
    exit 1
fi

GRAPH_FILE_DIR="../test_graphs"
mkdir -p "$GRAPH_FILE_DIR"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if ! command -v circuit-tracer &>/dev/null; then
    echo "ERROR: circuit-tracer not found on PATH." >&2
    echo "       Install with: pip install ." >&2
    exit 1
fi

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
echo "attribute-batch: $TOTAL prompt(s) from $CSV_FILE"
echo "Output dir: $GRAPH_FILE_DIR"
echo ""

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
ROW_NUM=0
while IFS=',' read -r slug prompt transcoder_set || [[ -n "$slug" ]]; do
    # Skip header and blank lines
    [[ "$slug" == "slug" ]] && continue
    [[ -z "${slug// }" ]] && continue

    ROW_NUM=$(( ROW_NUM + 1 ))
    slug="${slug// /}"
    slug="${slug//$'\r'/}"
    transcoder_set="${transcoder_set// /}"
    transcoder_set="${transcoder_set//$'\r'/}"

    step_banner "[$ROW_NUM/$TOTAL] slug: $slug"
    prompt="${prompt//$'\r'/}"

    echo "  Prompt:         $prompt"
    echo "  Transcoder set: $transcoder_set"

    GRAPH_JSON="$GRAPH_FILE_DIR/${slug}.json"
    if [[ -f "$GRAPH_JSON" ]]; then
        echo "  Graph already exists, skipping: $GRAPH_JSON"
        continue
    fi

    t=$(date +%s)
    circuit-tracer attribute \
        --prompt "$prompt" \
        --transcoder_set "$transcoder_set" \
        --slug "$slug" \
        --graph_file_dir "$GRAPH_FILE_DIR"
    elapsed "$t"

    echo "  Saved: $GRAPH_JSON"

done < "$CSV_FILE"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step_banner "SUCCESS — Attribution complete ($ROW_NUM slug(s) processed)"
echo ""
echo "  Graph files: $GRAPH_FILE_DIR/"
echo "  Next step:   cd custom_automation && ./batch_reproduce.sh $CSV_FILE"
echo "            or ./batch_all_groups.sh $CSV_FILE"
echo ""

# ---------------------------------------------------------------------------
# Optional: start viewer
# ---------------------------------------------------------------------------
if [ "$START_SERVER" = true ]; then
    echo "Starting viewer server..."
    circuit-tracer start-server --graph_file_dir "$GRAPH_FILE_DIR"
fi
