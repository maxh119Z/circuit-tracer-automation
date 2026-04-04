#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Run the custom_automation pipeline.
#
# Subcommands:
#   fetch           Steps 0-1 only: patch frontend + fetch activations.
#                   No OpenAI key needed. Run this first for any new prompt.
#   fetch-desc      Steps 0-2: fetch activations + generate descriptions.
#                   Run once per description variant before iterating on groups.
#   core / all      Full pipeline: steps 0-5 + validation (default).
#   validate        Validation only (step 6). Assumes core already ran.
#   desc-group      Steps 0-5 with a specific DESCRIPTION_VARIANT + GROUPING_VARIANT.
#   all-groups      Steps 0-2 once, then steps 3-6 for all 4 grouping variants (a0–a3).
#   all-variants    Steps 0-1 once, then steps 2-6 for all description variants (v0–v3).
#   compare         Compare all validation reports in artifacts/ → artifacts/comparison.md
#
# Usage examples:
#   ./reproduce.sh fetch
#   DESCRIPTION_VARIANT=v2 ./reproduce.sh fetch-desc
#   ./reproduce.sh core 0.40
#   DESCRIPTION_VARIANT=v2 GROUPING_VARIANT=a1 ./reproduce.sh desc-group
#   DESCRIPTION_VARIANT=v2 ./reproduce.sh all-groups
#   ./reproduce.sh compare
#
# Environment variables:
#   OPENAI_API_KEY   — Required for descriptions, grouping, and validation.
#   VIEWER_URL       — Optional. Base URL for the viewer (default: localhost:8041).
#
# After running: refresh the viewer page. Groups load automatically.
# Clear browser cache if regenerating a same-prompt graph.
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

RUN_FETCH=false
RUN_FETCH_DESC=false
RUN_CORE=false
RUN_VALIDATE=false
RUN_VARIANTS=false
RUN_DESC_GROUP=false
RUN_ALL_GROUPS=false
RUN_COMPARE=false
RUN_BATCH_SUMMARY=false

case "$SUBCOMMAND" in
    fetch)
        RUN_FETCH=true
        ;;
    fetch-desc)
        RUN_FETCH_DESC=true
        ;;
    core|all)
        RUN_CORE=true
        RUN_VALIDATE=true
        ;;
    validate)
        RUN_VALIDATE=true
        ;;
    all-variants)
        RUN_VARIANTS=true
        ;;
    desc-group)
        RUN_DESC_GROUP=true
        ;;
    all-groups)
        RUN_ALL_GROUPS=true
        ;;
    compare)
        RUN_COMPARE=true
        ;;
    batch-summary)
        RUN_BATCH_SUMMARY=true
        ;;
    *)
        echo "Unknown subcommand: $SUBCOMMAND"
        echo "Usage: ./reproduce.sh [fetch|fetch-desc|core|all|validate|all-variants|desc-group|all-groups|compare|batch-summary] [PRUNING_THRESHOLD]"
        echo ""
        echo "  fetch       — Steps 0-1 only (no OpenAI key needed)."
        echo "  fetch-desc  — Steps 0-2: fetch + descriptions."
        echo "    Ex: DESCRIPTION_VARIANT=v2 ./reproduce.sh fetch-desc"
        echo "  desc-group  — Steps 0-5 with specific DESCRIPTION_VARIANT and GROUPING_VARIANT."
        echo "    Ex: DESCRIPTION_VARIANT=v2 GROUPING_VARIANT=a1 ./reproduce.sh desc-group"
        echo "  all-groups  — Steps 0-2 once, then steps 3-6 for all 4 grouping variants."
        echo "    Ex: DESCRIPTION_VARIANT=v2 ./reproduce.sh all-groups"
        echo "  compare     — Read all validation reports in artifacts/ and write comparison.md."
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
echo "  ✓ DESCRIPTION_VARIANT=${DESCRIPTION_VARIANT:-v2}"
echo "  ✓ GROUPING_VARIANT=${GROUPING_VARIANT:-a0}"
echo "  ✓ Mode: $SUBCOMMAND"
if [ "$RUN_ALL_GROUPS" = true ]; then
    echo "  ✓ Description variant: ${DESCRIPTION_VARIANT:-v2} (all 4 grouping variants: a0 a1 a2 a3)"
fi

if [ "$RUN_FETCH" = true ] || [ "$RUN_COMPARE" = true ]; then
    echo "  ✓ fetch/compare mode — OPENAI_API_KEY not required."
elif [ -z "${OPENAI_API_KEY:-}" ]; then
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
# Fetch only (steps 0-1)
# ---------------------------------------------------------------------------
if [ "$RUN_FETCH" = true ]; then

    step_banner "Step 0 — Applying frontend patch (if needed)"
    python apply_frontend_patch.py

    step_banner "Step 1 — Fetching pruned feature activations"
    t=$(date +%s)
    python fetch_all_activation_text.py
    elapsed "$t"

fi

# ---------------------------------------------------------------------------
# Fetch + describe (steps 0-2)
# ---------------------------------------------------------------------------
if [ "$RUN_FETCH_DESC" = true ]; then
    export DESCRIPTION_VARIANT="${DESCRIPTION_VARIANT:-v2}"

    step_banner "Step 0 — Applying frontend patch (if needed)"
    python apply_frontend_patch.py

    step_banner "Step 1 — Fetching pruned feature activations"
    t=$(date +%s)
    python fetch_all_activation_text.py
    elapsed "$t"

    step_banner "Step 2 — Generating descriptions [desc=$DESCRIPTION_VARIANT]"
    t=$(date +%s)
    python generate_description.py
    elapsed "$t"

fi

# ---------------------------------------------------------------------------
# Single-variant pipeline (steps 0-5 + validation)
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

if [ "$RUN_VALIDATE" = true ]; then
    step_banner "Step 6 — Validating groups (M1: feature ID, M2: text match)"
    t=$(date +%s)
    python validate_groups.py
    elapsed "$t"
fi

# ---------------------------------------------------------------------------
# All-variants pipeline (steps 0-1 once, steps 2-6 for each variant)
# ---------------------------------------------------------------------------
if [ "$RUN_VARIANTS" = true ]; then
    VARIANTS=(v0 v1 v2 v3)

    # Step 0 — once
    step_banner "Step 0 — Applying frontend patch (if needed)"
    python apply_frontend_patch.py

    # Step 1 — once (pruned_activations.json is shared across all variants)
    step_banner "Step 1 — Fetching pruned feature activations (shared across variants)"
    t=$(date +%s)
    python fetch_all_activation_text.py
    elapsed "$t"

    # Steps 2-6 per variant
    for variant in "${VARIANTS[@]}"; do
        export DESCRIPTION_VARIANT="$variant"

        step_banner "[$variant] Step 2 — Generating descriptions"
        t=$(date +%s)
        python generate_description.py
        elapsed "$t"

        step_banner "[$variant] Step 3 — Grouping features into supernodes"
        t=$(date +%s)
        python generate_supernodes.py
        elapsed "$t"

        step_banner "[$variant] Step 4 — Writing groups into graph JSON"
        t=$(date +%s)
        python push_to_website.py
        elapsed "$t"

        step_banner "[$variant] Step 5 — Registering graph in viewer dropdown"
        t=$(date +%s)
        python update_metadata.py
        elapsed "$t"

        step_banner "[$variant] Step 6 — Validating groups"
        t=$(date +%s)
        python validate_groups.py
        elapsed "$t"
    done
fi

# ---------------------------------------------------------------------------
# Desc+Group variant pipeline (steps 0-1 once, steps 2-6 with specific desc+group combo)
# ---------------------------------------------------------------------------
if [ "$RUN_DESC_GROUP" = true ]; then
    export DESCRIPTION_VARIANT="${DESCRIPTION_VARIANT:-v2}"
    export GROUPING_VARIANT="${GROUPING_VARIANT:-a0}"

    # Step 0 — once
    step_banner "Step 0 — Applying frontend patch (if needed)"
    python apply_frontend_patch.py

    # Step 1 — once (pruned_activations.json is shared)
    step_banner "Step 1 — Fetching pruned feature activations (shared)"
    t=$(date +%s)
    python fetch_all_activation_text.py
    elapsed "$t"

    step_banner "[desc=$DESCRIPTION_VARIANT group=$GROUPING_VARIANT] Step 2 — Generating descriptions"
    t=$(date +%s)
    python generate_description.py
    elapsed "$t"

    step_banner "[desc=$DESCRIPTION_VARIANT group=$GROUPING_VARIANT] Step 3 — Grouping features into supernodes"
    t=$(date +%s)
    python generate_supernodes.py
    elapsed "$t"

    step_banner "[desc=$DESCRIPTION_VARIANT group=$GROUPING_VARIANT] Step 4 — Writing groups into graph JSON"
    t=$(date +%s)
    python push_to_website.py
    elapsed "$t"

    step_banner "[desc=$DESCRIPTION_VARIANT group=$GROUPING_VARIANT] Step 5 — Registering graph in viewer dropdown"
    t=$(date +%s)
    python update_metadata.py
    elapsed "$t"

    step_banner "[desc=$DESCRIPTION_VARIANT group=$GROUPING_VARIANT] Step 6 — Validating groups"
    t=$(date +%s)
    python validate_groups.py
    elapsed "$t"
fi

# ---------------------------------------------------------------------------
# All-groups pipeline (steps 0-2 once, steps 3-6 for each grouping variant)
# ---------------------------------------------------------------------------
if [ "$RUN_ALL_GROUPS" = true ]; then
    DESC_VAR="${DESCRIPTION_VARIANT:-v2}"
    GROUPING_VARIANTS_LIST=(a0 a1 a2 a3)
    BASE_GRAPH="../test_graphs/test-run.json"

    if [ ! -f "$BASE_GRAPH" ]; then
        echo "ERROR: $BASE_GRAPH not found — run './reproduce.sh core' first." >&2
        exit 1
    fi

    # Step 0 — once
    step_banner "Step 0 — Applying frontend patch (if needed)"
    python apply_frontend_patch.py

    # Step 1 — once (shared pruned activations)
    step_banner "Step 1 — Fetching pruned feature activations (shared)"
    t=$(date +%s)
    python fetch_all_activation_text.py
    elapsed "$t"

    # Step 2 — once (generate descriptions for chosen variant)
    export DESCRIPTION_VARIANT="$DESC_VAR"
    step_banner "Step 2 — Generating descriptions [desc=$DESC_VAR]"
    t=$(date +%s)
    python generate_description.py
    elapsed "$t"

    # Steps 3-6 for each grouping variant
    for gvar in "${GROUPING_VARIANTS_LIST[@]}"; do
        SLUG="${DESC_VAR}-${gvar}"
        export GROUPING_VARIANT="$gvar"
        export CURRENT_SLUG="$SLUG"

        mkdir -p "artifacts/${SLUG}"
        cp artifacts/pruned_activations.json "artifacts/${SLUG}/pruned_activations.json"
        cp "artifacts/feature_descriptions_${DESC_VAR}.json" \
           "artifacts/${SLUG}/feature_descriptions_${DESC_VAR}.json"
        cp "$BASE_GRAPH" "../test_graphs/${SLUG}.json"

        step_banner "[desc=$DESC_VAR group=$gvar] Step 3 — Grouping features into supernodes"
        t=$(date +%s)
        python generate_supernodes.py
        elapsed "$t"

        step_banner "[desc=$DESC_VAR group=$gvar] Step 4 — Writing groups into graph JSON"
        t=$(date +%s)
        python push_to_website.py
        elapsed "$t"

        step_banner "[desc=$DESC_VAR group=$gvar] Step 5 — Registering graph in viewer dropdown"
        t=$(date +%s)
        python update_metadata.py
        elapsed "$t"

        step_banner "[desc=$DESC_VAR group=$gvar] Step 6 — Validating groups"
        t=$(date +%s)
        python validate_groups.py
        elapsed "$t"

        echo ""
        echo "  [$SLUG] Artifacts → artifacts/${SLUG}/"
        echo "  [$SLUG] Graph     → test_graphs/${SLUG}.json"
        echo "  [$SLUG] Viewer dropdown: ${SLUG}"
    done

    unset CURRENT_SLUG
fi

# ---------------------------------------------------------------------------
# Compare (read all validation reports → artifacts/comparison.md)
# ---------------------------------------------------------------------------
if [ "$RUN_COMPARE" = true ]; then
    step_banner "Generating comparison.md"
    python compare_variants.py
fi

# ---------------------------------------------------------------------------
# Batch summary (average scores across all prompts → artifacts/batch_summary.md)
# ---------------------------------------------------------------------------
if [ "$RUN_BATCH_SUMMARY" = true ]; then
    step_banner "Generating batch_summary.md"
    python aggregate_batch.py
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
step_banner "SUCCESS — Pipeline complete ($SUBCOMMAND)"
echo ""
if [ "$RUN_CORE" = true ] || [ "$RUN_VARIANTS" = true ] || [ "$RUN_DESC_GROUP" = true ] || [ "$RUN_ALL_GROUPS" = true ] || [ "$RUN_FETCH_DESC" = true ]; then
    echo "  Your groups are saved in the graph JSON."
    echo "  The graph is registered in the viewer dropdown."
    echo "  👉 Just refresh the viewer page — select your graph, groups load automatically."
    echo "  Clear browser cache if regenerating a same prompt graph again. Unnecessary if first time."
    echo ""
fi
if [ "$RUN_VALIDATE" = true ] || [ "$RUN_VARIANTS" = true ] || [ "$RUN_DESC_GROUP" = true ]; then
    echo "  Validation report: artifacts/validation_report_<variant>.json"
    echo ""
fi
if [ "$RUN_ALL_GROUPS" = true ]; then
    _dv="${DESCRIPTION_VARIANT:-v2}"
    echo "  4 graphs registered in viewer dropdown:"
    for _gv in a0 a1 a2 a3; do
        echo "    ${_dv}-${_gv}  →  test_graphs/${_dv}-${_gv}.json"
    done
    echo "  Run './reproduce.sh compare' to generate artifacts/comparison.md"
    echo ""
fi
if [ "$RUN_COMPARE" = true ]; then
    echo "  Comparison → artifacts/comparison.md"
    echo ""
fi
