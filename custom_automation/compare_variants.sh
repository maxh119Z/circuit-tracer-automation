#!/usr/bin/env bash
# =============================================================================
# compare_variants.sh — Run the pipeline with multiple description variants
# and compare validation results.
#
# Usage:
#   ./compare_variants.sh                    Compare v0, v1, v2, v3
#   ./compare_variants.sh v1 v3              Compare specific variants
#   ./compare_variants.sh v1 v2 0.35         Compare v1, v2 with custom threshold
#
# Prerequisites:
#   - Step 1 (fetch_all_activation_text.py) must have already run.
#     This script reuses pruned_activations.json and only re-runs steps 2-6.
#   - OPENAI_API_KEY must be set.
#
# Output:
#   artifacts/variant_comparison.json — side-by-side validation results
#   artifacts/<variant>/                — per-variant artifacts (descriptions, groups, reports)
#
# Environment variables:
#   OPENAI_API_KEY   — Required.
#   PRUNING_THRESHOLD — Optional (default: 0.40).
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
VARIANTS=()
PRUNING_THRESHOLD="${PRUNING_THRESHOLD:-0.40}"

for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        PRUNING_THRESHOLD="$arg"
    elif [[ "$arg" =~ ^v[0-9]$ ]]; then
        VARIANTS+=("$arg")
    else
        echo "Unknown argument: $arg"
        echo "Usage: ./compare_variants.sh [v0|v1|v2|v3]... [PRUNING_THRESHOLD]"
        exit 1
    fi
done

# Default: run all variants
if [ ${#VARIANTS[@]} -eq 0 ]; then
    VARIANTS=(v0 v1 v2 v3)
fi

export PRUNING_THRESHOLD

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
# Pre-flight
# ---------------------------------------------------------------------------
step_banner "Pre-flight — Variant Comparison"

if [ ! -f "artifacts/pruned_activations.json" ]; then
    echo "ERROR: artifacts/pruned_activations.json not found." >&2
    echo "       Run './reproduce.sh core' first to complete Step 1." >&2
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo -n "OPENAI_API_KEY not found. Enter it now (hidden): "
    read -s USER_API_KEY
    echo ""
    if [ -z "$USER_API_KEY" ]; then
        echo "ERROR: API key cannot be empty." >&2
        exit 1
    fi
    export OPENAI_API_KEY="$USER_API_KEY"
fi

echo "  Variants: ${VARIANTS[*]}"
echo "  Pruning threshold: $PRUNING_THRESHOLD"
echo ""

# ---------------------------------------------------------------------------
# Run each variant
# ---------------------------------------------------------------------------
BASE_GRAPH="../test_graphs/test-run.json"

if [ ! -f "$BASE_GRAPH" ]; then
    echo "ERROR: $BASE_GRAPH not found — run './reproduce.sh core' first." >&2
    exit 1
fi

for variant in "${VARIANTS[@]}"; do
    step_banner "Running variant: $variant"

    SLUG="desc-${variant}"
    export DESCRIPTION_VARIANT="$variant"
    export CURRENT_SLUG="$SLUG"

    # Seed this variant's artifact dir with the shared pruned activations
    mkdir -p "artifacts/${SLUG}"
    cp artifacts/pruned_activations.json "artifacts/${SLUG}/pruned_activations.json"

    # Seed this variant's graph file from the base graph
    cp "$BASE_GRAPH" "../test_graphs/${SLUG}.json"

    # Step 2 — Generate descriptions
    step_banner "  [$variant] Step 2 — Generating descriptions"
    t=$(date +%s)
    python generate_description.py
    elapsed "$t"

    # Step 3 — Group features
    step_banner "  [$variant] Step 3 — Grouping features"
    t=$(date +%s)
    python generate_supernodes.py
    elapsed "$t"

    # Step 4 — Push groups into this variant's graph file
    step_banner "  [$variant] Step 4 — Writing groups to graph JSON"
    t=$(date +%s)
    python push_to_website.py
    elapsed "$t"

    # Step 5 — Register in viewer dropdown as "desc-<variant>"
    step_banner "  [$variant] Step 5 — Registering in viewer dropdown"
    t=$(date +%s)
    python update_metadata.py
    elapsed "$t"

    # Step 6 — Validate
    step_banner "  [$variant] Step 6 — Validation"
    t=$(date +%s)
    python validate_groups.py
    elapsed "$t"

    echo ""
    echo "  [$variant] Artifacts → artifacts/${SLUG}/"
    echo "  [$variant] Graph     → test_graphs/${SLUG}.json"
    echo "  [$variant] Viewable in dropdown as: ${SLUG}"
done

unset CURRENT_SLUG

# ---------------------------------------------------------------------------
# Compare results
# ---------------------------------------------------------------------------
step_banner "Comparing variants"

python3 -c "
import json, os, sys

variants = '${VARIANTS[*]}'.split()
comparison = {'variants': {}, 'pruning_threshold': float('${PRUNING_THRESHOLD}')}

for v in variants:
    gv = os.environ.get('GROUPING_VARIANT', 'a0')
    report_path = f'artifacts/desc-{v}/validation_report_{v}_{gv}.json'
    if not os.path.exists(report_path):
        print(f'  WARNING: {report_path} not found, skipping {v}', file=sys.stderr)
        continue
    with open(report_path) as f:
        report = json.load(f)

    entry = {'description_variant': v}

    # D1 — description accuracy
    d1 = report.get('description_accuracy', {}).get('macro_avg', {})
    entry['description_accuracy'] = {
        'mean': d1.get('mean_accuracy', 0.0),
        'stderr': d1.get('stderr_accuracy', 0.0),
    }

    # Group-level metrics (auto/easy)
    auto_easy = report.get('auto', {}).get('easy', {})
    m1 = auto_easy.get('method1', {}).get('macro_avg', {})
    m2 = auto_easy.get('method2', {}).get('macro_avg', {})
    entry['group_m1_easy'] = {
        'mean': m1.get('mean_accuracy', 0.0),
        'stderr': m1.get('stderr_accuracy', 0.0),
    }
    entry['group_m2_easy'] = {
        'mean': m2.get('mean_accuracy', 0.0),
        'stderr': m2.get('stderr_accuracy', 0.0),
    }

    entry['attribution_coverage'] = report.get('auto', {}).get('attribution_coverage', 0.0)
    comparison['variants'][v] = entry

# Print summary table
print()
print('=' * 90)
print('  VARIANT COMPARISON')
print('=' * 90)
print(f\"{'Variant':<10}  {'D1 Desc Acc':>14}  {'M1 Group (easy)':>16}  {'M2 Group (easy)':>16}  {'Coverage':>9}\")
print('-' * 90)
for v in variants:
    e = comparison['variants'].get(v)
    if not e:
        continue
    d1 = e['description_accuracy']
    m1 = e['group_m1_easy']
    m2 = e['group_m2_easy']
    cov = e['attribution_coverage']
    print(f\"{v:<10}  {d1['mean']:>6.1%} ± {d1['stderr']:.1%}    {m1['mean']:>6.1%} ± {m1['stderr']:.1%}    {m2['mean']:>6.1%} ± {m2['stderr']:.1%}    {cov:>7.1%}\")
print('=' * 90)

with open('artifacts/variant_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)
print(f'\nComparison saved to artifacts/variant_comparison.json')
"

step_banner "SUCCESS — Variant comparison complete"
echo "  Results:  artifacts/variant_comparison.json"
for variant in "${VARIANTS[@]}"; do
    echo "  desc-${variant}: artifacts/desc-${variant}/  |  test_graphs/desc-${variant}.json  |  viewer dropdown: desc-${variant}"
done
echo ""
