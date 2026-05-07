"""
Phase-2 cap sweep — produce supernode groupings at multiple feature caps from
ONE shared phase 1 + phase 2 run.

Usage:
    OPENAI_API_KEY=sk-... CURRENT_SLUG=gemma-G \
        CAPS=50,100,150,200,unlimited \
        python generate_supernodes_cap_sweep.py
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))  # custom_automation/   (for `config`)
sys.path.insert(0, str(_HERE.parent.parent))         # custom_automation/pipeline/  (for `generate_supernodes`)

from config import (
    ARTIFACTS_DIR,
    DESCRIPTION_VARIANT,
    GROUPING_BATCH_SIZE,
    GROUPING_TOP_K_SEED,
    GROUPING_VARIANT,
    setup_logging,
)
from generate_supernodes import (
    _api_usage,
    append_grouping_log,
    apply_phase1_output,
    apply_phase2_outputs,
    apply_phase3_actions,
    apply_postprocessing_to_snapshots,
    filter_active_groups_to_assigned,
    load_and_sort_features,
    run_phase1,
    run_phase2_batches,
    run_phase3,
    save_groupings,
    write_cost_csv_row,
)

log = setup_logging()

DEFAULT_CAPS = "50,100,150,200,unlimited"


# ---------------------------------------------------------------------------
# Cap parsing & file naming
# ---------------------------------------------------------------------------

def parse_caps(raw: str) -> list[int | None]:
    """Parse "50,100,unlimited" → [50, 100, None]. None means unlimited."""
    caps: list[int | None] = []
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("unlimited", "all", "none", "inf"):
            caps.append(None)
        else:
            try:
                n = int(tok)
            except ValueError:
                log.warning("Ignoring unparseable cap value: %r", tok)
                continue
            if n < GROUPING_TOP_K_SEED:
                log.warning(
                    "Cap %d is below GROUPING_TOP_K_SEED=%d — clamping to %d "
                    "(phase 1 always groups the seed).",
                    n, GROUPING_TOP_K_SEED, GROUPING_TOP_K_SEED,
                )
                n = GROUPING_TOP_K_SEED
            caps.append(n)
    if not caps:
        log.error("No valid caps parsed from %r", raw)
        sys.exit(1)
    return caps


def cap_paths(cap: int | None) -> tuple[Path, Path]:
    """Final + pre-phase-3 paths for a given cap. None ⇒ canonical (unsuffixed)."""
    suffix = "" if cap is None else f"_cap{cap}"
    final = ARTIFACTS_DIR / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}{suffix}.json"
    pre3  = ARTIFACTS_DIR / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}{suffix}_pre3.json"
    return final, pre3


def cap_label(cap: int | None) -> str:
    return "unlimited" if cap is None else f"cap{cap}"


# ---------------------------------------------------------------------------
# Per-cap stitching
# ---------------------------------------------------------------------------

def n_phase2_batches_for_cap(cap: int | None, total_remaining: int) -> int:
    """How many phase-2 batches should be stitched for this cap.

    cap=50 → 0 batches (phase 1 only).
    cap=100 → 1 batch. cap=150 → 2. cap=200 → 3.
    cap=None (unlimited) → all batches needed to cover total_remaining.
    """
    if cap is None:
        return (total_remaining + GROUPING_BATCH_SIZE - 1) // GROUPING_BATCH_SIZE
    additional_features = max(0, cap - GROUPING_TOP_K_SEED)
    return min(
        (additional_features + GROUPING_BATCH_SIZE - 1) // GROUPING_BATCH_SIZE,
        (total_remaining + GROUPING_BATCH_SIZE - 1) // GROUPING_BATCH_SIZE,
    )


async def build_snapshot_for_cap(
    cap: int | None,
    p1,
    phase2_outputs,
    all_features,
    prompt_text,
    output_context,
    total_remaining: int,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """Stitch phase 1 + the right phase-2 prefix, then run phase 3.

    Returns (final_assignments, pre_phase3_assignments, phase1_group_names).
    """
    label = cap_label(cap)
    log.info("=== Cap %s ===", label)

    active_groups: dict[str, str] = {}
    final_assignments: dict[str, str] = {}

    apply_phase1_output(p1, active_groups, final_assignments)
    phase1_group_names = list(active_groups.keys())

    n_batches = n_phase2_batches_for_cap(cap, total_remaining)
    if n_batches > 0:
        apply_phase2_outputs(phase2_outputs[:n_batches], active_groups, final_assignments)
        log.info("[%s] Stitched phase 1 + %d/%d phase-2 batches (%d features grouped).",
                 label, n_batches, len(phase2_outputs), len(final_assignments))
    else:
        log.info("[%s] Phase-1-only snapshot (%d features grouped).",
                 label, len(final_assignments))

    # Filter out phantom groups before phase 3 sees them — these are groups
    # invented in batches we didn't include (relevant only to smaller caps).
    active_groups = filter_active_groups_to_assigned(active_groups, final_assignments)

    pre_phase3 = copy.deepcopy(final_assignments)

    p3 = await run_phase3(final_assignments, all_features, prompt_text, output_context)
    apply_phase3_actions(p3, active_groups, final_assignments)
    append_grouping_log(f"{prompt_text} [{label}]", phase1_group_names, p3, final_assignments)

    return final_assignments, pre_phase3, phase1_group_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    raw_caps = os.environ.get("CAPS", DEFAULT_CAPS)
    caps = parse_caps(raw_caps)
    log.info("Caps: %s", [cap_label(c) for c in caps])

    _start = time.time()
    _api_usage.clear()

    features, prompt_text, output_context = load_and_sort_features()
    if not features:
        log.error("No described features found.")
        return

    log.info("Total features: %d  |  Prompt: %s", len(features), prompt_text)

    # Phase 1 — once for all caps.
    p1 = await run_phase1(features[:GROUPING_TOP_K_SEED], prompt_text, output_context)
    if p1 is None:
        return

    # Phase 2 — run enough batches for the largest needed cap.
    remaining = features[GROUPING_TOP_K_SEED:]
    largest_cap = None if any(c is None for c in caps) else max(c for c in caps if c is not None)
    if largest_cap is None:
        phase2_remaining = remaining
    else:
        n_extra = max(0, largest_cap - GROUPING_TOP_K_SEED)
        phase2_remaining = remaining[:n_extra]
    log.info("Phase 2 will cover %d features (largest cap = %s).",
             len(phase2_remaining), cap_label(largest_cap))

    # Build phase-1 active_groups view for context — same as the standard run.
    p1_active_groups: dict[str, str] = {}
    p1_final_assignments: dict[str, str] = {}
    apply_phase1_output(p1, p1_active_groups, p1_final_assignments)

    phase2_outputs = await run_phase2_batches(
        phase2_remaining, p1_active_groups, prompt_text, output_context,
    )
    log.info("Phase 2 complete: %d batches.", len(phase2_outputs))

    # Per-cap stitch + phase 3 + save.
    for cap in sorted(caps, key=lambda c: (c is None, c if c is not None else 0)):
        final_assignments, pre_phase3, _ = await build_snapshot_for_cap(
            cap, p1, phase2_outputs, features, prompt_text, output_context,
            total_remaining=len(remaining),
        )
        apply_postprocessing_to_snapshots([pre_phase3, final_assignments])
        final_path, pre3_path = cap_paths(cap)
        save_groupings(final_assignments, pre_phase3, final_path, pre3_path)

    write_cost_csv_row(_start, label="cap_sweep")


if __name__ == "__main__":
    asyncio.run(main())
