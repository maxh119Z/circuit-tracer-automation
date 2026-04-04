#!/usr/bin/env python3
"""
apply_frontend_patch.py — Apply the qParams supernodes fix to init-cg.js.

Run from the repo root:
    python custom_automation/apply_frontend_patch.py

This makes two surgical edits to init-cg.js:

1. Adds parsing for string-format supernodes/pinnedIds from qParams
   (so pipeline-written qParams are correctly loaded as arrays)

2. Fixes the `if (urlSupernodes)` → `if (urlSupernodes.length)` bug
   (empty arrays are truthy in JS, so the old code always overwrote qParams)

Both changes are needed — running this script twice is safe.
"""

import re
import sys
from pathlib import Path

# Locate the file relative to this script (custom_automation/ → repo root → circuit_tracer/)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TARGET = REPO_ROOT / "circuit_tracer" / "frontend" / "assets" / "attribution_graph" / "init-cg.js"


def apply_patches():
    if not TARGET.exists():
        print(f"ERROR: Cannot find {TARGET}")
        sys.exit(1)

    content = TARGET.read_text()
    original = content
    changes = []

    # -----------------------------------------------------------------------
    # PATCH 1: Add qParams parsing before the `clickedId?.includes('supernode')` line
    # -----------------------------------------------------------------------
    PARSE_BLOCK = """\
  // [PIPELINE PATCH] Parse qParams values that may be stored as strings
  if (typeof visState.supernodes === 'string') {
    try { visState.supernodes = JSON.parse(visState.supernodes) } catch(e) { visState.supernodes = [] }
  }
  if (typeof visState.pinnedIds === 'string') {
    visState.pinnedIds = visState.pinnedIds.split(',').filter(Boolean)
  }

"""

    ANCHOR = "if (visState.clickedId?.includes('supernode'))"

    if "CUSTOM EDITS START" not in content:
        # Insert the parse block right before the anchor line
        if ANCHOR in content:
            content = content.replace(ANCHOR, PARSE_BLOCK + "  " + ANCHOR)
            changes.append("Added qParams string parsing (supernodes + pinnedIds)")
        else:
            print(f"WARNING: Could not find anchor line: {ANCHOR!r}")
            print("  You may need to apply Patch 1 manually — see PATCH_INSTRUCTIONS.js")
    else:
        print("  Patch 1 already applied (CUSTOM EDITS block present). Skipping.")

    # -----------------------------------------------------------------------
    # PATCH 2: Fix the truthy-array bug
    # -----------------------------------------------------------------------
    OLD_LINE = "if (urlSupernodes) visState.supernodes = urlSupernodes"
    NEW_LINE = "if (urlSupernodes.length) visState.supernodes = urlSupernodes"

    if OLD_LINE in content:
        content = content.replace(OLD_LINE, NEW_LINE, 1)
        changes.append("Fixed urlSupernodes truthy-array override bug")
    elif NEW_LINE in content:
        print("  Patch 2 already applied (urlSupernodes.length). Skipping.")
    else:
        print("WARNING: Could not find the urlSupernodes line to patch.")
        print("  You may need to apply Patch 2 manually — see PATCH_INSTRUCTIONS.js")

    # -----------------------------------------------------------------------
    # Write back
    # -----------------------------------------------------------------------
    if content != original:
        TARGET.write_text(content)
        print(f"\n✅ Patched {TARGET}:")
        for c in changes:
            print(f"   • {c}")
    else:
        print(f"\n✓ No changes needed — {TARGET} is already patched.")


if __name__ == "__main__":
    apply_patches()