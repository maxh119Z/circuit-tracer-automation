"""
Step 3 — Merge generated descriptions into the attribution graph.

Reads ``feature_descriptions.json`` from the artifacts directory, matches
each description to its graph node by ``node_id``, and updates the label
fields (``clerp``, ``localClerp``, ``ppClerp``) so the viewer shows
human-readable feature names.

A **backup** of the graph file is created before overwriting.

Usage:
    python push_to_website.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from config import (
    FEATURE_DESCRIPTIONS_FILE,
    GRAPH_FILE,
    setup_logging,
)

log = setup_logging()


# ---------------------------------------------------------------------------
# Merge Logic
# ---------------------------------------------------------------------------


def build_description_map(desc_path: Path) -> dict[str, str]:
    """Load descriptions and return a mapping of node_id → description text."""
    with open(desc_path, "r") as f:
        desc_list: list[dict] = json.load(f)

    desc_map: dict[str, str] = {}
    for item in desc_list:
        key = str(item.get("id", ""))
        desc = item.get("generated_description", "")
        if key and desc:
            desc_map[key] = desc

    log.info("Loaded %d descriptions from %s", len(desc_map), desc_path)
    return desc_map


def merge_descriptions(graph_path: Path, desc_map: dict[str, str]) -> int:
    """Inject descriptions into graph nodes and save.  Returns match count."""
    with open(graph_path, "r") as f:
        graph_data: dict = json.load(f)

    nodes = graph_data.get("nodes", [])
    input_tokens = graph_data.get("input_tokens", [])
    log.info("Scanning %d graph nodes …", len(nodes))

    match_count = 0
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        if node_id not in desc_map:
            continue

        # Build optional token context suffix, e.g. " (on 'Dallas')"
        token_label = ""
        ctx_idx = node.get("ctx_idx")
        if ctx_idx is not None and isinstance(input_tokens, list):
            if 0 <= ctx_idx < len(input_tokens):
                token_str = str(input_tokens[ctx_idx]).strip()
                if token_str:
                    token_label = f" (on '{token_str}')"

        full_label = f"{desc_map[node_id]}{token_label}"

        node["clerp"] = full_label
        node["localClerp"] = full_label
        node["ppClerp"] = full_label
        match_count += 1

    if match_count == 0:
        log.warning("0 matches — verify that feature_descriptions.json IDs match graph node_ids.")
        return 0

    # --- Backup before writing ---
    backup_path = graph_path.with_suffix(".json.bak")
    shutil.copy2(graph_path, backup_path)
    log.info("Backup saved → %s", backup_path)

    with open(graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    log.info("Updated %d / %d nodes in %s", match_count, len(nodes), graph_path)
    return match_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not FEATURE_DESCRIPTIONS_FILE.exists():
        log.error("Missing %s — run add_description.py first (Step 2).", FEATURE_DESCRIPTIONS_FILE)
        sys.exit(1)

    if not GRAPH_FILE.exists():
        log.error("Missing graph file: %s", GRAPH_FILE)
        sys.exit(1)

    desc_map = build_description_map(FEATURE_DESCRIPTIONS_FILE)
    if not desc_map:
        log.error("Description file is empty or malformed.")
        sys.exit(1)

    matched = merge_descriptions(GRAPH_FILE, desc_map)
    if matched > 0:
        log.info("Done! Refresh localhost:8041 and clear browser cache to see updates.")


if __name__ == "__main__":
    main()
