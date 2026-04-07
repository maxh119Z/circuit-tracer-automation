"""
Step 4 — Push descriptions and groups into the graph JSON.

Writes:
  1. Feature descriptions directly into each node's `clerp` field
     (the viewer's hClerpUpdateFn uses node.clerp as fallback for ppClerp)
  2. Supernodes + pinnedIds into qParams (viewer reads on load)

Reads:
  - artifacts/feature_descriptions.json
  - artifacts/feature_groups.json
  - ../test_graphs/test-run.json

Outputs:
  - Updated test-run.json

Usage:
    python push_to_website.py
"""

from __future__ import annotations

import json
import os
import sys
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEFAULT_PRUNING_THRESHOLD,
    FEATURE_DESCRIPTIONS_FILE,
    FEATURE_GROUPS_FILE,
    GRAPH_FILE,
    VIEWER_URL_FILE,
    setup_logging,
)

log = setup_logging()

PRUNING_THRESHOLD = os.environ.get("PRUNING_THRESHOLD", str(DEFAULT_PRUNING_THRESHOLD))


def push_to_graph() -> None:
    # ==================================================================
    # 1. LOAD DATA
    # ==================================================================
    if not GRAPH_FILE.exists():
        log.error("Missing graph file: %s", GRAPH_FILE)
        return

    with open(GRAPH_FILE, "r") as f:
        graph_data = json.load(f)

    # Descriptions
    descriptions: dict[str, str] = {}
    if FEATURE_DESCRIPTIONS_FILE.exists():
        with open(FEATURE_DESCRIPTIONS_FILE, "r") as f:
            descriptions = {
                item["id"]: item["generated_description"] for item in json.load(f)
            }
    else:
        log.warning("No descriptions file found at %s", FEATURE_DESCRIPTIONS_FILE)

    # Groups
    groups: dict[str, str] = {}
    if FEATURE_GROUPS_FILE.exists():
        with open(FEATURE_GROUPS_FILE, "r") as f:
            groups = json.load(f)

    log.info("Loaded %d descriptions, %d group assignments.", len(descriptions), len(groups))

    # ==================================================================
    # 2. INJECT DESCRIPTIONS INTO NODE CLERP FIELDS
    # ==================================================================
    # The viewer's hClerpUpdateFn sets: ppClerp = localClerp || remoteClerp || clerp
    # Writing to `clerp` on the node makes descriptions show up as the fallback.
    nodes = graph_data.get("nodes", [])

    desc_count = 0
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        if node_id in descriptions:
            desc = descriptions[node_id]
            # Only set clerp if the node doesn't already have a meaningful one
            # (logit and embedding nodes have built-in clerps we want to keep)
            existing = node.get("clerp", "")
            if not existing or node.get("feature_type") == "cross layer transcoder":
                node["clerp"] = desc
                desc_count += 1

    log.info("Injected descriptions into %d node clerp fields.", desc_count)

    # ==================================================================
    # 3. BUILD SUPERNODES ARRAY
    # ==================================================================
    group_to_nodes: dict[str, list[str]] = {}
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        group_name = groups.get(node_id)
        if group_name and group_name != "Ungrouped":
            clean_name = group_name.replace("/", "-").replace("\\", "-")
            group_to_nodes.setdefault(clean_name, []).append(node_id)

    supernodes_array: list[list[str]] = []
    all_pinned_ids: list[str] = []
    for gname, member_ids in group_to_nodes.items():
        supernodes_array.append([gname] + member_ids)
        all_pinned_ids.extend(member_ids)

    log.info("Built %d supernode groups with %d total pinned nodes.",
             len(supernodes_array), len(all_pinned_ids))

    # ==================================================================
    # 4. WRITE INTO qParams
    # ==================================================================
    q_params = graph_data.get("qParams", {})
    q_params["supernodes"] = json.dumps(supernodes_array, separators=(",", ":"))
    q_params["pinnedIds"] = ",".join(all_pinned_ids)
    q_params["pruningThreshold"] = PRUNING_THRESHOLD
    # Remove clerps from qParams — descriptions live on nodes now
    q_params.pop("clerps", None)
    graph_data["qParams"] = q_params

    # ==================================================================
    # 5. BACKUP & SAVE
    # ==================================================================
    backup_path = str(GRAPH_FILE) + ".bak"
    shutil.copy2(GRAPH_FILE, backup_path)
    log.info("Backup saved → %s", backup_path)

    with open(GRAPH_FILE, "w") as f:
        json.dump(graph_data, f, indent=2)

    log.info("Graph updated → %s", GRAPH_FILE)
    log.info("  clerps on nodes: %d", desc_count)
    log.info("  supernodes:      %d groups", len(supernodes_array))
    log.info("  pinnedIds:       %d nodes", len(all_pinned_ids))
    log.info("  threshold:       %s", PRUNING_THRESHOLD)

    with open(VIEWER_URL_FILE, "w") as f:
        f.write("# Pipeline output saved to graph JSON — just refresh the viewer.\n")
        f.write(f"# Graph file: {GRAPH_FILE}\n")
        f.write(f"# Descriptions: {desc_count}, Groups: {len(supernodes_array)}, Pinned: {len(all_pinned_ids)}\n")

    print()
    print("=" * 60)
    print("  ✅  Descriptions + groups written into graph JSON")
    print("  👉  Just refresh the viewer page to see them!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    push_to_graph()