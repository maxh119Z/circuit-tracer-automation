"""
Step 4 — Push descriptions and groups into the graph JSON.

Instead of generating a fragile URL with encoded supernodes/pinnedIds/clerps,
this script writes everything into the graph's `qParams` field. The viewer
already reads qParams on load (init-cg.js spreads them into visState), so
groups render automatically when you refresh the page — no URL copying needed.

Reads:
  - artifacts/feature_descriptions.json
  - artifacts/feature_groups.json
  - ../test_graphs/test-run.json

Outputs:
  - Updated test-run.json with qParams containing supernodes + pinnedIds + clerps

Usage:
    python push_to_website.py
"""

from __future__ import annotations

import json
import os
import shutil

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

    # Descriptions (optional but recommended)
    descriptions: dict[str, str] = {}
    if FEATURE_DESCRIPTIONS_FILE.exists():
        with open(FEATURE_DESCRIPTIONS_FILE, "r") as f:
            descriptions = {
                item["id"]: item["generated_description"] for item in json.load(f)
            }
    else:
        log.warning("No descriptions file found at %s", FEATURE_DESCRIPTIONS_FILE)

    # Groups (optional — pipeline works without grouping)
    groups: dict[str, str] = {}
    if FEATURE_GROUPS_FILE.exists():
        with open(FEATURE_GROUPS_FILE, "r") as f:
            groups = json.load(f)

    log.info("Loaded %d descriptions, %d group assignments.", len(descriptions), len(groups))

    # ==================================================================
    # 2. BUILD SUPERNODES ARRAY
    # ==================================================================
    # Format: [["Group A", "id1", "id2"], ["Group B", "id3"], ...]
    nodes = graph_data.get("nodes", [])

    group_to_nodes: dict[str, list[str]] = {}
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        group_name = groups.get(node_id)
        if group_name and group_name != "Ungrouped":
            # Sanitize group names (no slashes)
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
    # 3. BUILD CLERPS ARRAY
    # ==================================================================
    # Format: [["node_id", "description"], ...]
    clerps_array: list[list[str]] = []
    for nid, desc in descriptions.items():
        clerps_array.append([nid, desc])

    log.info("Built %d clerp entries.", len(clerps_array))

    # ==================================================================
    # 4. WRITE INTO qParams
    # ==================================================================
    # The viewer (init-cg.js) reads qParams and spreads them into visState.
    # Values are stored as strings — same format as URL query parameters —
    # because that's what the Save button writes and what the parsing code expects.
    q_params = graph_data.get("qParams", {})

    # supernodes: JSON string of the array
    q_params["supernodes"] = json.dumps(supernodes_array, separators=(",", ":"))

    # pinnedIds: comma-separated string of node IDs
    q_params["pinnedIds"] = ",".join(all_pinned_ids)

    # clerps: JSON string of the [id, description] pairs
    if clerps_array:
        q_params["clerps"] = json.dumps(clerps_array, separators=(",", ":"))

    # pruningThreshold
    q_params["pruningThreshold"] = PRUNING_THRESHOLD

    graph_data["qParams"] = q_params

    # ==================================================================
    # 5. BACKUP & SAVE
    # ==================================================================
    backup_path = str(GRAPH_FILE) + ".bak"
    shutil.copy2(GRAPH_FILE, backup_path)
    log.info("Backup saved → %s", backup_path)

    with open(GRAPH_FILE, "w") as f:
        json.dump(graph_data, f, indent=2)

    log.info("Graph updated with qParams → %s", GRAPH_FILE)
    log.info("  supernodes: %d groups", len(supernodes_array))
    log.info("  pinnedIds:  %d nodes", len(all_pinned_ids))
    log.info("  clerps:     %d descriptions", len(clerps_array))
    log.info("  threshold:  %s", PRUNING_THRESHOLD)

    # Also write a summary file for reference
    summary = {
        "graph_file": str(GRAPH_FILE),
        "num_supernodes": len(supernodes_array),
        "num_pinned": len(all_pinned_ids),
        "num_clerps": len(clerps_array),
        "pruning_threshold": PRUNING_THRESHOLD,
        "groups": {g[0]: len(g) - 1 for g in supernodes_array},
    }
    with open(VIEWER_URL_FILE, "w") as f:
        # Keep the file for backwards compat but write useful info instead of a URL
        f.write("# Pipeline output saved to graph qParams — just refresh the viewer.\n")
        f.write(f"# Graph file: {GRAPH_FILE}\n")
        f.write(f"# Groups: {len(supernodes_array)}, Pinned: {len(all_pinned_ids)}, Clerps: {len(clerps_array)}\n")

    print()
    print("=" * 60)
    print("  ✅  Groups written into graph JSON (qParams)")
    print("  👉  Just refresh the viewer page to see them!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    push_to_graph()