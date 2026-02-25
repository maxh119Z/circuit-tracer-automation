"""
Step 4 — Push descriptions and groups into the graph, generate viewer URL.

Reads:
  - artifacts/feature_descriptions.json
  - artifacts/feature_groups.json
  - ../test_graphs/test-run.json

Outputs:
  - Updated test-run.json with clerp labels
  - artifacts/viewer_url.txt with a clickable URL containing supernodes & clerps

Usage:
    python push_to_graph.py
    VIEWER_URL=https://skqak5p63vr3g0-8041.proxy.runpod.net python push_to_website.py
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.parse

from config import (
    DEFAULT_PRUNING_THRESHOLD,
    FEATURE_DESCRIPTIONS_FILE,
    FEATURE_GROUPS_FILE,
    GRAPH_FILE,
    VIEWER_BASE_URL,
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

    # Descriptions (required)
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
    # 2. PROCESS NODES — (Skipping clerp injection to preserve originals)
    # ==================================================================
    nodes = graph_data.get("nodes", [])
    
    match_count = 0
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        if node_id in groups:
            # We no longer overwrite node["clerp"], node["localClerp"], or node["ppClerp"]
            match_count += 1
            
    # ==================================================================
    # 3. BUILD SUPERNODES ARRAY (for URL)
    # ==================================================================
    # Format: [["Group A", "id1", "id2"], ["Group B", "id3"], ...]
    group_to_nodes: dict[str, list[str]] = {}
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        group_name = groups.get(node_id)
        if group_name and group_name != "Ungrouped":
            group_to_nodes.setdefault(group_name, []).append(node_id)

    supernodes_array = []
    for gname, member_ids in group_to_nodes.items():
        supernodes_array.append([gname] + member_ids)

    # ==================================================================
    # 4. BUILD CLERPS ARRAY (for URL)
    # ==================================================================
    # Format: [["node_id", "description"], ...]
    clerps_array = [[nid, desc] for nid, desc in descriptions.items()]

    # ==================================================================
    # 5. BACKUP & SAVE GRAPH
    # ==================================================================
    backup_path = str(GRAPH_FILE) + ".bak"
    shutil.copy2(GRAPH_FILE, backup_path)
    log.info("Backup saved → %s", backup_path)

    with open(GRAPH_FILE, "w") as f:
        json.dump(graph_data, f, indent=2)

    log.info("Updated %d / %d nodes in graph.", match_count, len(nodes))
    log.info("Built %d supernode groups.", len(supernodes_array))

    # ==================================================================
    # 6. GENERATE VIEWER URL
    # ==================================================================
    clean_supernodes = []
    all_pinned_ids = []
    for group in supernodes_array:
        clean_name = group[0].replace("/", "-").replace("\\", "-")
        member_ids = group[1:]
        clean_supernodes.append([clean_name] + member_ids)
        all_pinned_ids.extend(member_ids)

    compact_json = json.dumps(clean_supernodes, separators=(",", ":"))
    encoded_supernodes = urllib.parse.quote_plus(compact_json)
    encoded_pinned = urllib.parse.quote_plus(",".join(all_pinned_ids))

    base = VIEWER_BASE_URL.rstrip("/") + "/"

    # Notice: &clerps parameter is removed entirely
    full_url = (
        f"{base}"
        f"?pruningThreshold={PRUNING_THRESHOLD}"
        f"&pinnedIds={encoded_pinned}"
        f"&supernodes={encoded_supernodes}"
    )

    log.info("")
    log.info("🔗 Open this URL in your browser:")
    print(full_url)
    print()

    with open(VIEWER_URL_FILE, "w") as f:
        f.write(full_url)
    log.info("URL also saved to %s", VIEWER_URL_FILE)


if __name__ == "__main__":
    push_to_graph()