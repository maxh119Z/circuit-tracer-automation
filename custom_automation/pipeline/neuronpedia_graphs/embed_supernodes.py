"""
Embed feature group assignments and descriptions into test_graphs/*.json.

Mirrors push_to_website.py but operates on the final named graph files rather
than test-run.json, so it can be run across all 16 slugs at once.

For each slug:
  1. Injects feature descriptions into node `clerp` fields
  2. Builds supernodes array from feature_groups and writes as compact JSON string
  3. Sets pinnedIds (comma-sep string of all grouped node IDs)
  4. Sets pruningThreshold to 1.0 (graphs already pruned by Neuronpedia)

Usage:
    python custom_automation/pipeline/embed_supernodes.py
    python custom_automation/pipeline/embed_supernodes.py --slugs gemma-addition,gemma-G
    python custom_automation/pipeline/embed_supernodes.py --clear   # remove all supernodes
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DESCRIPTION_VARIANT, GROUPING_VARIANT, PACKAGE_DIR, setup_logging

log = setup_logging()

ARTIFACTS_ROOT = PACKAGE_DIR / "artifacts"
TEST_GRAPHS_DIR = PACKAGE_DIR.parent / "test_graphs"
PRUNING_THRESHOLD = "1.0"


def embed_slug(slug: str, clear: bool) -> bool:
    graph_path = TEST_GRAPHS_DIR / f"{slug}.json"
    if not graph_path.exists():
        log.warning("[%s] graph JSON not found: %s", slug, graph_path)
        return False

    with open(graph_path, encoding="utf-8") as f:
        graph = json.load(f)

    q_params = graph.get("qParams", {})

    if clear:
        q_params["supernodes"] = "[]"
        q_params["pinnedIds"] = ""
        graph["qParams"] = q_params
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
        log.info("[%s] cleared supernodes", slug)
        return True

    artifact_dir = ARTIFACTS_ROOT / slug
    groups_path = artifact_dir / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}.json"
    desc_path = artifact_dir / f"feature_descriptions_{DESCRIPTION_VARIANT}.json"

    if not groups_path.exists():
        log.warning("[%s] feature_groups not found: %s", slug, groups_path)
        return False

    with open(groups_path, encoding="utf-8") as f:
        groups: dict[str, str] = json.load(f)

    # Load descriptions if available
    descriptions: dict[str, str] = {}
    if desc_path.exists():
        with open(desc_path, encoding="utf-8") as f:
            descriptions = {item["id"]: item["generated_description"] for item in json.load(f)}

    nodes = graph.get("nodes", [])

    # 1. Inject descriptions into node clerp fields
    desc_count = 0
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        if node_id in descriptions:
            existing = node.get("clerp", "")
            if not existing or node.get("feature_type") == "cross layer transcoder":
                node["clerp"] = descriptions[node_id]
                desc_count += 1

    # 2. Build supernodes
    group_to_nodes: dict[str, list[str]] = {}
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        group_name = groups.get(node_id)
        if group_name and group_name != "Ungrouped":
            clean_name = group_name.replace("/", "-").replace("\\", "-")
            group_to_nodes.setdefault(clean_name, []).append(node_id)

    supernodes_array: list[list[str]] = []
    all_pinned_ids: list[str] = []
    for gname, member_ids in sorted(group_to_nodes.items()):
        supernodes_array.append([gname] + member_ids)
        all_pinned_ids.extend(member_ids)

    # 3. Write into qParams
    q_params["supernodes"] = json.dumps(supernodes_array, separators=(",", ":"))
    q_params["pinnedIds"] = ",".join(all_pinned_ids)
    q_params["pruningThreshold"] = PRUNING_THRESHOLD
    q_params.pop("clerps", None)
    graph["qParams"] = q_params

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    log.info(
        "[%s] %d groups, %d pinned, %d clerps injected",
        slug, len(supernodes_array), len(all_pinned_ids), desc_count,
    )
    return True


def discover_slugs() -> list[str]:
    fg_name = f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}.json"
    return [
        child.name
        for child in sorted(ARTIFACTS_ROOT.iterdir())
        if child.is_dir()
        and (child / fg_name).exists()
        and (TEST_GRAPHS_DIR / f"{child.name}.json").exists()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slugs", help="Comma-separated slugs. Default: auto-detect.")
    parser.add_argument("--clear", action="store_true", help="Remove all supernodes instead of embedding.")
    args = parser.parse_args()

    slugs = (
        [s.strip() for s in args.slugs.split(",") if s.strip()]
        if args.slugs
        else discover_slugs()
    )
    if not slugs:
        log.error("No slugs found with both feature_groups and graph JSON.")
        sys.exit(1)

    log.info("%s %d slug(s): %s", "Clearing" if args.clear else "Embedding", len(slugs), slugs)
    ok = sum(embed_slug(s, args.clear) for s in slugs)
    log.info("Done: %d/%d succeeded", ok, len(slugs))


if __name__ == "__main__":
    main()
