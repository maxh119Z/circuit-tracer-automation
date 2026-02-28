"""
Step 5 — Register the graph in the viewer dropdown.

Copies test-run.json to a uniquely named file (based on the prompt) and adds it
to graph-metadata.json so it appears in the viewer's dropdown selector. This lets
you run the pipeline on multiple prompts and switch between them in the UI.

Reads:
  - ../test_graphs/test-run.json  (the finalized graph with qParams)

Outputs:
  - ../test_graphs/<slug>.json    (named copy of the graph)
  - ../test_graphs/graph-metadata.json  (updated with the new entry)

Usage:
    python update_metadata.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from config import (
    GRAPH_FILE,
    setup_logging,
)

log = setup_logging()

# graph-metadata.json lives alongside the graph files
METADATA_FILE = GRAPH_FILE.parent / "graph-metadata.json"


def generate_slug(prompt: str) -> str:
    """Create a clean filename slug from the prompt text."""
    clean = prompt.replace("<bos>", "")
    clean = re.sub(r"[^a-zA-Z0-9]", "-", clean)
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean[:40].lower() if clean else "custom-run"


def main() -> None:
    if not GRAPH_FILE.exists():
        log.error("Graph file not found: %s", GRAPH_FILE)
        return

    # ------------------------------------------------------------------
    # 1. Read the finalized graph (already has qParams from push_to_website.py)
    # ------------------------------------------------------------------
    with open(GRAPH_FILE, "r") as f:
        graph_data = json.load(f)

    file_metadata = graph_data.get("metadata", {})
    prompt_str = file_metadata.get("prompt", "")
    input_tokens = file_metadata.get("prompt_tokens", [])

    if not prompt_str and input_tokens:
        prompt_str = "".join(str(t) for t in input_tokens)
    if not prompt_str:
        log.warning("No prompt found in graph metadata — using 'unknown-prompt'")
        prompt_str = "unknown-prompt"

    slug = generate_slug(prompt_str)
    log.info("Prompt: %s", prompt_str)
    log.info("Slug:   %s", slug)

    # ------------------------------------------------------------------
    # 2. Update the slug inside the graph data so the viewer knows its own identity
    # ------------------------------------------------------------------
    file_metadata["slug"] = slug
    graph_data["metadata"] = file_metadata

    # ------------------------------------------------------------------
    # 3. Save a uniquely named copy of the graph
    # ------------------------------------------------------------------
    named_graph_path = GRAPH_FILE.parent / f"{slug}.json"
    with open(named_graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    log.info("Saved graph copy → %s", named_graph_path)

    # ------------------------------------------------------------------
    # 4. Update graph-metadata.json (the dropdown registry)
    # ------------------------------------------------------------------
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
    else:
        log.info("Creating new graph-metadata.json")
        metadata = {"graphs": []}

    # Remove existing entry with same slug (so we always have the latest)
    graphs_list = metadata.get("graphs", [])
    graphs_list = [g for g in graphs_list if g.get("slug") != slug]

    # Build the new entry
    new_entry = {
        "slug": slug,
        "scan": file_metadata.get("scan", "mwhanna/gemma-scope-transcoders"),
        "transcoder_list": file_metadata.get("transcoder_list", []),
        "prompt_tokens": input_tokens,
        "prompt": prompt_str,
        "node_threshold": file_metadata.get("node_threshold", 0.8),
        "schema_version": file_metadata.get("schema_version", 1),
    }

    graphs_list.append(new_entry)
    metadata["graphs"] = graphs_list

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Registered '%s' in dropdown (%d total graphs)", slug, len(graphs_list))
    print()
    print(f"  ✅  Graph '{slug}' is now in the viewer dropdown.")
    print(f"      Select it from the dropdown menu and groups load automatically.")
    print()


if __name__ == "__main__":
    main()