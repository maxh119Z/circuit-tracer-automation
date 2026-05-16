"""
Step 1 — Fetch activating text for pruned circuit features.

Reads the attribution graph, filters nodes by type/score/connectivity,
then downloads top-activation examples from HuggingFace for each feature.
It also fetches the top/bottom logits (what the feature promotes vs suppresses).

Usage:
    PRUNING_THRESHOLD=0.40 python custom_automation/pipeline/fetch_all_activation_text.py
"""

from __future__ import annotations

import concurrent.futures
import gzip
import io
import json
import os
from collections import Counter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACTS_DIR,
    DEFAULT_PRUNING_THRESHOLD,
    GRAPH_FILE,
    HF_FEATURES_BASE,
    MAX_LAYER_INDEX,
    MAX_WORKERS,
    PRUNED_ACTIVATIONS_FILE,
    make_session,
    setup_logging,
)

log = setup_logging()

try:
    PRUNING_THRESHOLD = float(os.environ.get("PRUNING_THRESHOLD", DEFAULT_PRUNING_THRESHOLD))
except ValueError:
    PRUNING_THRESHOLD = DEFAULT_PRUNING_THRESHOLD


# ---------------------------------------------------------------------------
# Graph Loading & Pruning
# ---------------------------------------------------------------------------


def load_pruned_nodes_from_graph(graph_path: Path = GRAPH_FILE) -> list[dict]:
    """Load the attribution graph and return nodes that pass all filters.

    Filters applied (in order):
        1. **Type** — only transcoder feature nodes (excludes error / embedding / logit nodes).
           Note: the graph builder hardcodes ``feature_type="cross layer transcoder"`` on every
           transcoder feature node regardless of which suite produced it. This pipeline uses
           single-layer (per-layer) GemmaScope transcoders (``--transcoder_set gemma`` →
           ``mwhanna/gemma-scope-transcoders``); the string is just the frontend label.
        2. **Score** — influence/score <= ``PRUNING_THRESHOLD``.
        3. **Connectivity** — node must have at least one link (no orphans).
        4. **Layer cap** — ``layer_idx < MAX_LAYER_INDEX``.
    """
    if not graph_path.exists():
        log.error("Graph file not found: %s", graph_path)
        return []

    log.info("Loading graph from %s", graph_path)
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_nodes = data.get("nodes", [])
    all_links = data.get("links", [])

    # Build set of node IDs that participate in at least one link.
    connected_ids: set[str] = set()
    for link in all_links:
        connected_ids.add(link["source"])
        connected_ids.add(link["target"])

    log.info(
        "Pruning — threshold: %.2f | type: transcoder feature nodes (single-layer GemmaScope PLTs) | orphan filter: on",
        PRUNING_THRESHOLD,
    )

    pruned: list[dict] = []
    for node in all_nodes:
        # graph_models.py hardcodes this label on every transcoder feature node regardless
        # of suite; matching it just keeps feature nodes and drops error/embedding/logit nodes.
        feature_type = node.get("feature_type", "unknown")
        if feature_type != "cross layer transcoder":
            continue

        score = float(node.get("influence") or node.get("score") or 0.0)
        if score > PRUNING_THRESHOLD:
            continue

        node_id = node.get("node_id", "")
        if node_id not in connected_ids:
            continue

        try:
            parts = node_id.split("_")
            layer_idx = int(parts[0])
            feature_idx = int(parts[1])

            ctx_idx = node.get("ctx_idx")
            if ctx_idx is None and len(parts) > 2:
                ctx_idx = int(parts[2])
            if ctx_idx is None:
                ctx_idx = 0

            if layer_idx >= MAX_LAYER_INDEX:
                continue

            pruned.append(
                {
                    "layer": layer_idx,
                    "feature": feature_idx,
                    "ctx_idx": ctx_idx,
                    "score": score,
                    "type": feature_type,
                }
            )
        except (KeyError, IndexError, ValueError) as exc:
            log.debug("Skipping node %s: %s", node_id, exc)
            continue

    return pruned


# ---------------------------------------------------------------------------
# HuggingFace Data Fetching
# ---------------------------------------------------------------------------


def load_global_index(session) -> dict | None:
    """Download and decompress the feature index from HuggingFace."""
    url = f"{HF_FEATURES_BASE}/index.json.gz"
    log.info("Downloading global index from %s", url)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as f:
            return json.load(f)
    except Exception as exc:
        log.error("Failed to load global index: %s", exc)
        return None


def fetch_single_feature(node: dict, index_data: dict, session) -> dict | None:
    """Download activation data for a single feature using byte-range requests."""
    layer = str(node["layer"])
    feature_idx = node["feature"]

    layer_info = index_data.get(layer)
    if layer_info is None:
        return None

    offsets = layer_info["offsets"]
    if feature_idx >= len(offsets) - 1:
        return None

    start = offsets[feature_idx]
    end = offsets[feature_idx + 1]
    url = f"{HF_FEATURES_BASE}/{layer_info['filename']}"
    headers = {"Range": f"bytes={start}-{end - 1}"}

    try:
        resp = session.get(url, headers=headers, timeout=15)
        content = resp.content
        gzip_start = content.find(b"\x1f\x8b")
        if gzip_start != -1:
            data = json.loads(gzip.decompress(content[gzip_start:]))
        else:
            data = json.loads(content)
        return _parse_feature(node, data)
    except Exception as exc:
        log.debug("Failed to fetch layer=%s feature=%s: %s", layer, feature_idx, exc)
        return None


def _parse_feature(node: dict, raw_json: dict) -> dict:
    """Extract a clean activation summary from the raw HuggingFace JSON, including global output logits."""
    unique_id = f"{node['layer']}_{node['feature']}_{node['ctx_idx']}"

    parsed: dict = {
        "id": unique_id,
        "layer": node["layer"],
        "ctx_idx": node["ctx_idx"],
        "influence_score": node["score"],
        "top_activations": [],
        "promotes": [],
        "demotes": [],
    }

    # --- 1. Extract Input Activations ---
    examples = (
        raw_json.get("examples_quantiles", [{}])[0].get("examples", [])
        or raw_json.get("activations", [])
    )

    for ex in examples[:5]:
        tokens = ex.get("tokens", [])
        scores = ex.get("tokens_acts_list", []) or ex.get("values", [])
        if scores and len(scores) == len(tokens):
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
            parsed["top_activations"].append(
                {
                    "triggers": [str(tokens[i]) for i in top_indices],
                    "score": round(scores[top_indices[0]], 2),
                    "context": "".join(str(t) for t in tokens).replace("\n", " "),
                }
            )

    # --- 2. Extract Global Output Tokens ---
    # For promoting and demoting logits.
    top_logits = raw_json.get("top_logits", [])
    bottom_logits = raw_json.get("bottom_logits", [])

    parsed["promotes"] = [str(token) for token in top_logits]
    parsed["demotes"] = [str(token) for token in bottom_logits]

    return parsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    nodes = load_pruned_nodes_from_graph()

    if not nodes:
        log.warning("No nodes found meeting the pruning criteria.")
        return

    layer_counts = Counter(n["layer"] for n in nodes)
    log.info("Pruned circuit summary:")
    for layer in sorted(layer_counts):
        log.info("  Layer %02d: %d nodes", layer, layer_counts[layer])

    session = make_session()
    global_index = load_global_index(session)
    if global_index is None:
        return

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(fetch_single_feature, n, global_index, session): n for n in nodes
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()
            if res:
                results.append(res)
            if i % 10 == 0:
                log.info("Downloaded %d / %d features…", i, len(nodes))

    output_path = PRUNED_ACTIVATIONS_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Done — %d features saved to %s", len(results), output_path)


if __name__ == "__main__":
    main()