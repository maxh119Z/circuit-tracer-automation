"""
Standalone utility — inspect top activations for a single feature.

Useful for debugging / spot-checking before running the full pipeline.

Usage:
    python fetch_1.py --layer 14 --feature 2268
    python fetch_1.py -l 0 -f 7118
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys

from config import HF_FEATURES_BASE, make_session, setup_logging

log = setup_logging()


def get_feature_data(session, layer: int, feature: int) -> dict | None:
    """Fetch and decompress activation data for a single (layer, feature)."""
    index_url = f"{HF_FEATURES_BASE}/index.json.gz"
    log.info("Fetching index …")
    resp = session.get(index_url, timeout=30)
    resp.raise_for_status()

    with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as f:
        index_data = json.load(f)

    layer_info = index_data.get(str(layer))
    if layer_info is None:
        log.error("Layer %d not found in the index.", layer)
        return None

    offsets = layer_info["offsets"]
    if feature >= len(offsets) - 1:
        log.error("Feature %d out of range for layer %d (max %d).", feature, layer, len(offsets) - 2)
        return None

    start, end = offsets[feature], offsets[feature + 1]
    filename = layer_info["filename"]
    url = f"{HF_FEATURES_BASE}/{filename}"
    log.info("Fetching bytes %d–%d from %s …", start, end, filename)

    resp = session.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=15)
    content = resp.content

    gzip_start = content.find(b"\x1f\x8b")
    if gzip_start != -1:
        return json.loads(gzip.decompress(content[gzip_start:]))
    return json.loads(content)


def display(data: dict, layer: int, feature: int) -> None:
    """Pretty-print the top activation examples."""
    print(f"\nLayer {layer}, Feature {feature} — Top Activations")
    print("-" * 60)

    examples = (
        data.get("examples_quantiles", [{}])[0].get("examples", [])
        or data.get("activations", [])
    )

    for i, ex in enumerate(examples[:10], 1):
        tokens = ex.get("tokens", [])
        scores = ex.get("tokens_acts_list") or ex.get("values") or []

        if scores and len(scores) == len(tokens):
            max_score = max(scores)
            main_token = str(tokens[scores.index(max_score)])
            full_text = "".join(str(t) for t in tokens).replace("\n", " ")
            print(f"  Example {i}:")
            print(f"    TRIGGER : '{main_token}' (score: {max_score:.2f})")
            print(f"    CONTEXT : \"{full_text[:120]}…\"")
            print()
        else:
            print(f"  Example {i}: token/score length mismatch ({len(tokens)} vs {len(scores)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect activations for a single feature.")
    parser.add_argument("-l", "--layer", type=int, required=True, help="Transcoder layer index")
    parser.add_argument("-f", "--feature", type=int, required=True, help="Feature index within the layer")
    args = parser.parse_args()

    session = make_session()
    data = get_feature_data(session, args.layer, args.feature)
    if data is None:
        sys.exit(1)
    display(data, args.layer, args.feature)


if __name__ == "__main__":
    main()
