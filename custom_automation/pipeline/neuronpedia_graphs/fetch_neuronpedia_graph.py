"""
Fetch a full Neuronpedia attribution graph (JSON + supernode labels) into the
local layout, ready for the validation pipeline.

What this does, end to end:
  1. Parses model_id + slug + supernodes out of a Neuronpedia share URL.
  2. Downloads the graph JSON itself, trying a list of known endpoints.
  3. Writes:
       test_graphs/<local_slug>.json                          (the graph)
       custom_automation/artifacts/<local_slug>/manual_groups.json
                                                               (human labels)
  4. Prints a sanity report — prompt, node count, supernode count, and how
     many supernode IDs actually exist as nodes in the graph.

Notes:
  - The "Normal URL", "iFrame embed URL", and "HTML code snippet" share buttons
    all encode the same data (the embed-only variants just append &embed=true).
    Pass any of them — the parser ignores the extra flags.
  - The graph file URL on Neuronpedia is not standardized in their public docs.
    If the built-in candidates fail, open the Neuronpedia page, then in browser
    DevTools Network tab find the request that downloads the JSON, and pass
    its URL via --graph-url.

Usage:
  python fetch_neuronpedia_graph.py \\
      --url "https://www.neuronpedia.org/gemma-2-2b/graph?slug=gemma-fact-dallas-austin&...&supernodes=..." \\
      --slug dallas-austin

  # Bypass auto-detect and provide the JSON URL directly:
  python fetch_neuronpedia_graph.py --url "<share>" --slug X --graph-url "<json>"

  # Skip writing manual_groups.json (graph only):
  python fetch_neuronpedia_graph.py --url "<share>" --slug X --no-supernodes
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PACKAGE_DIR, REPO_ROOT, setup_logging

# Reuse the small parsers from the supernode fetcher.
from fetch_neuronpedia_groups import (
    parse_supernodes_from_url,
    slug_from_url,
    supernodes_to_manual_groups,
    write_manual_groups,
)

log = setup_logging()

ARTIFACTS_ROOT = PACKAGE_DIR / "artifacts"
TEST_GRAPHS_DIR = REPO_ROOT / "test_graphs"

# Endpoints that have served Neuronpedia graph JSONs in the wild. Tried in
# order; first one returning valid graph schema wins. Override via --graph-url
# when none work.
GRAPH_URL_TEMPLATES: list[str] = [
    "https://www.neuronpedia.org/api/graph/{model_id}/{slug}",
    "https://www.neuronpedia.org/api/graph/data/{model_id}/{slug}",
    "https://www.neuronpedia.org/api/graph/{slug}",
]

REQUEST_HEADERS = {
    "User-Agent": "circuit-tracer-automation/0.1 (+local research script)",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

def model_id_from_url(url: str) -> str | None:
    """Pull the model id (e.g. 'gemma-2-2b') from the URL path."""
    parts = [p for p in urlparse(url).path.split("/") if p]
    if not parts:
        return None
    # Path is typically /<model_id>/graph
    return parts[0]


# ---------------------------------------------------------------------------
# HTTP — try candidate graph URLs until one returns valid graph JSON
# ---------------------------------------------------------------------------

def _looks_like_graph(data: object) -> bool:
    """Minimal schema check: real graph JSONs have metadata + nodes."""
    return (
        isinstance(data, dict)
        and isinstance(data.get("metadata"), dict)
        and isinstance(data.get("nodes"), list)
    )


def _http_get_json(url: str, timeout: int = 30) -> object | None:
    """GET `url` and return parsed JSON, or None on any failure."""
    log.debug("trying %s", url)
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        log.debug("  %s → HTTP %d", url, exc.code)
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        log.debug("  %s → %s", url, exc)
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.debug("  %s → bad JSON: %s", url, exc)
        return None


def _resolve_graph_payload(data: object) -> dict | None:
    """Return a graph dict from a top-level response, unwrapping common shapes.

    Neuronpedia's `/api/graph/<model>/<slug>` returns a metadata record with
    a `url` field that points at the actual graph JSON on S3. We follow that
    indirection here. Other endpoints sometimes wrap under `graph`/`data`.
    """
    if _looks_like_graph(data):
        return data  # type: ignore[return-value]
    if isinstance(data, dict):
        # Indirection: metadata record → fetch the JSON it points at.
        s3_url = data.get("url")
        if isinstance(s3_url, str) and s3_url.startswith("http"):
            log.info("  metadata record → following url=%s", s3_url)
            inner = _http_get_json(s3_url)
            if _looks_like_graph(inner):
                return inner  # type: ignore[return-value]
        # Wrappers: {"graph": {...}}, etc.
        for key in ("graph", "data", "result"):
            inner = data.get(key)
            if _looks_like_graph(inner):
                return inner
    return None


def fetch_graph_json(
    model_id: str,
    slug: str,
    explicit_url: str | None = None,
) -> dict:
    """Fetch the graph JSON from Neuronpedia. Raises if no candidate works."""
    candidates = (
        [explicit_url]
        if explicit_url
        else [tmpl.format(model_id=model_id, slug=slug) for tmpl in GRAPH_URL_TEMPLATES]
    )
    for url in candidates:
        log.info("Fetching graph from %s …", url)
        data = _http_get_json(url)
        if data is None:
            log.info("  ✗ request failed")
            continue
        graph = _resolve_graph_payload(data)
        if graph is not None:
            log.info("  ✓ valid graph (%d nodes)", len(graph.get("nodes", [])))
            return graph
        log.info("  ✗ unexpected payload shape")
    raise RuntimeError(
        "Could not download the graph JSON from any known endpoint. "
        "Open the Neuronpedia page in a browser, watch DevTools → Network, "
        "find the request that returns the graph JSON, and re-run with "
        "--graph-url <that_url>."
    )


# ---------------------------------------------------------------------------
# Sanity report
# ---------------------------------------------------------------------------

def sanity_report(
    graph: dict,
    manual_groups: dict[str, str],
    local_slug: str,
    np_slug: str,
) -> None:
    """Print human-friendly sanity info for the freshly downloaded graph."""
    meta = graph.get("metadata", {}) or {}
    nodes = graph.get("nodes", []) or []

    prompt = meta.get("prompt") or "".join(meta.get("prompt_tokens", []) or [])
    node_ids = {str(n.get("node_id", "")) for n in nodes}
    n_groups = len(set(manual_groups.values())) if manual_groups else 0
    matched = sum(1 for fid in manual_groups if fid in node_ids)

    print("\n=== Neuronpedia graph fetched ===")
    print(f"  local slug     : {local_slug}")
    print(f"  np  slug       : {np_slug}")
    print(f"  prompt         : {prompt}")
    print(f"  nodes          : {len(nodes)}")
    print(f"  scan/transcoder: {meta.get('scan', '?')}")
    print(f"  supernodes     : {n_groups} groups, {len(manual_groups)} feature IDs")
    if manual_groups:
        status = "[OK]" if matched == len(manual_groups) else "[WARN: some IDs not in graph]"
        print(f"  IDs in graph   : {matched}/{len(manual_groups)} {status}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="Neuronpedia share URL (Normal, iFrame, or HTML — any).")
    parser.add_argument("--slug", required=True, help="Local slug to write under (artifacts/<slug>/, test_graphs/<slug>.json).")
    parser.add_argument("--graph-url", help="Direct URL to the graph JSON. Overrides built-in candidates.")
    parser.add_argument("--no-supernodes", action="store_true", help="Do not write manual_groups.json.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing test_graphs/<slug>.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan but do not write any files.")
    args = parser.parse_args()

    np_slug = slug_from_url(args.url)
    if not np_slug:
        log.error("Could not find `slug=` in the URL. Make sure you copied the full share URL.")
        sys.exit(1)

    model_id = model_id_from_url(args.url) or "gemma-2-2b"

    # 1. Fetch graph JSON
    graph = fetch_graph_json(model_id, np_slug, explicit_url=args.graph_url)

    # 2. Decide where to save it
    out_path = TEST_GRAPHS_DIR / f"{args.slug}.json"
    if out_path.exists() and not args.force and not args.dry_run:
        log.error("%s already exists. Pass --force to overwrite.", out_path)
        sys.exit(1)

    # 3. Parse supernodes from URL (independent of the graph download)
    supernodes = parse_supernodes_from_url(args.url) if not args.no_supernodes else []
    manual_groups = supernodes_to_manual_groups(supernodes) if supernodes else {}

    # 4. Write everything (or just plan)
    if args.dry_run:
        log.info("[dry-run] Would write graph to %s", out_path)
        if manual_groups:
            log.info("[dry-run] Would write %d manual entries to artifacts/%s/manual_groups.json",
                     len(manual_groups), args.slug)
    else:
        TEST_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
        log.info("Wrote graph → %s", out_path)
        if manual_groups:
            write_manual_groups(manual_groups, args.slug, dry_run=False)

    sanity_report(graph, manual_groups, args.slug, np_slug)


if __name__ == "__main__":
    main()
