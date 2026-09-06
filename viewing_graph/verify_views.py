#!/usr/bin/env python3
"""
verify_views.py — check that the links in `links_<source>.csv` actually carry our view.

Opening 100 links by hand is not a check. For each row in the CSV this asks
Neuronpedia what it stored and compares it to the graph JSON we built it from:

  1. `POST /api/graph/subgraph/list` -> our saved subgraph for that graph.
     Confirms the subgraph exists and its supernodes + clerps round-tripped
     intact (this is what the `&subgraph=` link loads).
  2. `--check-nodes` (slower): downloads the natively generated graph and
     reports how many of our supernode members actually exist in it. This is
     the plan's core assumption — matching `node_id`s — and its known pitfall:
     generation pruning can drop members, which silently don't group.

Usage:
    python viewing_graph/verify_views.py                      # round-trip check, all rows
    python viewing_graph/verify_views.py --limit 1 --check-nodes
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_views import (  # noqa: E402
    DEFAULT_SOURCE,
    MODEL_ID,
    load_env_file,
    VIEWING_DIR,
    GraphView,
    Neuronpedia,
    SOURCES,
    fetch_graph_file,
    load_view,
    log,
)


def fetch_saved_subgraph(np_client: Neuronpedia, model_id: str, graph_slug: str, subgraph_id: str) -> dict | None:
    """Return the stored subgraph record, or None if it is gone."""
    response = np_client._graph.send_request(  # noqa: SLF001 — no client method for this route
        method="POST", uri="subgraph/list", json={"modelId": model_id, "slug": graph_slug}
    )
    for subgraph in (response or {}).get("subgraphs", []):
        if subgraph.get("id") == subgraph_id:
            return subgraph
    return None


def generated_node_ids(np_client: Neuronpedia, model_id: str, graph_slug: str) -> set[str]:
    """Node ids in the graph Neuronpedia generated for this slug."""
    metadata = np_client.get_graph(model_id, graph_slug)
    if metadata is None:
        return set()
    with urlopen(metadata.json_url, timeout=120) as response:  # noqa: S310 — URL comes from the API
        data = json.load(response)
    return {str(node.get("node_id")) for node in data.get("nodes", [])}


def compare(local: GraphView, saved: dict) -> list[str]:
    """Differences between the view we sent and the one Neuronpedia stored."""
    problems: list[str] = []

    saved_supernodes = [[str(x) for x in group] for group in saved.get("supernodes") or []]
    if saved_supernodes != local.supernodes:
        problems.append(
            f"supernodes differ (sent {len(local.supernodes)}, stored {len(saved_supernodes)})"
        )

    saved_clerps = {str(pair[0]): str(pair[1]) for pair in saved.get("clerps") or [] if len(pair) >= 2}
    sent_clerps = {node_id: text for node_id, text in local.clerps}
    if saved_clerps != sent_clerps:
        missing = sorted(set(sent_clerps) - set(saved_clerps))
        problems.append(
            f"clerps differ (sent {len(sent_clerps)}, stored {len(saved_clerps)}"
            + (f", missing {missing[:3]}" if missing else "") + ")"
        )

    saved_pinned = [str(x) for x in saved.get("pinnedIds") or []]
    if saved_pinned != local.pinned_ids:
        problems.append(f"pinnedIds differ (sent {len(local.pinned_ids)}, stored {len(saved_pinned)})")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the saved Neuronpedia views behind links_<source>.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, choices=sorted(SOURCES),
                        help="Which test_graphs folder the links were built from.")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Links CSV to verify (default: links_<source>.csv).")
    parser.add_argument("--graphs-dir", type=Path, default=None,
                        help="Local test_graphs dir (default: fetch each graph from HF).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--check-nodes", action="store_true",
                        help="Also download each generated graph and report supernode coverage.")
    args = parser.parse_args(argv)
    load_env_file()
    if args.csv is None:
        args.csv = VIEWING_DIR / f"links_{args.source}.csv"

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} does not exist — run build_views.py first.")
    with args.csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"{args.csv} has no rows.")

    subpath = SOURCES[args.source]
    np_client = Neuronpedia()

    def local_graph(slug: str) -> Path:
        """The graph JSON for a slug — from --graphs-dir, else fetched from HF."""
        if args.graphs_dir:
            return args.graphs_dir / f"{slug}.json"
        return fetch_graph_file(f"{subpath}/{slug}.json", os.environ.get("HF_TOKEN"))

    ok, bad = 0, 0
    for index, row in enumerate(rows, 1):
        slug, graph_slug = row["slug"], row["graph_slug"]
        try:
            local = load_view(local_graph(slug))
            saved = fetch_saved_subgraph(np_client, args.model_id, graph_slug, row["subgraph_id"])
            if saved is None:
                raise ValueError(f"subgraph {row['subgraph_id']} not found on {graph_slug}")
            problems = compare(local, saved)

            coverage = ""
            if args.check_nodes:
                node_ids = generated_node_ids(np_client, args.model_id, graph_slug)
                members = local.supernode_members
                present = [m for m in members if m in node_ids]
                coverage = f" | nodes {len(present)}/{len(members)}"
                if not node_ids:
                    problems.append("generated graph has no nodes")
                elif len(present) < len(members):
                    dropped = [m for m in members if m not in node_ids]
                    problems.append(
                        f"{len(dropped)} supernode member(s) missing from the generated "
                        f"graph, e.g. {dropped[:3]}"
                    )
        except Exception as exc:  # noqa: BLE001 — report and keep going
            bad += 1
            log.error("[%d/%d] %s — CHECK FAILED: %s", index, len(rows), slug, exc)
            continue

        if problems:
            bad += 1
            log.error("[%d/%d] %s%s — %s", index, len(rows), slug, coverage, "; ".join(problems))
        else:
            ok += 1
            log.info("[%d/%d] %s — supernodes=%d clerps=%d round-tripped%s",
                     index, len(rows), slug, len(local.supernodes), len(local.clerps), coverage)

    log.info("Verified %d/%d links.", ok, len(rows))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
