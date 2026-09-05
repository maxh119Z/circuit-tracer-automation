#!/usr/bin/env python3
"""
build_views.py — turn our SLT `test_graphs/` into shareable Neuronpedia links.

Approach (see `plan.md`): we do NOT upload our graph JSON. We ask Neuronpedia to
*generate* the graph natively from the same prompt (so every node resolves —
descriptions, pos/neg logits, exemplars), then *save our view* (pinnedIds +
supernodes + clerps) onto it as a subgraph. The share URL is short and stable:

    https://neuronpedia.org/{model}/graph?slug={graph_slug}&subgraph={subgraph_id}

Pipeline, per graph:
  1. Pull `gemma-2/capital/test_graphs/*` from the HF dataset (once, up front).
  2. Read `metadata.prompt` (keep the leading `<bos>`) + `qParams`
     (`pinnedIds`, `supernodes`), and reshape node clerps -> [[node_id, clerp], ...].
  3. `get(model, graph_slug)` -> reuse if it already exists, else `generate(...)`.
  4. `POST /api/graph/subgraph/save` with our view -> `subgraphId`.
  5. Append `slug, prompt, graph_slug, subgraph_id, url` to the CSV — incrementally,
     so a crash mid-batch keeps progress and a re-run resumes.

Auth:
    HF_TOKEN             pull graphs from HuggingFace (or a cached `hf auth login`)
    NEURONPEDIA_API_KEY  generate graphs + save subgraphs (neuronpedia.org/account)

Usage:
    # checkpoint (a): pull the graphs and report what is readable
    python viewing_graph/build_views.py --download-only

    # checkpoints (b)/(c): one graph end-to-end
    python viewing_graph/build_views.py --slug-prefix slt --limit 1

    # checkpoint (d): the full capitals batch
    python viewing_graph/build_views.py --slug-prefix slt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

VIEWING_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIEWING_DIR.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_ID = "circuit-tracer-automation/pipeline_automation"

# MVP is capitals only. P2 adds the `--source` switch; only this subpath changes.
SOURCE = "capital"
HF_SUBPATH = "gemma-2/{source}/test_graphs"

# Neuronpedia only supports gemma-2-2b for native generation.
MODEL_ID = "gemma-2-2b"
NP_BASE_URL = "https://neuronpedia.org"

# Neuronpedia's generate defaults (as sent by the `neuronpedia` python client).
# Its edgeThreshold (0.98) is stricter than the webapp's own default (0.85); it
# only prunes links, never nodes, so node_id matching is unaffected either way.
DEFAULT_MAX_N_LOGITS = 10
DEFAULT_DESIRED_LOGIT_PROB = 0.95
DEFAULT_NODE_THRESHOLD = 0.8
DEFAULT_EDGE_THRESHOLD = 0.98

# Server-side cap on the generate endpoint.
GRAPH_MAX_TOKENS = 64

# Neuronpedia rate-limits graph generation (~30 graphs trips it, and it stays
# tripped for the best part of an hour). Wait it out rather than failing a graph.
DEFAULT_RATE_LIMIT_WAIT = 300.0
DEFAULT_RATE_LIMIT_ATTEMPTS = 15

# Neuronpedia slugs are globally unique across all users *and* `generate` is
# unauthenticated, so any slug can already be held by someone else. Ours carry a
# project prefix: `ct-slt-v1-<our slug>`. All 100 capital targets were free under
# this prefix; `ct-slt` was not (albuquerque was taken).
SLUG_PREFIX = "ct-slt-v1"

CSV_FIELDS = ["slug", "prompt", "graph_slug", "subgraph_id", "url"]
FAILURE_FIELDS = ["slug", "stage", "error"]

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_views")


# ---------------------------------------------------------------------------
# Graph JSON -> our view
# ---------------------------------------------------------------------------

@dataclass
class GraphView:
    """The parts of a `test_graphs/<slug>.json` that make up our Neuronpedia view."""

    slug: str
    prompt: str
    pinned_ids: list[str]
    supernodes: list[list[str]]
    clerps: list[list[str]] = field(default_factory=list)
    pruning_threshold: float | None = None
    density_threshold: float | None = None

    @property
    def supernode_members(self) -> list[str]:
        """Every node id that belongs to a supernode, in order, deduplicated."""
        seen: dict[str, None] = {}
        for group in self.supernodes:
            for node_id in group[1:]:
                seen.setdefault(node_id, None)
        return list(seen)


def _as_pinned_list(value: Any) -> list[str]:
    """`pinnedIds` is a comma-string in our JSON but a list in the viewer's model."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _as_supernode_list(value: Any) -> list[list[str]]:
    """`supernodes` is a JSON-string in our JSON but a list-of-lists in the viewer's model."""
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    groups: list[list[str]] = []
    for entry in value:
        if not entry:
            continue
        groups.append([str(item) for item in entry])
    return groups


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_view(path: Path, include_pinned_clerps: bool = False) -> GraphView:
    """Read one `test_graphs/<slug>.json` into the view we will save on Neuronpedia."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    metadata = data.get("metadata") or {}
    prompt = metadata.get("prompt")
    if not prompt:
        raise ValueError("metadata.prompt is missing or empty")

    q_params = data.get("qParams") or {}
    supernodes = _as_supernode_list(q_params.get("supernodes"))
    pinned_ids = _as_pinned_list(q_params.get("pinnedIds"))

    view = GraphView(
        slug=metadata.get("slug") or path.stem,
        prompt=prompt,
        pinned_ids=pinned_ids,
        supernodes=supernodes,
        pruning_threshold=_as_float(q_params.get("pruningThreshold")),
        density_threshold=_as_float(q_params.get("densityThreshold")),
    )

    # Every supernode member should be pinned; fall back to the members if the
    # graph was written without an explicit pinnedIds string.
    if not view.pinned_ids:
        view.pinned_ids = view.supernode_members

    view.clerps = collect_clerps(data, view, include_pinned=include_pinned_clerps)
    return view


def collect_clerps(data: dict, view: GraphView, include_pinned: bool = False) -> list[list[str]]:
    """Reshape the clerps already on the nodes into the [[node_id, clerp], ...] the API wants.

    Nothing is generated here — these are our descriptions, which
    `push_to_website.py` writes onto `node["clerp"]`. Nodes without one (`Emb:`,
    logits) are left out so Neuronpedia renders its native label.
    """
    wanted = view.supernode_members
    if include_pinned:
        wanted = list(dict.fromkeys(wanted + view.pinned_ids))
    if not wanted:
        return []

    by_node_id = {
        str(node.get("node_id")): (node.get("clerp") or "").strip()
        for node in data.get("nodes") or []
    }
    return [[node_id, by_node_id[node_id]] for node_id in wanted if by_node_id.get(node_id)]


def make_graph_slug(prefix: str, slug: str) -> str:
    """Neuronpedia slugs must be globally unique and alphanumeric + `_` + `-`."""
    combined = f"{prefix}-{slug}" if prefix else slug
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", combined).strip("-").lower()
    if len(cleaned) < 2:
        raise ValueError(f"slug {combined!r} is too short after sanitizing")
    return cleaned


def share_url(model_id: str, graph_slug: str, subgraph_id: str) -> str:
    return f"{NP_BASE_URL}/{model_id}/graph?slug={graph_slug}&subgraph={subgraph_id}"


# ---------------------------------------------------------------------------
# Neuronpedia
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _is_not_found(exc: Exception) -> bool:
    return "Resource not found" in str(exc)


def _is_retryable(exc: Exception) -> bool:
    """Rate limits, 5xx, and transport hiccups are worth another try; 4xx are not."""
    import requests
    from neuronpedia.requests.base_request import NPInvalidResponseError, NPRateLimitError

    if isinstance(exc, (NPRateLimitError, NPInvalidResponseError)):
        return True
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    # The client flattens HTTP errors to a message string, so match on that.
    return isinstance(exc, requests.exceptions.HTTPError) and "Server error occurred" in str(exc)


def with_retries(
    fn: Callable[[], T],
    *,
    what: str,
    attempts: int,
    base_delay: float,
    rate_limit_wait: float = DEFAULT_RATE_LIMIT_WAIT,
    rate_limit_attempts: int = DEFAULT_RATE_LIMIT_ATTEMPTS,
) -> T:
    """Retry `fn`, backing off differently for the two things that actually go wrong.

    Transient errors (5xx, dropped connections) clear in seconds, so they get a
    short exponential backoff. Neuronpedia's rate limiter does not: generating
    ~30 graphs trips it and it stays tripped for the best part of an hour, so a
    429 gets a long flat wait and its own generous attempt budget.
    """
    from neuronpedia.requests.base_request import NPRateLimitError

    transient, limited = 0, 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — re-raised below when not retryable
            if isinstance(exc, NPRateLimitError):
                limited += 1
                if limited >= rate_limit_attempts:
                    raise
                log.warning("%s rate-limited (%d/%d) — waiting %.0fs",
                            what, limited, rate_limit_attempts, rate_limit_wait)
                time.sleep(rate_limit_wait)
                continue

            transient += 1
            if transient >= attempts or not _is_retryable(exc):
                raise
            delay = base_delay * 2 ** (transient - 1)
            log.warning("%s failed (attempt %d/%d): %s — retrying in %.0fs",
                        what, transient, attempts, exc, delay)
            time.sleep(delay)


class Neuronpedia:
    """Thin wrapper over the official client, plus the one REST call it lacks."""

    def __init__(
        self,
        api_key: str | None = None,
        attempts: int = 4,
        base_delay: float = 5.0,
        rate_limit_wait: float = DEFAULT_RATE_LIMIT_WAIT,
        rate_limit_attempts: int = DEFAULT_RATE_LIMIT_ATTEMPTS,
    ):
        from neuronpedia.requests.graph_request import GraphRequest

        self._graph = GraphRequest(api_key=api_key)
        self._retry_kwargs = {
            "attempts": attempts,
            "base_delay": base_delay,
            "rate_limit_wait": rate_limit_wait,
            "rate_limit_attempts": rate_limit_attempts,
        }

    def get_graph(self, model_id: str, graph_slug: str):
        """Return the graph metadata, or None if it has not been generated yet."""
        try:
            return with_retries(
                lambda: self._graph.get(model_id, graph_slug),
                what=f"get({graph_slug})", **self._retry_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_not_found(exc):
                return None
            raise

    def generate_graph(self, model_id: str, prompt: str, graph_slug: str, **params):
        try:
            return with_retries(
                lambda: self._graph.generate(model_id, prompt, graph_slug, **params),
                what=f"generate({graph_slug})", **self._retry_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            # Someone (or an earlier crashed run) already made it — reuse it.
            if "Model + Slug/ID Exists" in str(exc):
                log.info("Graph %s already exists — reusing it.", graph_slug)
                existing = self.get_graph(model_id, graph_slug)
                if existing is not None:
                    return existing
            raise

    def save_subgraph(
        self,
        model_id: str,
        graph_slug: str,
        display_name: str,
        pinned_ids: list[str],
        supernodes: list[list[str]],
        clerps: list[list[str]],
        pruning_threshold: float | None = None,
        density_threshold: float | None = None,
        overwrite_id: str | None = None,
    ) -> str:
        """POST /api/graph/subgraph/save -> subgraphId.

        `pruningThreshold` / `densityThreshold` are nullable but *required* keys in
        the server's schema, so they are always sent.
        """
        payload: dict[str, Any] = {
            "modelId": model_id,
            "slug": graph_slug,
            "displayName": display_name,
            "pinnedIds": pinned_ids,
            "supernodes": supernodes,
            "clerps": clerps,
            "pruningThreshold": pruning_threshold,
            "densityThreshold": density_threshold,
        }
        if overwrite_id:
            payload["overwriteId"] = overwrite_id

        response = with_retries(
            lambda: self._graph.send_request(method="POST", uri="subgraph/save", json=payload),
            what=f"subgraph/save({graph_slug})", **self._retry_kwargs,
        )
        subgraph_id = (response or {}).get("subgraphId")
        if not subgraph_id:
            raise ValueError(f"subgraph/save returned no subgraphId: {response}")
        return subgraph_id


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

def download_graphs(source: str, token: str | None, repo_id: str = HF_REPO_ID) -> Path:
    """Pull `gemma-2/<source>/test_graphs/*` and return the local directory."""
    from huggingface_hub import snapshot_download

    subpath = HF_SUBPATH.format(source=source)

    # (repo_type, token) pairs, tried in order. The dataset is public, so a stale
    # cached login must not block the pull: HF answers 401 and then quietly falls
    # back to whatever is already in the local cache, so we check for files
    # ourselves rather than trusting the call to raise, and retry anonymously.
    # `token=False` is what disables the cached-login lookup.
    attempts: list[tuple[str, str | bool | None]] = [
        ("dataset", token or False), ("model", token or False),
    ]
    if token:
        attempts.append(("dataset", False))

    last_error: Exception | None = None
    for repo_type, attempt_token in attempts:
        label = f"{repo_type} repo" + ("" if attempt_token else ", anonymous")
        try:
            log.info("Pulling %s/%s from HF (%s)…", repo_id, subpath, label)
            local_root = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                allow_patterns=f"{subpath}/*",
                token=attempt_token,
            )
        except Exception as exc:  # noqa: BLE001 — try the next (repo_type, token) pair
            last_error = exc
            log.debug("Not readable as a %s: %s", label, exc)
            continue

        graphs_dir = Path(local_root) / subpath
        if any(graphs_dir.glob("*.json")):
            return graphs_dir
        log.debug("No graph JSONs under %s (%s)", graphs_dir, label)

    raise SystemExit(
        f"Could not pull {repo_id}/{subpath} from HuggingFace. "
        f"Set a valid HF_TOKEN (or run `hf auth login`).\nLast error: {last_error}"
    )


# The viewer's index file lives alongside the graphs in test_graphs/ — it is a
# list of graph metadata, not a graph.
NON_GRAPH_FILENAMES = {"graph-metadata.json"}


def find_graph_files(graphs_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(
        p for p in graphs_dir.glob("*.json")
        if p.is_file() and p.name not in NON_GRAPH_FILENAMES
    )
    return files[:limit] if limit is not None else files


# ---------------------------------------------------------------------------
# CSV (the state — re-runs skip slugs already in it)
# ---------------------------------------------------------------------------

def read_done_slugs(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return {row["slug"] for row in csv.DictReader(fh) if row.get("slug") and row.get("url")}


def append_row(csv_path: Path, fields: list[str], row: dict[str, Any]) -> None:
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


# ---------------------------------------------------------------------------
# Per-graph pipeline
# ---------------------------------------------------------------------------

def build_one(
    np_client: Neuronpedia,
    view: GraphView,
    graph_slug: str,
    model_id: str,
    generate_params: dict[str, Any],
) -> tuple[str, str]:
    """Generate (or reuse) the graph, save our view onto it. Returns (graph_slug, url)."""
    existing = np_client.get_graph(model_id, graph_slug)
    if existing is not None:
        # Slugs are global and `generate` is unauthenticated, so anyone can hold
        # one. Only reuse a graph that came from *our* prompt — overlaying our
        # node_ids on a graph built from different text would group nothing and
        # silently produce a wrong view.
        if (existing.prompt or "").strip() != view.prompt.strip():
            raise ValueError(
                f"slug {graph_slug} is taken by a graph with a different prompt "
                f"({existing.prompt!r} != {view.prompt!r}) — use a different --slug-prefix"
            )
        log.info("  graph %s already generated — reusing", graph_slug)
    else:
        log.info("  generating %s (%d chars)…", graph_slug, len(view.prompt))
        np_client.generate_graph(model_id, view.prompt, graph_slug, **generate_params)

    subgraph_id = np_client.save_subgraph(
        model_id=model_id,
        graph_slug=graph_slug,
        display_name=view.slug,
        pinned_ids=view.pinned_ids,
        supernodes=view.supernodes,
        clerps=view.clerps,
        pruning_threshold=view.pruning_threshold,
        density_threshold=view.density_threshold,
    )
    return subgraph_id, share_url(model_id, graph_slug, subgraph_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build shareable Neuronpedia links from our capital test_graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--slug-prefix", default=SLUG_PREFIX,
                        help="Prefix for the Neuronpedia graph slug (keeps it globally unique).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N graphs.")
    parser.add_argument("--out", type=Path, default=VIEWING_DIR / f"links_{SOURCE}.csv",
                        help="Output CSV. Doubles as the resume state.")
    parser.add_argument("--graphs-dir", type=Path, default=None,
                        help="Use this local test_graphs dir instead of pulling from HF.")
    parser.add_argument("--download-only", action="store_true",
                        help="Pull the graphs, report what is readable, then stop (checkpoint a).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse every graph and print what would be sent — no API calls.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run slugs that are already in the CSV.")
    parser.add_argument("--include-pinned-clerps", action="store_true",
                        help="Also send clerps for pinned nodes outside any supernode.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-n-logits", type=int, default=DEFAULT_MAX_N_LOGITS)
    parser.add_argument("--desired-logit-prob", type=float, default=DEFAULT_DESIRED_LOGIT_PROB)
    parser.add_argument("--node-threshold", type=float, default=DEFAULT_NODE_THRESHOLD)
    parser.add_argument("--edge-threshold", type=float, default=DEFAULT_EDGE_THRESHOLD)
    parser.add_argument("--attempts", type=int, default=4,
                        help="Attempts per API call before giving up on a graph.")
    parser.add_argument("--retry-delay", type=float, default=5.0,
                        help="Base delay (seconds) for exponential backoff.")
    parser.add_argument("--rate-limit-wait", type=float, default=DEFAULT_RATE_LIMIT_WAIT,
                        help="Seconds to wait out a 429 before retrying.")
    parser.add_argument("--rate-limit-attempts", type=int, default=DEFAULT_RATE_LIMIT_ATTEMPTS,
                        help="How many rate-limit waits to sit through per API call.")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Seconds to pause between graphs (be kind to the generate queue).")
    return parser.parse_args(argv)


def load_env_file() -> None:
    """Pick up HF_TOKEN / NEURONPEDIA_API_KEY from a `.env`, like the NP client does."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (VIEWING_DIR / ".env", REPO_ROOT / ".env"):
        if candidate.exists():
            load_dotenv(candidate)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file()

    if args.graphs_dir:
        graphs_dir = args.graphs_dir
        if not graphs_dir.is_dir():
            raise SystemExit(f"--graphs-dir {graphs_dir} is not a directory")
    else:
        graphs_dir = download_graphs(SOURCE, token=os.environ.get("HF_TOKEN"))

    files = find_graph_files(graphs_dir, args.limit)
    if not files:
        raise SystemExit(f"No graph JSONs found in {graphs_dir}")
    log.info("Found %d graph JSON(s) in %s", len(files), graphs_dir)

    # Checkpoint (a): confirm every graph on disk parses and carries a view.
    if args.download_only or args.dry_run:
        ok = 0
        for path in files:
            try:
                view = load_view(path, include_pinned_clerps=args.include_pinned_clerps)
            except Exception as exc:  # noqa: BLE001
                log.error("%s — UNREADABLE: %s", path.name, exc)
                continue
            ok += 1
            log.info(
                "%s -> %s | prompt=%r | pinned=%d supernodes=%d clerps=%d",
                path.name, make_graph_slug(args.slug_prefix, view.slug), view.prompt,
                len(view.pinned_ids), len(view.supernodes), len(view.clerps),
            )
        log.info("%d/%d graphs readable (prompt + qParams).", ok, len(files))
        return 0 if ok == len(files) else 1

    if not os.environ.get("NEURONPEDIA_API_KEY"):
        raise SystemExit("NEURONPEDIA_API_KEY is not set (get one at neuronpedia.org/account).")

    done = set() if args.force else read_done_slugs(args.out)
    if done:
        log.info("Resuming — %d slug(s) already in %s", len(done), args.out)

    failures_path = args.out.with_suffix(".failures.csv")
    np_client = Neuronpedia(
        attempts=args.attempts,
        base_delay=args.retry_delay,
        rate_limit_wait=args.rate_limit_wait,
        rate_limit_attempts=args.rate_limit_attempts,
    )
    generate_params = {
        "max_n_logits": args.max_n_logits,
        "desired_logit_prob": args.desired_logit_prob,
        "node_threshold": args.node_threshold,
        "edge_threshold": args.edge_threshold,
    }

    succeeded, skipped, failed = 0, 0, 0
    for index, path in enumerate(files, 1):
        slug = path.stem
        if slug in done:
            skipped += 1
            continue

        log.info("[%d/%d] %s", index, len(files), slug)
        stage = "parse"
        try:
            view = load_view(path, include_pinned_clerps=args.include_pinned_clerps)
            if not view.supernodes:
                raise ValueError("no supernodes in qParams — nothing to overlay")
            graph_slug = make_graph_slug(args.slug_prefix, view.slug)
            log.info("  pinned=%d supernodes=%d clerps=%d",
                     len(view.pinned_ids), len(view.supernodes), len(view.clerps))

            stage = "neuronpedia"
            subgraph_id, url = build_one(
                np_client, view, graph_slug, args.model_id, generate_params
            )
        except Exception as exc:  # noqa: BLE001 — one bad graph must not stop the batch
            failed += 1
            log.error("  FAILED (%s): %s", stage, exc)
            append_row(failures_path, FAILURE_FIELDS,
                       {"slug": slug, "stage": stage, "error": str(exc)[:500]})
            continue

        append_row(args.out, CSV_FIELDS, {
            "slug": view.slug,
            "prompt": view.prompt,
            "graph_slug": graph_slug,
            "subgraph_id": subgraph_id,
            "url": url,
        })
        succeeded += 1
        log.info("  -> %s", url)

        if args.sleep and index < len(files):
            time.sleep(args.sleep)

    log.info("Done. %d built, %d skipped (already in CSV), %d failed.", succeeded, skipped, failed)
    if failed:
        log.info("Failures logged to %s — re-run to retry them.", failures_path)
    log.info("Links -> %s", args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
