#!/usr/bin/env python3
"""
build_views.py: turn our SLT `test_graphs/` into shareable Neuronpedia links.

Idea:
0. You choose which graphs you want generated.
1. Neuronpedia API generates the graph natively
2. Neuronpedia API saves our view*(pinnedIds + supernodes + clerps) onto it as a subgraph.
    
Pipeline per graph:
  1. Fetch the graph JSON from the HF dataset (sequentially).
  2. Read `metadata.prompt` + `qParams` (`pinnedIds`, `supernodes`), and reshape node clerps -> [[node_id, clerp], ...].
  3. Generate graph (if new): `get(model, graph_slug)` -> reuse if it already exists, else `generate(...)`.
  4. Save our subgrpah: `POST /api/graph/subgraph/save` with our view -> `subgraphId`.
  5. Append `slug, prompt, graph_slug, subgraph_id, url` to the CSV sequentially.

Sources — `--source` picks a `test_graphs` folder and names the CSV:
    source             graphs  CSV
    capital               100  links_capital.csv
    math                  200  links_math.csv
    neuronpedia           105  links_neuronpedia.csv
    wikipedia             500  links_wikipedia.csv
    wikipedia_context     500  links_wikipedia_context.csv
    Note on wikipedia_context: 49 of its 500 prompts exceed the 64-token generate cap (up to 106 tokens). Those are refused by load_view before any API call; 451 are usable.
Additional Srouce
    source                                  graphs  CSV
    wikipedia_interesting (from our paper). 2       links_wikipedia_interesting.csv

Graphs are downloaded one at a time, as each is processed, because they are
large — a bulk pull of wikipedia would cost 43 GB before the first graph is even
looked at, and would ignore --first-n. Use --discard-downloads on the big sets to
keep peak disk at one graph instead of the whole set.

Auth:
    HF_TOKEN             pull graphs from HuggingFace (optional — dataset is public)
    NEURONPEDIA_API_KEY  generate graphs + save subgraphs (neuronpedia.org/account)

Usage:
    # what is in a source, without generating anything
    python viewing_graph/build_views.py --source wikipedia --download-only --first-n 5

    # generate a few links from one source
    python viewing_graph/build_views.py --source math --first-n 5

    # generate the whole source (resumes from the CSV if interrupted)
    python viewing_graph/build_views.py --source capital

Note: Neuronpedia rate-limits generation to roughly 30 graphs an hour. So 600 graphs will take 20 hours.
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
from typing import Any, Callable, TypeVar

VIEWING_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIEWING_DIR.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_ID = "circuit-tracer-automation/pipeline_automation"
SOURCES = {
    "capital": "gemma-2/capital/test_graphs",
    "math": "gemma-2/math_problems/test_graphs",
    "neuronpedia": "gemma-2/neuronpedia/test_graphs",
    "wikipedia": "gemma-2/wikipedia/test_graphs_plain",
    "wikipedia_context": "gemma-2/wikipedia_context/test_graphs",
    # Same folder as `wikipedia`, narrowed to WIKIPEDIA_INTERESTING in list_source_files.
    "wiki_interesting": "gemma-2/wikipedia/test_graphs_plain",
}
# The specific wikipedia graphs highlighted in our paper's exploration figure.
WIKIPEDIA_INTERESTING = {
    "oregon-route-53-s5-12-pl",  # "…the highway begins to follow the North" -> fork
    "2008-in-anime-s5-12-pl",    # La Maison en Petits Cubes won the Academy Award -> animated
}
DEFAULT_SOURCE = "capital"
HF_REPO_TYPE = "dataset"

# Filename globs limiting which graphs in a source we process. The neuronpedia
# folder holds two things: our annotated runs (`*_ours*`) and the 15 original
# Neuronpedia replication graphs they were derived from. The originals are the
# human-curated baseline the validation sweep scores against — their groupings
# live in `artifacts_neuronpedia/<slug>/manual_groups.json`, not in qParams — so
# they are not ours to publish. Override with --include "*".
SOURCE_INCLUDE = {
    "neuronpedia": "*_ours*",
}


MODEL_ID = "gemma-2-2b"
NP_BASE_URL = "https://neuronpedia.org"

# Neuronpedia's generate defaults
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
# project prefix: `llm-slt-<our slug>` to be unique.
SLUG_PREFIX = "llm-slt"

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
    prompt_token_count: int
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
    prompt = (metadata.get("prompt") or "").strip()

    tokenized = "".join(metadata.get("prompt_tokens") or [])
    if tokenized and prompt and tokenized != prompt:
        log.warning("%s: metadata.prompt disagrees with prompt_tokens — using the "
                    "tokenized prompt (%r, not %r)", path.name, tokenized, prompt)
    prompt = tokenized or prompt
    if not prompt:
        raise ValueError("metadata.prompt and metadata.prompt_tokens are both missing")

    q_params = data.get("qParams") or {}
    supernodes = _as_supernode_list(q_params.get("supernodes"))
    pinned_ids = _as_pinned_list(q_params.get("pinnedIds"))

    view = GraphView(
        slug=metadata.get("slug") or path.stem,
        prompt=prompt,
        prompt_token_count=len(metadata.get("prompt_tokens") or []),
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

def _hf(call, token):
    """Run an HF call with the token if there is one, else anonymously.

    The dataset is public, and a stale cached login 401s (reported as "repo not
    found") rather than degrading gracefully — so anonymous is always the last
    resort.
    """
    last_error: Exception | None = None
    for attempt_token in ([token, False] if token else [False]):
        try:
            return call(attempt_token)
        except Exception as exc:  # noqa: BLE001 — fall through to anonymous
            last_error = exc
    raise SystemExit(
        f"HuggingFace request failed. Set a valid HF_TOKEN (or run `hf auth login`).\n"
        f"Last error: {last_error}"
    )


def list_source_files(source: str, token: str | None = None,
                      include: str | None = None) -> list[str]:
    """Repo-relative paths of a source's graph JSONs, sorted, without downloading them.

    `include` is a filename glob; it defaults to the source's entry in
    SOURCE_INCLUDE, and "*" means everything.
    """
    import fnmatch

    from huggingface_hub import list_repo_files

    try:
        subpath = SOURCES[source]
    except KeyError:
        raise SystemExit(
            f"unknown --source {source!r} (known: {', '.join(sorted(SOURCES))})"
        ) from None

    everything = _hf(
        lambda t: list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE, token=t), token
    )
    files = sorted(
        p for p in everything
        if p.startswith(f"{subpath}/") and p.endswith(".json")
        and p.rsplit("/", 1)[-1] not in NON_GRAPH_FILENAMES
    )
    if not files:
        raise SystemExit(f"No graph JSONs found under {subpath} in {HF_REPO_ID}")

    # `wiki_interesting`: the wikipedia folder narrowed to exactly WIKIPEDIA_INTERESTING.
    if source == "wiki_interesting":
        files = [p for p in files if p.rsplit("/", 1)[-1][: -len(".json")] in WIKIPEDIA_INTERESTING]
        missing = WIKIPEDIA_INTERESTING - {p.rsplit("/", 1)[-1][: -len(".json")] for p in files}
        if missing:
            raise SystemExit(f"wiki_interesting: slug(s) not found under {subpath}: {sorted(missing)}")
        return files

    pattern = include if include is not None else SOURCE_INCLUDE.get(source)
    if pattern and pattern != "*":
        kept = [p for p in files if fnmatch.fnmatch(p.rsplit("/", 1)[-1], f"{pattern}.json")]
        if not kept:
            raise SystemExit(f"--include {pattern!r} matched none of the {len(files)} "
                             f"graphs under {subpath}")
        if len(kept) < len(files):
            log.info("Source %r: %d of %d graphs match %r; the rest are skipped.",
                     source, len(kept), len(files), pattern)
        files = kept
    return files


def fetch_graph_file(repo_path: str, token: str | None = None) -> Path:
    """Download one graph JSON (cached by HF) and return its local path.

    Deliberately per-file rather than `snapshot_download` of the whole folder:
    these graphs are large (wikipedia is ~86 MB each, ~43 GB for the set), so a
    bulk pull would cost tens of GB before the first graph is even processed,
    and would ignore --first-n entirely.
    """
    from huggingface_hub import hf_hub_download

    return Path(_hf(
        lambda t: hf_hub_download(HF_REPO_ID, repo_path, repo_type=HF_REPO_TYPE, token=t),
        token,
    ))


def discard_download(path: Path) -> None:
    """Drop a cached graph file so a long run does not fill the disk."""
    try:
        blob = path.resolve()
        if path.is_symlink():
            path.unlink(missing_ok=True)
        if blob.exists():
            blob.unlink()
    except OSError as exc:
        log.debug("Could not discard %s: %s", path, exc)


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
        description="Build shareable Neuronpedia links from our test_graphs on HuggingFace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, choices=sorted(SOURCES),
                        help="Which test_graphs folder to pull. Sets the default CSV name.")
    parser.add_argument("--slug-prefix", default=SLUG_PREFIX,
                        help="Prefix for the Neuronpedia graph slug (keeps it globally unique).")
    parser.add_argument("--first-n", type=int, default=None,
                        help="Process only the first N graphs, sorted by filename (default: all).")
    parser.add_argument("--include", default=None,
                        help="Filename glob selecting which graphs to process. Defaults to "
                             "the source's SOURCE_INCLUDE entry (neuronpedia: '*_ours*', which "
                             "leaves out the manual replication graphs). Pass '*' for all.")
    parser.add_argument("--discard-downloads", action="store_true",
                        help="Delete each graph JSON after use, keeping peak disk at one "
                             "graph. The big sets are 43-51 GB; the cost is that re-runs "
                             "and verification must download again.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output CSV (default: links_<source>.csv). Doubles as the resume state.")
    parser.add_argument("--graphs-dir", type=Path, default=None,
                        help="Use this local test_graphs dir instead of pulling from HF.")
    parser.add_argument("--download-only", action="store_true",
                        help="Preview: pull + parse every graph and print what would be sent "
                             "(target slug, token count, pinned/supernode/clerp counts), then stop "
                             "— no API calls.")
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
    if args.out is None:
        args.out = VIEWING_DIR / f"links_{args.source}.csv"

    hf_token = os.environ.get("HF_TOKEN")

    # Each entry is (slug, fetch) — `fetch` returns the local path, downloading
    # the graph only when we are about to read it.
    if args.graphs_dir:
        if not args.graphs_dir.is_dir():
            raise SystemExit(f"--graphs-dir {args.graphs_dir} is not a directory")
        local_files = find_graph_files(args.graphs_dir, args.first_n)
        if not local_files:
            raise SystemExit(f"No graph JSONs found in {args.graphs_dir}")
        entries = [(f.stem, (lambda f=f: f)) for f in local_files]
        where = str(args.graphs_dir)
    else:
        repo_paths = list_source_files(args.source, token=hf_token, include=args.include)
        total = len(repo_paths)
        if args.first_n is not None:
            repo_paths = repo_paths[: args.first_n]
        entries = [
            (rp.rsplit("/", 1)[-1][: -len(".json")], (lambda rp=rp: fetch_graph_file(rp, hf_token)))
            for rp in repo_paths
        ]
        where = f"{HF_REPO_ID}/{SOURCES[args.source]}"
        if args.first_n is not None and args.first_n < total:
            log.info("Source %r has %d graphs; --first-n %d selected.", args.source, total, args.first_n)

    if not entries:
        raise SystemExit("Nothing to do — --first-n selected no graphs.")
    log.info("%d graph(s) from %s", len(entries), where)

    # Checkpoint (a): confirm every graph on disk parses and carries a view.
    if args.download_only:
        ok = 0
        for slug, fetch in entries:
            path = None
            try:
                path = fetch()
                view = load_view(path, include_pinned_clerps=args.include_pinned_clerps)
            except Exception as exc:  # noqa: BLE001
                log.error("%s — UNREADABLE: %s", slug, exc)
                if args.discard_downloads and path is not None:
                    discard_download(path)
                continue
            ok += 1
            over = " OVER TOKEN CAP" if view.prompt_token_count > GRAPH_MAX_TOKENS else ""
            log.info(
                "%s -> %s | tokens=%d%s | pinned=%d supernodes=%d clerps=%d | prompt=%r",
                path.name, make_graph_slug(args.slug_prefix, view.slug),
                view.prompt_token_count, over,
                len(view.pinned_ids), len(view.supernodes), len(view.clerps), view.prompt,
            )
            if args.discard_downloads:
                discard_download(path)
        log.info("%d/%d graphs readable (prompt + qParams).", ok, len(entries))
        return 0 if ok == len(entries) else 1

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

    succeeded, skipped, no_view, failed = 0, 0, 0, 0
    for index, (slug, fetch) in enumerate(entries, 1):
        if slug in done:
            skipped += 1
            continue

        log.info("[%d/%d] %s", index, len(entries), slug)
        stage = "download"
        path = None
        try:
            path = fetch()
            stage = "parse"
            view = load_view(path, include_pinned_clerps=args.include_pinned_clerps)
            if not view.supernodes:
                # Not a failure — there is simply no view of ours to overlay.
                log.info("  no supernodes in qParams — skipping (nothing to overlay)")
                no_view += 1
                continue
            if view.prompt_token_count > GRAPH_MAX_TOKENS:
                raise ValueError(
                    f"prompt is {view.prompt_token_count} tokens, over Neuronpedia's "
                    f"{GRAPH_MAX_TOKENS}-token generate cap"
                )
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
        finally:
            if args.discard_downloads and path is not None:
                discard_download(path)

        append_row(args.out, CSV_FIELDS, {
            "slug": view.slug,
            "prompt": view.prompt,
            "graph_slug": graph_slug,
            "subgraph_id": subgraph_id,
            "url": url,
        })
        succeeded += 1
        log.info("  -> %s", url)

        if args.sleep and index < len(entries):
            time.sleep(args.sleep)

    log.info("Done. %d built, %d skipped (already in CSV), %d skipped (no groupings), "
             "%d failed.", succeeded, skipped, no_view, failed)
    if failed:
        log.info("Failures logged to %s — re-run to retry them.", failures_path)
    log.info("Links -> %s", args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
