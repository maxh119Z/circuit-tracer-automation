"""
Batch process a CSV of Neuronpedia share URLs into validation-ready artifacts.

For each row in the CSV, this runs the steps needed before validation:

  1. Fetch graph JSON + parse supernodes  -> test_graphs/<slug>.json
                                           + artifacts/<slug>/manual_groups.json
  2. fetch_all_activation_text.py         -> artifacts/<slug>/pruned_activations.json
  3. generate_description.py             -> artifacts/<slug>/feature_descriptions_<desc>.json
  4. generate_supernodes.py              -> artifacts/<slug>/feature_groups_<desc>_<group>.json
                                           + artifacts/<slug>/feature_groups_<desc>_<group>_pre3.json

Each step is skipped when its output already exists, so re-runs resume cleanly.
After this finishes, slugs are ready for `run_validation_sweep.py`.

CSV format (header required):
    slug,share_url[,notes]

`slug` is the local artifact dir name (your choice). `share_url` is any of
the three Neuronpedia "Share" URL variants — Normal, iFrame, or HTML embed.
`notes` is free-form (ignored by this script).

Quick start (run all 16 Anthropic demo graphs):
    cd <repo_root>
    OPENAI_API_KEY=sk-... python custom_automation/pipeline/neuronpedia_graphs/batch_fetch_neuronpedia.py \
        --csv prompts/neuronpedia_graphs.csv

Then validate:
    OPENAI_API_KEY=sk-... python custom_automation/pipeline/neuronpedia_graphs/run_validation_sweep.py
                   # re-run even if outputs exist
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import certifi

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    DESCRIPTION_VARIANT,
    GROUPING_VARIANT,
    PACKAGE_DIR,
    REPO_ROOT,
    setup_logging,
)

log = setup_logging()

PIPELINE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = PACKAGE_DIR / "artifacts"
TEST_GRAPHS_DIR = REPO_ROOT / "test_graphs"
DEFAULT_CSV = REPO_ROOT / "prompts" / "neuronpedia_graphs.csv"

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
# URL parsing helpers
# ---------------------------------------------------------------------------

def slug_from_url(url: str) -> str | None:
    qs = parse_qs(urlparse(url).query)
    val = qs.get("slug", [None])[0]
    return val.strip() if val else None


def model_id_from_url(url: str) -> str | None:
    parts = [p for p in urlparse(url).path.split("/") if p]
    return parts[0] if parts else None


def parse_supernodes_from_url(url: str) -> list[list[str]]:
    qs = parse_qs(urlparse(url).query)
    raw = qs.get("supernodes", [None])[0]
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Could not parse supernodes JSON from URL: %s", exc)
        return []
    return parsed if isinstance(parsed, list) else []


def supernodes_to_manual_groups(supernodes: list[list[str]]) -> dict[str, str]:
    """Convert [[label, fid, fid, ...], ...] → {fid: label}."""
    groups: dict[str, str] = {}
    seen_labels: dict[str, int] = {}
    for entry in supernodes:
        if not entry or len(entry) < 2:
            continue
        label = str(entry[0]).strip() or "Unnamed"
        feature_ids = [str(fid).strip() for fid in entry[1:] if fid]
        if label in seen_labels:
            seen_labels[label] += 1
            label = f"{label} ({seen_labels[label]})"
        else:
            seen_labels[label] = 1
        for fid in feature_ids:
            if fid in groups:
                log.warning("Feature %s assigned twice (%s vs %s) — keeping first.",
                            fid, groups[fid], label)
                continue
            groups[fid] = label
    return groups


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------

def _looks_like_graph(data: object) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("metadata"), dict)
        and isinstance(data.get("nodes"), list)
    )


def _http_get_json(url: str, timeout: int = 30) -> object | None:
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            body = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        log.info("  %s → %s", url, exc)
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _resolve_graph_payload(data: object) -> dict | None:
    if _looks_like_graph(data):
        return data  # type: ignore[return-value]
    if isinstance(data, dict):
        s3_url = data.get("url")
        if isinstance(s3_url, str) and s3_url.startswith("http"):
            log.info("  metadata record → following url=%s", s3_url)
            inner = _http_get_json(s3_url)
            if _looks_like_graph(inner):
                return inner  # type: ignore[return-value]
        for key in ("graph", "data", "result"):
            inner = data.get(key)
            if _looks_like_graph(inner):
                return inner
    return None


def fetch_graph_json(model_id: str, np_slug: str) -> dict:
    candidates = [tmpl.format(model_id=model_id, slug=np_slug) for tmpl in GRAPH_URL_TEMPLATES]
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
        "Could not download graph JSON from any known endpoint. "
        "Open Neuronpedia in a browser, find the graph JSON request in DevTools → Network, "
        "and verify the share URL is correct."
    )


# ---------------------------------------------------------------------------
# File writing
# ---------------------------------------------------------------------------

def fetch_and_write_graph(slug: str, url: str, force: bool) -> bool:
    """Fetch graph JSON and manual_groups.json for one slug. Returns True on success."""
    np_slug = slug_from_url(url)
    if not np_slug:
        log.error("[%s] Could not find `slug=` in URL.", slug)
        return False

    model_id = model_id_from_url(url) or "gemma-2-2b"
    out_path = TEST_GRAPHS_DIR / f"{slug}.json"

    if out_path.exists() and not force:
        log.error("[%s] %s already exists. Pass --force to overwrite.", slug, out_path)
        return False

    try:
        graph = fetch_graph_json(model_id, np_slug)
    except RuntimeError as exc:
        log.error("[%s] %s", slug, exc)
        return False

    TEST_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)
    log.info("[%s] Wrote graph → %s", slug, out_path)

    supernodes = parse_supernodes_from_url(url)
    if supernodes:
        manual_groups = supernodes_to_manual_groups(supernodes)
        artifact_dir = ARTIFACTS_ROOT / slug
        artifact_dir.mkdir(parents=True, exist_ok=True)
        groups_path = artifact_dir / "manual_groups.json"
        with open(groups_path, "w", encoding="utf-8") as f:
            json.dump(manual_groups, f, indent=2)
        log.info("[%s] Wrote %d manual group entries → %s", slug, len(manual_groups), groups_path)

        meta = graph.get("metadata", {}) or {}
        nodes = graph.get("nodes", []) or []
        node_ids = {str(n.get("node_id", "")) for n in nodes}
        matched = sum(1 for fid in manual_groups if fid in node_ids)
        n_groups = len(set(manual_groups.values()))
        log.info("[%s] prompt: %s", slug,
                 meta.get("prompt") or "".join(meta.get("prompt_tokens", []) or []))
        log.info("[%s] %d nodes, %d supernodes, %d/%d IDs matched",
                 slug, len(nodes), n_groups, matched, len(manual_groups))
    else:
        log.warning("[%s] No supernodes in URL — manual_groups.json not written.", slug)

    return True


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        log.error("CSV not found: %s", path)
        sys.exit(1)
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            slug = (r.get("slug") or r.get("local_slug") or "").strip()
            url = (r.get("share_url") or r.get("url") or "").strip()
            if not slug or slug.startswith("#"):
                continue
            if not url:
                log.warning("Row missing share_url for slug %s — skipping.", slug)
                continue
            rows.append({"slug": slug, "url": url, "notes": (r.get("notes") or "").strip()})
    return rows


# ---------------------------------------------------------------------------
# Subprocess runner (for pipeline steps that are separate scripts)
# ---------------------------------------------------------------------------

def run_step(name: str, args: list[str], env: dict[str, str], log_lines: list[str]) -> bool:
    log.info("  -> %s", name)
    proc = subprocess.Popen(
        args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(PACKAGE_DIR),
    )
    output_lines: list[str] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        if line:
            print(f"    {line}", flush=True)
            output_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        log.error("  FAIL: %s (rc=%d)", name, proc.returncode)
        if output_lines:
            log.error("    output tail:\n%s", "\n".join(output_lines[-20:]))
        log_lines.append(f"  FAIL: {name}")
        return False
    log_lines.append(f"  OK:   {name}")
    return True


# ---------------------------------------------------------------------------
# Per-slug processing
# ---------------------------------------------------------------------------

def process_one(
    slug: str,
    url: str,
    *,
    only_fetch: bool,
    force: bool,
    pruning_threshold: float | None = None,
    caps: str | None = None,
) -> tuple[bool, list[str]]:
    artifact_dir = ARTIFACTS_ROOT / slug
    graph_path = TEST_GRAPHS_DIR / f"{slug}.json"
    manual_path = artifact_dir / "manual_groups.json"
    pruned_path = artifact_dir / "pruned_activations.json"
    desc_path = artifact_dir / f"feature_descriptions_{DESCRIPTION_VARIANT}.json"
    groups_path = artifact_dir / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}.json"
    pre3_path = artifact_dir / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}_pre3.json"

    # When --caps is set, the cap-sweep driver also produces ours-full at the
    # canonical filenames, so the existence check covers both modes.
    cap_suffixed_paths: list[Path] = []
    if caps:
        for tok in caps.split(","):
            tok = tok.strip().lower()
            if not tok or tok in ("unlimited", "all", "none", "inf"):
                continue
            cap_suffixed_paths.append(
                artifact_dir / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}_cap{tok}.json"
            )

    log.info("\n=== %s ===", slug)
    steps: list[str] = []

    threshold = pruning_threshold if pruning_threshold is not None else 0.7

    env = os.environ.copy()
    env["CURRENT_SLUG"] = slug
    env["PRUNING_THRESHOLD"] = f"{threshold:.4f}"
    env.setdefault("DESCRIPTION_VARIANT", DESCRIPTION_VARIANT)
    env.setdefault("GROUPING_VARIANT", GROUPING_VARIANT)
    log.info("  pruning_threshold=%.4f", threshold)

    # --- Step 1: graph + supernodes ----------------------------------------
    have_graph = graph_path.exists() and manual_path.exists()
    if have_graph and not force:
        log.info("  skip: graph + manual_groups already present")
        steps.append("  SKIP: fetch graph (already present)")
    else:
        ok = fetch_and_write_graph(slug, url, force=force)
        steps.append("  OK:   fetch graph + supernodes" if ok else "  FAIL: fetch graph + supernodes")
        if not ok:
            return False, steps

    if only_fetch:
        return True, steps

    # --- Step 2: pruned activations ----------------------------------------
    if pruned_path.exists() and not force:
        log.info("  skip: pruned_activations.json already present")
        steps.append("  SKIP: fetch activations (already present)")
    else:
        ok = run_step(
            "fetch activations",
            [sys.executable, str(PIPELINE_DIR / "fetch_all_activation_text.py")],
            env, steps,
        )
        if not ok:
            return False, steps

    # --- Step 3: descriptions ----------------------------------------------
    if desc_path.exists() and not force:
        log.info("  skip: feature_descriptions_%s.json already present", DESCRIPTION_VARIANT)
        steps.append("  SKIP: descriptions (already present)")
    else:
        ok = run_step(
            "generate descriptions",
            [sys.executable, str(PIPELINE_DIR / "generate_description.py")],
            env, steps,
        )
        if not ok:
            return False, steps

    # --- Step 4: auto supernodes (with pre-phase-3 snapshot) ---------------
    have_canonical_groups = groups_path.exists() and pre3_path.exists()
    have_all_caps = have_canonical_groups and all(p.exists() for p in cap_suffixed_paths)
    if have_all_caps and not force:
        log.info("  skip: feature_groups + pre3 snapshot already present (and all cap variants)")
        steps.append("  SKIP: supernodes (already present)")
    else:
        if caps:
            env["CAPS"] = caps
            ok = run_step(
                f"generate supernodes (cap sweep: {caps})",
                [sys.executable, str(Path(__file__).resolve().parent / "generate_supernodes_cap_sweep.py")],
                env, steps,
            )
        else:
            ok = run_step(
                "generate supernodes",
                [sys.executable, str(PIPELINE_DIR / "generate_supernodes.py")],
                env, steps,
            )
        if not ok:
            return False, steps

    return True, steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help=f"CSV path (default: {DEFAULT_CSV}).")
    parser.add_argument("--slugs", help="Comma-separated subset of slugs to process. Default: all rows.")
    parser.add_argument("--urls", nargs="+", metavar="SLUG=URL",
                        help="One or more slug=url pairs to process directly, bypassing the CSV.")
    parser.add_argument("--only-fetch", action="store_true",
                        help="Only fetch graph + manual_groups.json. Skip activations/descriptions/groups.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every step even if outputs exist.")
    parser.add_argument("--pruning-threshold", type=float, default=None,
                        help="Pruning threshold to use for all slugs. Default: 0.7 "
                             "(matches the value most Neuronpedia share URLs encode). "
                             "Pass a different float to override.")
    parser.add_argument("--caps", default=None,
                        help="Comma-separated phase-2 feature caps to sweep, e.g. "
                             "'50,100,150,200,unlimited'. When set, uses the cap-sweep "
                             "driver; the 'unlimited' cap writes to the canonical filenames "
                             "(so ours-full validation still works) and each numeric cap "
                             "writes feature_groups_..._cap<N>.json. Default: off "
                             "(standard single-run pipeline).")
    args = parser.parse_args()

    if args.urls:
        rows = []
        for pair in args.urls:
            if "=" not in pair:
                log.error("--urls entries must be slug=url, got: %s", pair)
                sys.exit(1)
            slug, url = pair.split("=", 1)
            rows.append({"slug": slug.strip(), "url": url.strip(), "notes": ""})
    else:
        csv_path = Path(args.csv)
        rows = read_csv(csv_path)
        if not rows:
            log.error("No usable rows in %s.", csv_path)
            sys.exit(1)

    if args.slugs:
        wanted = {s.strip() for s in args.slugs.split(",") if s.strip()}
        before = len(rows)
        rows = [r for r in rows if r["slug"] in wanted]
        missing = wanted - {r["slug"] for r in rows}
        log.info("Filtered %d → %d rows.", before, len(rows))
        if missing:
            log.warning("Slugs in --slugs but not in CSV: %s", sorted(missing))

    if not args.only_fetch and not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set (required for descriptions/groups). "
                  "Pass --only-fetch to download graphs without API calls.")
        sys.exit(1)

    source = "command line" if args.urls else str(Path(args.csv))
    log.info("Batch: %d slug(s) from %s", len(rows), source)
    log.info("Mode: only_fetch=%s force=%s", args.only_fetch, args.force)
    log.info("Variants: desc=%s grouping=%s", DESCRIPTION_VARIANT, GROUPING_VARIANT)

    success: list[tuple[str, list[str]]] = []
    failed: list[tuple[str, list[str]]] = []
    for row in rows:
        ok, steps = process_one(
            row["slug"], row["url"],
            only_fetch=args.only_fetch,
            force=args.force,
            pruning_threshold=args.pruning_threshold,
            caps=args.caps,
        )
        (success if ok else failed).append((row["slug"], steps))

    print("\n" + "=" * 70)
    print(f"  Batch summary: {len(success)} ok, {len(failed)} failed (of {len(rows)})")
    print("=" * 70)
    for slug, steps in success:
        print(f"\n[OK]   {slug}")
        for s in steps:
            print(s)
    for slug, steps in failed:
        print(f"\n[FAIL] {slug}")
        for s in steps:
            print(s)

    if not args.only_fetch and success:
        print("\nNext step: validate.")
        print("  python custom_automation/pipeline/neuronpedia_graphs/run_validation_sweep.py \\")
        print(f"      --slugs {','.join(s for s, _ in success)} \\")
        print("      --min-sizes 2,3,4,5,6")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
