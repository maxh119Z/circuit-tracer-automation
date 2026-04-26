"""
Fetch human-curated supernode labels from Neuronpedia share URLs.

Neuronpedia encodes saved supernode groupings inside the graph share URL as the
`supernodes` query parameter — a URL-encoded JSON list of lists, where each
inner list is `[label, feature_id, feature_id, ...]`.

This script parses those URLs and writes them as `manual_groups.json` in the
target artifact directory, in the `{feature_id: label}` schema that
`validate_groups.py` expects.

Registry of known graphs is at the bottom — extend it as more share URLs are
collected. Each entry maps a Neuronpedia URL to a local artifact slug.

Usage:
    # Pull every entry in the registry
    python fetch_neuronpedia_groups.py --all

    # Pull a single ad-hoc URL into a target slug's artifact dir
    python fetch_neuronpedia_groups.py --slug vancouver-capital --url "https://www.neuronpedia.org/gemma-2-2b/graph?...&supernodes=..."

    # Dry-run (print what would be written, don't touch disk)
    python fetch_neuronpedia_groups.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PACKAGE_DIR, setup_logging

log = setup_logging()

ARTIFACTS_ROOT = PACKAGE_DIR / "artifacts"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_supernodes_from_url(url: str) -> list[list[str]]:
    """Extract the `supernodes` JSON array from a Neuronpedia share URL.

    Returns a list of [label, fid, fid, ...] entries. Empty list if the URL
    has no `supernodes` param or the param is empty.
    """
    qs = parse_qs(urlparse(url).query)
    raw = qs.get("supernodes", [None])[0]
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Could not parse supernodes JSON from URL: %s", exc)
        return []
    if not isinstance(parsed, list):
        log.error("Expected a JSON list for supernodes, got %s", type(parsed).__name__)
        return []
    return parsed


def supernodes_to_manual_groups(
    supernodes: list[list[str]],
) -> dict[str, str]:
    """Convert [[label, fid, fid, ...], ...] → {fid: label}.

    Handles label collisions across groups by suffixing duplicates.
    """
    groups: dict[str, str] = {}
    seen_labels: dict[str, int] = {}
    for entry in supernodes:
        if not entry or len(entry) < 2:
            continue
        label = str(entry[0]).strip() or "Unnamed"
        feature_ids = [str(fid).strip() for fid in entry[1:] if fid]

        # Disambiguate same-label groups (rare in practice)
        if label in seen_labels:
            seen_labels[label] += 1
            label = f"{label} ({seen_labels[label]})"
        else:
            seen_labels[label] = 1

        for fid in feature_ids:
            if fid in groups:
                log.warning(
                    "Feature %s assigned twice (%s vs %s) — keeping first.",
                    fid, groups[fid], label,
                )
                continue
            groups[fid] = label
    return groups


def slug_from_url(url: str) -> str | None:
    """Pull the `slug` query param from a Neuronpedia URL (best-effort)."""
    qs = parse_qs(urlparse(url).query)
    val = qs.get("slug", [None])[0]
    return val.strip() if val else None


# ---------------------------------------------------------------------------
# Validation against local artifacts
# ---------------------------------------------------------------------------

def check_feature_overlap(
    manual_groups: dict[str, str],
    artifact_dir: Path,
) -> tuple[int, int]:
    """Compare manual feature IDs against the local feature_descriptions file.

    Returns (n_matched, n_total) so the caller can warn if the mapping is bad.
    Looks for any feature_descriptions_*.json in the artifact dir.
    """
    desc_files = list(artifact_dir.glob("feature_descriptions_*.json"))
    if not desc_files:
        log.warning("No feature_descriptions_*.json in %s — cannot verify ID overlap.", artifact_dir)
        return 0, len(manual_groups)

    desc_file = desc_files[0]
    try:
        with open(desc_file) as f:
            features = json.load(f)
    except Exception as exc:
        log.warning("Failed to load %s: %s", desc_file, exc)
        return 0, len(manual_groups)

    local_ids = {f.get("id") for f in features if isinstance(f, dict)}
    matched = sum(1 for fid in manual_groups if fid in local_ids)
    return matched, len(manual_groups)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_manual_groups(
    manual_groups: dict[str, str],
    slug: str,
    dry_run: bool = False,
) -> Path:
    """Write {fid: label} to artifacts/<slug>/manual_groups.json."""
    artifact_dir = ARTIFACTS_ROOT / slug
    out_path = artifact_dir / "manual_groups.json"

    if dry_run:
        log.info("[dry-run] Would write %d entries to %s", len(manual_groups), out_path)
        return out_path

    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manual_groups, f, indent=2)
    log.info("Wrote %d manual group entries → %s", len(manual_groups), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Registry — known Neuronpedia share URLs paired with local slugs
# ---------------------------------------------------------------------------
#
# Each entry is (local_slug, neuronpedia_share_url).  Local slug must match
# the artifact directory name under custom_automation/artifacts/.
#
# To add more: copy a "Share" URL from Neuronpedia (the URL must already have
# supernodes saved in it — empty supernodes won't help) and append.
#
# The 3 URLs below come from demos/circuit_tracing_tutorial.ipynb. They are
# small (1 supernode each) but real human labels.

NEURONPEDIA_REGISTRY: list[tuple[str, str]] = [
    (
        "vancouver-capital",
        "https://www.neuronpedia.org/gemma-2-2b/graph"
        "?slug=gemma-fact-vancouver-victoria"
        "&clerps=%5B%5D"
        "&clickedId=4_11742_9"
        "&pruningThreshold=0.45"
        "&pinnedIds=27_18221_10%2CE_32936_9%2C21_2236_10%2C19_15123_10%2C18_1025_10%2C14_12562_9%2C14_5600_9%2C6_11873_9%2C4_8439_9%2CE_6037_4%2CE_19412_7%2C4_11742_9%2C0_16137_7"
        "&supernodes=%5B%5B%22Canada%22%2C%2214_5600_9%22%2C%226_11873_9%22%2C%224_8439_9%22%5D%5D",
    ),
    (
        "gemma-big-small-fr",
        "https://www.neuronpedia.org/gemma-2-2b/graph"
        "?slug=gemma-big-small-fr"
        "&clerps=%5B%5D"
        "&pruningThreshold=0.65"
        "&clickedId=21_9082_8"
        "&pinnedIds=27_64986_8%2CE_21996_5%2C21_9082_8%2C15_5756_5%2C6_4362_5%2C3_2873_5%2C2_4298_5"
        "&supernodes=%5B%5B%22large+%2F+huge%22%2C%2215_5756_5%22%2C%226_4362_5%22%2C%223_2873_5%22%2C%222_4298_5%22%5D%5D",
    ),
    (
        "gemma-small-min-fr",
        "https://www.neuronpedia.org/gemma-2-2b/graph"
        "?slug=gemma-small-min-fr"
        "&clerps=%5B%5D"
        "&pruningThreshold=0.8"
        "&pinnedIds=27_64986_9%2C27_69658_9%2CE_64986_6%2C8_11850_3%2CE_14904_2%2C4_13762_3%2C6_10175_3%2C3_15891_3"
        "&supernodes=%5B%5B%22synonymy%22%2C%223_15891_3%22%2C%228_11850_3%22%2C%224_13762_3%22%2C%226_10175_3%22%5D%5D"
        "&clickedId=5_13985_3",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_one(local_slug: str, url: str, dry_run: bool = False) -> dict:
    """Parse one URL into manual_groups.json. Returns a small status dict."""
    np_slug = slug_from_url(url) or "?"
    supernodes = parse_supernodes_from_url(url)
    if not supernodes:
        log.warning("[%s ← %s] No supernodes in URL — skipping.", local_slug, np_slug)
        return {
            "local_slug": local_slug,
            "neuronpedia_slug": np_slug,
            "n_groups": 0,
            "n_features": 0,
            "matched": 0,
            "wrote": False,
        }

    manual_groups = supernodes_to_manual_groups(supernodes)
    n_groups = len({v for v in manual_groups.values()})

    artifact_dir = ARTIFACTS_ROOT / local_slug
    matched, total = check_feature_overlap(manual_groups, artifact_dir)
    if total > 0 and matched == 0 and artifact_dir.exists():
        log.warning(
            "[%s ← %s] 0/%d feature IDs match local artifacts — slug may be wrong.",
            local_slug, np_slug, total,
        )
    elif total > 0:
        log.info(
            "[%s ← %s] %d/%d feature IDs match local artifacts (%d groups).",
            local_slug, np_slug, matched, total, n_groups,
        )

    write_manual_groups(manual_groups, local_slug, dry_run=dry_run)
    return {
        "local_slug": local_slug,
        "neuronpedia_slug": np_slug,
        "n_groups": n_groups,
        "n_features": total,
        "matched": matched,
        "wrote": not dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="Pull every entry in the registry.")
    parser.add_argument("--slug", help="Local slug to write manual_groups.json under (with --url).")
    parser.add_argument("--url", help="A Neuronpedia share URL containing a `supernodes` param.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written, but don't touch disk.")
    args = parser.parse_args()

    results: list[dict] = []

    if args.all:
        if not NEURONPEDIA_REGISTRY:
            log.error("Registry is empty — add entries to NEURONPEDIA_REGISTRY first.")
            sys.exit(1)
        for local_slug, url in NEURONPEDIA_REGISTRY:
            results.append(fetch_one(local_slug, url, dry_run=args.dry_run))
    elif args.slug and args.url:
        results.append(fetch_one(args.slug, args.url, dry_run=args.dry_run))
    else:
        parser.error("Pass either --all, or both --slug and --url.")

    # Summary
    print("\n=== fetch_neuronpedia_groups summary ===")
    print(f"{'local_slug':<32} {'np_slug':<32} {'groups':>6} {'feats':>6} {'matched':>8}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['local_slug'][:32]:<32} "
            f"{r['neuronpedia_slug'][:32]:<32} "
            f"{r['n_groups']:>6} "
            f"{r['n_features']:>6} "
            f"{r['matched']:>8}"
        )


if __name__ == "__main__":
    main()
