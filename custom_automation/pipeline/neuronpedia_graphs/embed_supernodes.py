"""
Embed feature group assignments and descriptions into a NEW graph JSON file
per slug × variant, so the original `test_graphs/<slug>.json` (with human
supernodes from Neuronpedia) is preserved alongside the LLM annotations.

For each (slug, variant) it produces `test_graphs/<slug>_ours[<variant>].json`
that contains, relative to the original graph:
  1. Feature descriptions injected into node `clerp` fields
  2. `qParams.supernodes` rebuilt from feature_groups (compact JSON string)
  3. `qParams.pinnedIds` set to all grouped node IDs (comma-separated)
  4. `qParams.pruningThreshold` set to PRUNING_THRESHOLD

Variants the script can read from the artifacts directory:
  full        -> feature_groups_v2_a2.json              ->  <slug>_ours.json
  cap50       -> feature_groups_v2_a2_cap50.json        ->  <slug>_ours_cap50.json
  cap100      -> feature_groups_v2_a2_cap100.json       ->  <slug>_ours_cap100.json
  cap150      -> feature_groups_v2_a2_cap150.json       ->  <slug>_ours_cap150.json
  cap200      -> feature_groups_v2_a2_cap200.json       ->  <slug>_ours_cap200.json
  no-reconcile-> feature_groups_v2_a2_pre3.json         ->  <slug>_ours_no_reconcile.json

Usage:
    # Default: write <slug>_ours.json for every slug (canonical "ours full")
    python custom_automation/pipeline/neuronpedia_graphs/embed_supernodes.py

    # Specific slugs only
    python custom_automation/pipeline/neuronpedia_graphs/embed_supernodes.py --slugs gemma-G,gemma-addition

    # Specific variants (comma-separated). "all" expands to every variant present.
    python custom_automation/pipeline/neuronpedia_graphs/embed_supernodes.py --variants full,cap100
    python custom_automation/pipeline/neuronpedia_graphs/embed_supernodes.py --variants all

    # Clear LLM-injected annotations (deletes the suffixed files)
    python custom_automation/pipeline/neuronpedia_graphs/embed_supernodes.py --clear

    # Legacy in-place mode (overwrites <slug>.json itself; requires explicit opt-in)
    python custom_automation/pipeline/neuronpedia_graphs/embed_supernodes.py --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DESCRIPTION_VARIANT, GROUPING_VARIANT, PACKAGE_DIR, setup_logging

log = setup_logging()

ARTIFACTS_ROOT = PACKAGE_DIR / "artifacts"
TEST_GRAPHS_DIR = PACKAGE_DIR.parent / "test_graphs"

# Threshold to write into qParams. Matches what we actually pruned at when
# fetching activations / generating descriptions, so the viewer shows the
# same feature set we grouped.
PRUNING_THRESHOLD = "0.7"

VARIANT_SPECS: dict[str, tuple[str, str, str]] = {
    # variant key  -> (groups filename suffix, output filename suffix, display tag)
    "full":          ("",          "_ours",                "ours full"),
    "no-reconcile":  ("_pre3",     "_ours_no_reconcile",   "ours no-reconcile"),
    "cap50":         ("_cap50",    "_ours_cap50",          "ours cap50"),
    "cap100":        ("_cap100",   "_ours_cap100",         "ours cap100"),
    "cap150":        ("_cap150",   "_ours_cap150",         "ours cap150"),
    "cap200":        ("_cap200",   "_ours_cap200",         "ours cap200"),
}


def groups_path_for(slug: str, variant: str) -> Path:
    suffix, _, _ = VARIANT_SPECS[variant]
    return ARTIFACTS_ROOT / slug / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}{suffix}.json"


def output_path_for(slug: str, variant: str, in_place: bool) -> Path:
    if in_place:
        return TEST_GRAPHS_DIR / f"{slug}.json"
    _, out_suffix, _ = VARIANT_SPECS[variant]
    return TEST_GRAPHS_DIR / f"{slug}{out_suffix}.json"


def embed_one(slug: str, variant: str, *, in_place: bool) -> bool:
    src_path = TEST_GRAPHS_DIR / f"{slug}.json"
    if not src_path.exists():
        log.warning("[%s/%s] source graph not found: %s", slug, variant, src_path)
        return False

    groups_path = groups_path_for(slug, variant)
    if not groups_path.exists():
        log.info("[%s/%s] no groups file (%s) — skipping", slug, variant, groups_path.name)
        return False

    desc_path = ARTIFACTS_ROOT / slug / f"feature_descriptions_{DESCRIPTION_VARIANT}.json"

    with open(src_path, encoding="utf-8") as f:
        graph = json.load(f)

    with open(groups_path, encoding="utf-8") as f:
        groups: dict[str, str] = json.load(f)

    descriptions: dict[str, str] = {}
    if desc_path.exists():
        with open(desc_path, encoding="utf-8") as f:
            descriptions = {item["id"]: item["generated_description"] for item in json.load(f)}

    nodes = graph.get("nodes", [])

    # 1. Inject descriptions
    desc_count = 0
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        if node_id in descriptions:
            existing = node.get("clerp", "")
            if not existing or node.get("feature_type") == "cross layer transcoder":
                node["clerp"] = descriptions[node_id]
                desc_count += 1

    # 2. Build supernodes (skip Ungrouped + Emb/Output pseudo-groups)
    group_to_nodes: dict[str, list[str]] = {}
    for node in nodes:
        node_id = str(node.get("node_id", ""))
        gname = groups.get(node_id)
        if not gname or gname == "Ungrouped":
            continue
        clean = gname.replace("/", "-").replace("\\", "-")
        group_to_nodes.setdefault(clean, []).append(node_id)

    supernodes_array: list[list[str]] = []
    pinned_ids: list[str] = []
    for gname, members in sorted(group_to_nodes.items()):
        supernodes_array.append([gname] + members)
        pinned_ids.extend(members)

    # 3. Write into qParams (preserves whatever else was there — clickedId, slug, scan, etc.)
    q_params = graph.get("qParams", {}) or {}
    q_params["supernodes"] = json.dumps(supernodes_array, separators=(",", ":"))
    q_params["pinnedIds"] = ",".join(pinned_ids)
    q_params["pruningThreshold"] = PRUNING_THRESHOLD
    q_params.pop("clerps", None)
    graph["qParams"] = q_params

    # Stamp the slug field so it matches the new filename (the viewer key),
    # and tag the prompt so the dropdown can distinguish variants.
    out_path = output_path_for(slug, variant, in_place=in_place)
    if not in_place:
        new_slug = out_path.stem
        meta = graph.setdefault("metadata", {})
        meta["slug"] = new_slug
        _, _, display_tag = VARIANT_SPECS[variant]
        orig_prompt = meta.get("prompt", "")
        meta["prompt"] = f"{orig_prompt} [{display_tag}]" if orig_prompt else f"[{display_tag}]"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    log.info(
        "[%s/%s] %d groups, %d pinned, %d clerps -> %s",
        slug, variant, len(supernodes_array), len(pinned_ids), desc_count, out_path.name,
    )
    return True


def discover_variants_for(slug: str) -> list[str]:
    """Return every variant key whose groups file exists for this slug."""
    return [v for v in VARIANT_SPECS if groups_path_for(slug, v).exists()]


def discover_slugs() -> list[str]:
    """Slugs that have at least the canonical (full) groups file AND a graph JSON."""
    slugs: list[str] = []
    canonical = f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}.json"
    for child in sorted(ARTIFACTS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if not (child / canonical).exists():
            continue
        if not (TEST_GRAPHS_DIR / f"{child.name}.json").exists():
            continue
        slugs.append(child.name)
    return slugs


def parse_variants(raw: str | None, slug: str) -> list[str]:
    """Resolve --variants spec to concrete variant keys for this slug."""
    if not raw:
        return ["full"]
    requested = [v.strip() for v in raw.split(",") if v.strip()]
    if "all" in requested:
        return discover_variants_for(slug)
    invalid = [v for v in requested if v not in VARIANT_SPECS]
    if invalid:
        log.error("Unknown variant(s): %s. Valid: %s", invalid, list(VARIANT_SPECS))
        sys.exit(1)
    return requested


def clear_for(slug: str, variants: list[str]) -> int:
    removed = 0
    for variant in variants:
        out = output_path_for(slug, variant, in_place=False)
        if out.exists():
            out.unlink()
            log.info("[%s/%s] removed %s", slug, variant, out.name)
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slugs", help="Comma-separated slugs. Default: auto-detect every slug with both a graph file and ours-full groups.")
    parser.add_argument("--variants", default="full",
                        help=f"Comma-separated variants from {{full, no-reconcile, cap50, cap100, cap150, cap200}} or 'all'. Default: full.")
    parser.add_argument("--clear", action="store_true",
                        help="Delete the suffixed output files for the requested slugs/variants instead of writing.")
    parser.add_argument("--in-place", action="store_true",
                        help="Legacy mode: overwrite test_graphs/<slug>.json itself instead of writing a separate file. "
                             "Only valid with --variants full. Use only if you intentionally want to replace the original.")
    args = parser.parse_args()

    if args.in_place and args.variants != "full":
        log.error("--in-place only supports --variants full (it would otherwise clobber the same file repeatedly).")
        sys.exit(1)

    slugs = (
        [s.strip() for s in args.slugs.split(",") if s.strip()]
        if args.slugs else discover_slugs()
    )
    if not slugs:
        log.error("No slugs found.")
        sys.exit(1)

    if args.clear:
        log.info("Clearing %d slug(s)…", len(slugs))
        total = 0
        for slug in slugs:
            variants = parse_variants(args.variants, slug)
            total += clear_for(slug, variants)
        log.info("Removed %d file(s).", total)
        return

    log.info("Embedding into %d slug(s) (in_place=%s)…", len(slugs), args.in_place)
    successes = 0
    attempts = 0
    for slug in slugs:
        variants = parse_variants(args.variants, slug)
        for variant in variants:
            attempts += 1
            if embed_one(slug, variant, in_place=args.in_place):
                successes += 1
    log.info("Done: %d/%d (slug × variant) wrote successfully.", successes, attempts)


if __name__ == "__main__":
    main()