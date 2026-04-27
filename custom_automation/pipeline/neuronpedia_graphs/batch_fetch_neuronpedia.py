"""
Batch process a CSV of Neuronpedia share URLs into validation-ready artifacts.

For each row in the CSV, this runs the steps needed before validation:

  1. fetch_neuronpedia_graph.py     -> test_graphs/<slug>.json
                                     + artifacts/<slug>/manual_groups.json
  2. fetch_all_activation_text.py   -> artifacts/<slug>/pruned_activations.json
  3. generate_description.py        -> artifacts/<slug>/feature_descriptions_<desc>.json
  4. generate_supernodes.py         -> artifacts/<slug>/feature_groups_<desc>_<group>.json
                                     + artifacts/<slug>/feature_groups_<desc>_<group>_pre3.json

Each step is skipped when its output already exists, so re-runs resume cleanly.
After this finishes, slugs are ready for `run_validation_sweep.py`.

CSV format (header required):
    slug,share_url[,notes]

`slug` is the local artifact dir name (your choice). `share_url` is any of
the three Neuronpedia "Share" URL variants — Normal, iFrame, or HTML embed.
`notes` is free-form (ignored by this script).

Usage:
    OPENAI_API_KEY=sk-... python batch_fetch_neuronpedia.py
    OPENAI_API_KEY=sk-... python batch_fetch_neuronpedia.py --csv my_urls.csv
    OPENAI_API_KEY=sk-... python batch_fetch_neuronpedia.py --slugs dallas-austin
    OPENAI_API_KEY=sk-... python batch_fetch_neuronpedia.py --only-fetch     # graphs + supernodes only, no API calls
    OPENAI_API_KEY=sk-... python batch_fetch_neuronpedia.py --skip-groups   # stop after descriptions
    OPENAI_API_KEY=sk-... python batch_fetch_neuronpedia.py --force          # re-run even if outputs exist
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    DESCRIPTION_VARIANT,
    GROUPING_VARIANT,
    PACKAGE_DIR,
    REPO_ROOT,
    setup_logging,
)

log = setup_logging()

# Sibling Neuronpedia scripts live in this dir; standard pipeline steps
# (fetch_all_activation_text.py, generate_description.py, generate_supernodes.py)
# live one level up.
NEURONPEDIA_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = NEURONPEDIA_DIR.parent
ARTIFACTS_ROOT = PACKAGE_DIR / "artifacts"
TEST_GRAPHS_DIR = REPO_ROOT / "test_graphs"
DEFAULT_CSV = REPO_ROOT / "prompts" / "neuronpedia_graphs.csv"


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

_URL_RE = __import__("re").compile(r"https?://\S+")


def _clean_url(raw: str) -> str:
    """Recover a usable URL from a CSV cell, even with copy-paste quirks.

    Handles:
      - Leading/trailing whitespace
      - Surrounding single or double quotes (e.g. `"https://..."` from
        accidentally double-quoting a CSV field, or `, "https...` where a
        stray space before the quote made csv.DictReader treat it as part
        of the value)
      - `<iframe src="https://..." ...>` HTML pasted instead of the URL itself
    """
    s = raw.strip()
    # Strip surrounding matched quotes (potentially nested)
    while len(s) >= 2 and s[0] in "\"'" and s[-1] in "\"'":
        s = s[1:-1].strip()
    # If anything else is wrapping the URL, extract the first http(s) match
    m = _URL_RE.search(s)
    if m:
        candidate = m.group(0).rstrip("\"'>")
        return candidate
    return s


def read_csv(path: Path) -> list[dict]:
    """Return a list of {slug, url, notes} dicts. Skips empty rows + comments."""
    if not path.exists():
        log.error("CSV not found: %s", path)
        sys.exit(1)
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for r in reader:
            slug = (r.get("slug") or r.get("local_slug") or "").strip()
            url_raw = r.get("share_url") or r.get("url") or ""
            url = _clean_url(url_raw)
            if not slug or slug.startswith("#"):
                continue
            if not url or not url.startswith(("http://", "https://")):
                log.warning("Row for slug %s has no usable URL — skipping. (raw=%r)", slug, url_raw[:80])
                continue
            rows.append({"slug": slug, "url": url, "notes": (r.get("notes") or "").strip()})
    return rows


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_step(
    name: str,
    args: list[str],
    env: dict[str, str],
    log_lines: list[str],
) -> bool:
    """Run a pipeline step as a subprocess; return True on success."""
    log.info("  -> %s", name)
    proc = subprocess.run(
        args, env=env, capture_output=True, text=True, cwd=str(PACKAGE_DIR),
    )
    if proc.returncode != 0:
        log.error("  FAIL: %s (rc=%d)", name, proc.returncode)
        tail = proc.stderr[-1500:] if proc.stderr else "(no stderr)"
        log.error("    stderr tail:\n%s", tail)
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
    skip_groups: bool,
    force: bool,
) -> tuple[bool, list[str]]:
    """Run the pipeline steps for one slug. Returns (success, per-step log)."""
    artifact_dir = ARTIFACTS_ROOT / slug
    graph_path = TEST_GRAPHS_DIR / f"{slug}.json"
    manual_path = artifact_dir / "manual_groups.json"
    pruned_path = artifact_dir / "pruned_activations.json"
    desc_path = artifact_dir / f"feature_descriptions_{DESCRIPTION_VARIANT}.json"
    groups_path = artifact_dir / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}.json"
    pre3_path = artifact_dir / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}_pre3.json"

    log.info("\n=== %s ===", slug)
    steps: list[str] = []

    env = os.environ.copy()
    env["CURRENT_SLUG"] = slug
    env.setdefault("DESCRIPTION_VARIANT", DESCRIPTION_VARIANT)
    env.setdefault("GROUPING_VARIANT", GROUPING_VARIANT)

    # --- Step 1: graph + supernodes ----------------------------------------
    have_graph = graph_path.exists() and manual_path.exists()
    if have_graph and not force:
        log.info("  skip: graph + manual_groups already present")
        steps.append("  SKIP: fetch graph (already present)")
    else:
        ok = run_step(
            "fetch graph + supernodes",
            [
                sys.executable, str(NEURONPEDIA_DIR / "fetch_neuronpedia_graph.py"),
                "--slug", slug, "--url", url,
                *(["--force"] if force else []),
            ],
            env, steps,
        )
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

    if skip_groups:
        return True, steps

    # --- Step 4: auto supernodes (with pre-phase-3 snapshot) ---------------
    have_groups = groups_path.exists() and pre3_path.exists()
    if have_groups and not force:
        log.info("  skip: feature_groups + pre3 snapshot already present")
        steps.append("  SKIP: supernodes (already present)")
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
    parser.add_argument("--only-fetch", action="store_true",
                        help="Only fetch graph + manual_groups.json. Skip activations/descriptions/groups.")
    parser.add_argument("--skip-groups", action="store_true",
                        help="Run through descriptions but skip generate_supernodes.py.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every step even if outputs exist.")
    args = parser.parse_args()

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

    needs_api = not args.only_fetch
    if needs_api and not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set (required for descriptions/groups). "
                  "Pass --only-fetch to download graphs without API calls.")
        sys.exit(1)

    log.info("Batch: %d slug(s) from %s", len(rows), csv_path)
    log.info("Mode: only_fetch=%s skip_groups=%s force=%s", args.only_fetch, args.skip_groups, args.force)
    log.info("Variants: desc=%s grouping=%s", DESCRIPTION_VARIANT, GROUPING_VARIANT)

    success: list[tuple[str, list[str]]] = []
    failed: list[tuple[str, list[str]]] = []
    for row in rows:
        ok, steps = process_one(
            row["slug"], row["url"],
            only_fetch=args.only_fetch,
            skip_groups=args.skip_groups,
            force=args.force,
        )
        (success if ok else failed).append((row["slug"], steps))

    # Summary
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

    if not args.only_fetch and not args.skip_groups and success:
        print("\nNext step: validate.")
        print("  python custom_automation/pipeline/run_validation_sweep.py \\")
        print(f"      --slugs {','.join(s for s, _ in success)} \\")
        print("      --min-sizes 2,3,4,5,6 --standardize-names")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
