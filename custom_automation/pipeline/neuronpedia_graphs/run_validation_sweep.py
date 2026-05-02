"""
Sweep validation across slugs × min_group_size.

For each (slug, min_group_size) combination, invoke validate_neuropedia_groups.py
as a subprocess (so each run gets a fresh config import) with the appropriate
env vars. After each run, parse the per-slug validation report and append rows
to a single aggregated CSV: one row per (slug, min_group_size, condition).

The four canonical conditions (random, human, ours-full, ours-no-reconciliation)
are all evaluated in each subprocess run, since they share data loading.

Slugs default to every artifact directory that has both
`feature_groups_<desc>_<group>.json` (so ours-full is available) and
`manual_groups.json` (so human is available). Pass --slugs to override.

Usage:
    python run_validation_sweep.py
    python run_validation_sweep.py --slugs vancouver-capital,gemma-big-small-fr
    python run_validation_sweep.py --min-sizes 2,3,4
    python run_validation_sweep.py --out my_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    DESCRIPTION_VARIANT,
    GROUPING_VARIANT,
    PACKAGE_DIR,
    setup_logging,
)

log = setup_logging()

ARTIFACTS_ROOT = PACKAGE_DIR / "artifacts"
DEFAULT_OUT = PACKAGE_DIR / "analysis" / "results" / "validation_sweep.csv"
PIPELINE_SCRIPT = Path(__file__).resolve().parent / "validate_neuropedia_groups.py"

DEFAULT_MIN_SIZES = (2, 3, 4, 5, 6)
# Base conditions; per-slug `ours-cap<N>` variants are added dynamically when
# the cap-sweep driver has produced them.
BASE_CONDITIONS = ("random", "human", "ours-full", "ours-no-reconciliation")


# ---------------------------------------------------------------------------
# Slug discovery
# ---------------------------------------------------------------------------

def discover_slugs(require_manual: bool = True) -> list[str]:
    if not ARTIFACTS_ROOT.exists():
        return []
    slugs: list[str] = []
    feature_groups_filename = f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}.json"
    for child in sorted(ARTIFACTS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if not (child / feature_groups_filename).exists():
            continue
        if require_manual and not (child / "manual_groups.json").exists():
            continue
        slugs.append(child.name)
    return slugs


# ---------------------------------------------------------------------------
# Run one (slug, min_group_size) combination
# ---------------------------------------------------------------------------

def run_one(
    slug: str,
    min_group_size: int,
    conditions: list[str],
) -> dict | None:
    artifact_dir = ARTIFACTS_ROOT / slug
    report_path = artifact_dir / f"validation_report_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}_neuropedia.json"

    env = os.environ.copy()
    env["CURRENT_SLUG"] = slug
    env["VALIDATION_MIN_GROUP_SIZE"] = str(min_group_size)
    env["VALIDATION_CONDITIONS"] = ",".join(conditions)
    env["VALIDATION_DIFFICULTY"] = "medium"
    env.setdefault("DESCRIPTION_VARIANT", DESCRIPTION_VARIANT)
    env.setdefault("GROUPING_VARIANT", GROUPING_VARIANT)

    log.info("→ %s  min_group_size=%d", slug, min_group_size)
    proc = subprocess.run(
        [sys.executable, str(PIPELINE_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PACKAGE_DIR),
    )
    if proc.returncode != 0:
        log.error("validate_groups failed for %s @ size=%d (rc=%d)", slug, min_group_size, proc.returncode)
        log.error("stderr tail:\n%s", proc.stderr[-1500:])
        return None
    if not report_path.exists():
        log.error("Report missing after run: %s", report_path)
        return None
    try:
        with open(report_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.error("Failed to parse report %s: %s", report_path, exc)
        return None


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------

def report_to_rows(slug: str, min_group_size: int, report: dict) -> list[dict]:
    rows: list[dict] = []
    conditions = report.get("conditions", {})
    difficulty = report.get("difficulty", "medium")

    # Iterate over every condition present (BASE_CONDITIONS + any ours-cap<N>
    # the validator discovered for this slug).
    for cond in conditions.keys():
        c = conditions.get(cond)
        if c is None:
            continue
        if c.get("skipped"):
            rows.append({
                "slug": slug,
                "prompt": report.get("prompt", ""),
                "min_group_size": min_group_size,
                "difficulty": difficulty,
                "condition": cond,
                "skipped": True,
                "skip_reason": c.get("reason", ""),
            })
            continue
        src_data = c.get("medium", {}) if difficulty != "both" else c.get("medium", c.get("easy", {}))
        m1 = src_data.get("method1", {}).get("macro_avg", {})
        m2 = src_data.get("method2", {}).get("macro_avg", {})
        rows.append({
            "slug": slug,
            "prompt": report.get("prompt", ""),
            "min_group_size": min_group_size,
            "difficulty": difficulty,
            "condition": cond,
            "skipped": False,
            "skip_reason": "",
            "m1_mean": m1.get("mean_accuracy"),
            "m1_stderr": m1.get("stderr_accuracy"),
            "m2_mean": m2.get("mean_accuracy"),
            "m2_stderr": m2.get("stderr_accuracy"),
            "attribution_coverage": c.get("attribution_coverage"),
            "n_groups_total": c.get("n_groups_total"),
            "n_groups_valid": c.get("n_groups_valid"),
            "m1_total_trials": src_data.get("method1", {}).get("total_trials"),
            "m2_total_tasks": src_data.get("method2", {}).get("total_tasks"),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slugs", help="Comma-separated slugs. Default: auto-detect those with manual_groups.json.")
    parser.add_argument("--include-no-manual", action="store_true",
                        help="Auto-detect slugs even without manual_groups.json (random + ours conditions only).")
    parser.add_argument("--min-sizes", default=",".join(str(n) for n in DEFAULT_MIN_SIZES),
                        help="Comma-separated min_group_size values to sweep. Default: 2,3,4,5,6.")
    parser.add_argument("--conditions", default="",
                        help="Comma-separated conditions to evaluate per run. "
                             "Default empty = let the validator auto-detect "
                             "(BASE_CONDITIONS + every ours-cap<N> file present).")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output CSV path (default: {DEFAULT_OUT}).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip (slug, min_group_size) combos already present in the CSV.")
    args = parser.parse_args()

    if args.slugs:
        slugs = parse_str_list(args.slugs)
    else:
        slugs = discover_slugs(require_manual=not args.include_no_manual)
        if not slugs:
            log.error(
                "No slugs found. Run batch_fetch_neuronpedia.py to populate artifacts, "
                "or pass --slugs / --include-no-manual."
            )
            sys.exit(1)

    min_sizes = parse_int_list(args.min_sizes)
    conditions = parse_str_list(args.conditions)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Sweep: %d slug(s) × %d min_group_size value(s) = %d runs",
             len(slugs), len(min_sizes), len(slugs) * len(min_sizes))
    log.info("Slugs: %s", slugs)
    log.info("Min sizes: %s", min_sizes)
    log.info("Conditions: %s", conditions)
    log.info("Output: %s", out_path)

    fieldnames = [
        "slug", "prompt", "min_group_size", "difficulty",
        "condition", "skipped", "skip_reason",
        "m1_mean", "m1_stderr", "m2_mean", "m2_stderr",
        "attribution_coverage", "n_groups_total", "n_groups_valid",
        "m1_total_trials", "m2_total_tasks",
        "ran_at",
    ]

    # Load existing (slug, size) combos to support --skip-existing
    done: set[tuple[str, int]] = set()
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    done.add((row["slug"], int(row["min_group_size"])))
                except (ValueError, KeyError):
                    pass
        log.info("--skip-existing: %d (slug, size) combos already in CSV", len(done))

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for slug in slugs:
            for size in min_sizes:
                if args.skip_existing and (slug, size) in done:
                    log.info("→ skip %s @ size=%d (already done)", slug, size)
                    continue
                report = run_one(slug, size, conditions)
                if report is None:
                    log.warning("Skipping rows for %s @ size=%d (no report)", slug, size)
                    continue
                rows = report_to_rows(slug, size, report)
                ran_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                for row in rows:
                    row["ran_at"] = ran_at
                    writer.writerow(row)
                f.flush()
                log.info("  + wrote %d row(s) for %s @ size=%d", len(rows), slug, size)

    log.info("Sweep complete. Aggregated CSV: %s", out_path)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
