"""
Aggregate the phase-2 cap sweep into the two tables the mentor asked for.

Inputs (per slug, in custom_automation/artifacts/<slug>/):
  feature_descriptions_v2.json            — for influence scores (attribution coverage)
  manual_groups.json                      — human condition
  feature_groups_v2_a2.json               — ours-full (unlimited)
  feature_groups_v2_a2_cap{N}.json        — cap variants
  validation_history_v2_a2_neuropedia.json — M1/M2 macros for every min_group_size
                                             that has been validated (Table B).

Outputs (in custom_automation/analysis/phase2_cap_results/), one .md + .csv per min_group_size:
  phase2_cap_sweep_summary_min{N}.md
  phase2_cap_sweep_summary_min{N}.csv

Table A — structural metrics, averaged over slugs (min_group_size-independent
          since these are properties of the grouping itself):
  | condition | avg features grouped | avg #groups | avg group size | avg coverage |

Table B — autointerp metrics for ours-full and the cap-closest-to-human:
  | condition | M1 | M2 |

"Closest to human" picks the cap whose corpus-mean group size is nearest the
corpus-mean human group size.

Run:
    # All min sizes that the validation history covers:
    python custom_automation/analysis/summarize_phase2_cap_sweep.py

    # Or explicitly:
    python custom_automation/analysis/summarize_phase2_cap_sweep.py --min-sizes 2,3,4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
RESULTS_DIR = Path(__file__).resolve().parent / "phase2_cap_results"
DESCRIPTION_VARIANT = "v2"
GROUPING_VARIANT = "a2"
DIFFICULTY = "medium"

HISTORY_NAME = f"validation_history_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}_neuropedia.json"
DESC_NAME = f"feature_descriptions_{DESCRIPTION_VARIANT}.json"
OURS_FULL_NAME = f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}.json"
MANUAL_NAME = "manual_groups.json"
CAP_RE = re.compile(rf"^feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}_cap(\d+)\.json$")


# ---------------------------------------------------------------------------
# Structural-metric collection (Table A — min_group_size-independent)
# ---------------------------------------------------------------------------

def is_content_group(name: str) -> bool:
    return name != "Ungrouped" and not name.startswith(('Emb: "', 'Output: "'))


def structural_stats(groups: dict[str, str], id_to_score: dict[str, float]) -> dict:
    by_group: dict[str, list[str]] = {}
    for fid, gname in groups.items():
        if not is_content_group(gname):
            continue
        by_group.setdefault(gname, []).append(fid)

    features_grouped = sum(len(v) for v in by_group.values())
    n_groups = len(by_group)
    avg_size = (features_grouped / n_groups) if n_groups else 0.0

    total_inf = sum(id_to_score.values())
    if total_inf > 0:
        annotated_inf = sum(
            id_to_score.get(fid, 0.0)
            for gname, fids in by_group.items()
            for fid in fids
        )
        coverage = annotated_inf / total_inf
    else:
        coverage = 0.0

    return {
        "features_grouped": features_grouped,
        "n_groups": n_groups,
        "avg_group_size": avg_size,
        "coverage": coverage,
    }


def discover_slugs() -> list[str]:
    slugs: list[str] = []
    for child in sorted(ARTIFACTS_DIR.iterdir()):
        if child.is_dir() and (child / OURS_FULL_NAME).exists():
            slugs.append(child.name)
    return slugs


def discover_caps_for_slug(slug: str) -> list[int]:
    caps: list[int] = []
    for f in (ARTIFACTS_DIR / slug).glob(f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}_cap*.json"):
        if "_pre3" in f.name:
            continue
        m = CAP_RE.match(f.name)
        if m:
            caps.append(int(m.group(1)))
    return sorted(caps)


def load_id_to_score(slug: str) -> dict[str, float]:
    desc_file = ARTIFACTS_DIR / slug / DESC_NAME
    if not desc_file.exists():
        return {}
    with open(desc_file, encoding="utf-8") as f:
        feats = json.load(f)
    return {
        f["id"]: float(f.get("influence_score", 0.0))
        for f in feats if "id" in f
    }


def load_groups(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect_structural_per_slug(slug: str) -> dict:
    slug_dir = ARTIFACTS_DIR / slug
    id_to_score = load_id_to_score(slug)
    out: dict[str, dict] = {}

    human = load_groups(slug_dir / MANUAL_NAME)
    if human is not None:
        out["human"] = structural_stats(human, id_to_score)

    full = load_groups(slug_dir / OURS_FULL_NAME)
    if full is not None:
        out["ours-full"] = structural_stats(full, id_to_score)

    for cap in discover_caps_for_slug(slug):
        cap_groups = load_groups(slug_dir / f"feature_groups_{DESCRIPTION_VARIANT}_{GROUPING_VARIANT}_cap{cap}.json")
        if cap_groups is not None:
            out[f"ours-cap{cap}"] = structural_stats(cap_groups, id_to_score)

    return out


def aggregate_structural(per_slug_structural: dict[str, dict]) -> dict[str, dict[str, float]]:
    by_cond: dict[str, dict[str, list[float]]] = {}
    for slug_data in per_slug_structural.values():
        for cond, stats in slug_data.items():
            d = by_cond.setdefault(cond, {"features_grouped": [], "n_groups": [], "avg_group_size": [], "coverage": []})
            for k, v in stats.items():
                d[k].append(v)

    averaged: dict[str, dict[str, float]] = {}
    for cond, lists in by_cond.items():
        averaged[cond] = {
            k: (sum(vs) / len(vs) if vs else 0.0)
            for k, vs in lists.items()
        }
        averaged[cond]["n_slugs"] = len(lists["features_grouped"])
    return averaged


# ---------------------------------------------------------------------------
# Autointerp collection (Table B — read from validation history per min_size)
# ---------------------------------------------------------------------------

def macro_from_report(report: dict, condition: str, method: str) -> tuple[float | None, float | None]:
    cond = report.get("conditions", {}).get(condition, {})
    macro = cond.get(DIFFICULTY, {}).get(method, {}).get("macro_avg", {})
    if not macro:
        return None, None
    n_valid = cond.get("n_groups_valid", 0)
    if n_valid == 0:
        return None, None
    return macro.get("mean_accuracy"), macro.get("stderr_accuracy")


def collect_autointerp_history() -> dict[int, dict[str, dict[str, dict[str, tuple[float | None, float | None]]]]]:
    """Returns {min_size: {slug: {condition: {m1: (mean, se), m2: (mean, se)}}}}."""
    out: dict = {}
    for path in sorted(ARTIFACTS_DIR.glob(f"*/{HISTORY_NAME}")):
        slug = path.parent.name
        try:
            with open(path, encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for entry in history:
            size = entry.get("min_group_size")
            if size is None:
                continue
            slug_bucket = out.setdefault(size, {}).setdefault(slug, {})
            for cond in entry.get("conditions", {}):
                slug_bucket[cond] = {
                    "m1": macro_from_report(entry, cond, "method1"),
                    "m2": macro_from_report(entry, cond, "method2"),
                }
    return out


def discover_min_sizes() -> list[int]:
    sizes: set[int] = set()
    for path in ARTIFACTS_DIR.glob(f"*/{HISTORY_NAME}"):
        try:
            with open(path, encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for entry in history:
            s = entry.get("min_group_size")
            if isinstance(s, int):
                sizes.add(s)
    return sorted(sizes)


def average_autointerp(by_slug: dict, condition: str, method: str) -> tuple[float | None, int]:
    means = [
        d[condition][method][0]
        for d in by_slug.values()
        if condition in d and d[condition][method][0] is not None
    ]
    if not means:
        return None, 0
    return sum(means) / len(means), len(means)


def pick_closest_cap(structural_avg: dict[str, dict[str, float]]) -> str | None:
    if "human" not in structural_avg:
        return None
    target = structural_avg["human"]["avg_group_size"]
    cap_conds = [c for c in structural_avg if c.startswith("ours-cap")]
    if not cap_conds:
        return None
    return min(cap_conds, key=lambda c: abs(structural_avg[c]["avg_group_size"] - target))


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

CONDITION_DISPLAY_ORDER = ["human", "ours-cap50", "ours-cap100", "ours-cap150", "ours-cap200", "ours-full"]


def sort_conditions(conds: list[str]) -> list[str]:
    rank = {c: i for i, c in enumerate(CONDITION_DISPLAY_ORDER)}
    cap_re = re.compile(r"ours-cap(\d+)")

    def key(c: str):
        if c in rank:
            return (rank[c], 0)
        m = cap_re.match(c)
        if m:
            return (50 + int(m.group(1)) // 50, 0)
        return (1000, c)

    return sorted(conds, key=key)


def fmt_pct(value: float | None, stderr: float | None = None) -> str:
    if value is None:
        return "—"
    if stderr is None:
        return f"{value * 100:.1f}%"
    return f"{value * 100:.1f}% ± {stderr * 100:.1f}%"


def fmt_num(value: float | None, places: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"


def write_markdown(
    out_path: Path,
    min_size: int,
    structural_avg: dict,
    structural_per_slug: dict,
    autointerp_for_size: dict,
    closest_cap: str | None,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = []
    lines.append(f"# Phase-2 Cap Sweep Summary — min_group_size={min_size}")
    lines.append("")
    lines.append(f"**Generated:** {now}  ")
    lines.append(f"**Variant:** desc=`{DESCRIPTION_VARIANT}`, grouping=`{GROUPING_VARIANT}`  ")
    lines.append(f"**Difficulty:** {DIFFICULTY}  |  **min_group_size:** {min_size}")
    lines.append("")

    # Table A — structural (size-independent)
    lines.append("## Table A — Structural metrics (averaged across slugs)")
    lines.append("")
    lines.append("Each row averages the per-slug metric over slugs that have that condition. "
                 "These values are **min_group_size-independent** (properties of the grouping itself); "
                 "they're identical across all min-size summary files.")
    lines.append("")
    lines.append("| condition | avg features grouped | avg # groups | avg group size | avg coverage | n_slugs |")
    lines.append("|:--|---:|---:|---:|---:|---:|")
    for cond in sort_conditions(list(structural_avg.keys())):
        s = structural_avg[cond]
        lines.append(
            f"| **{cond}** "
            f"| {fmt_num(s['features_grouped'])} "
            f"| {fmt_num(s['n_groups'])} "
            f"| {fmt_num(s['avg_group_size'], 2)} "
            f"| {fmt_pct(s['coverage'])} "
            f"| {int(s['n_slugs'])} |"
        )
    lines.append("")

    # Table B — autointerp (size-specific)
    lines.append("## Table B — Autointerp metrics (M1/M2)")
    lines.append("")
    if closest_cap is None:
        lines.append("_No `ours-cap<N>` conditions or no `human` condition found — "
                     "cannot pick closest cap. Showing M1/M2 for `ours-full` only._")
        focus = ["ours-full"]
    else:
        target_size = structural_avg["human"]["avg_group_size"]
        cap_size = structural_avg[closest_cap]["avg_group_size"]
        lines.append(f"`ours-full` (unlimited features) and `{closest_cap}` "
                     f"(closest to human's avg group size: target {target_size:.2f}, "
                     f"picked {cap_size:.2f}).")
        focus = ["ours-full", closest_cap]
    lines.append("")
    lines.append("| condition | M1 (mean across slugs) | M2 (mean across slugs) | n_slugs |")
    lines.append("|:--|---:|---:|---:|")
    for cond in focus:
        m1_mean, m1_n = average_autointerp(autointerp_for_size, cond, "m1")
        m2_mean, m2_n = average_autointerp(autointerp_for_size, cond, "m2")
        n = max(m1_n, m2_n)
        lines.append(
            f"| **{cond}** "
            f"| {fmt_pct(m1_mean)} "
            f"| {fmt_pct(m2_mean)} "
            f"| {n} |"
        )
    lines.append("")

    # Per-slug structural detail
    lines.append("## Per-slug structural detail")
    lines.append("")
    lines.append("| slug | condition | features grouped | # groups | avg group size | coverage |")
    lines.append("|:--|:--|---:|---:|---:|---:|")
    for slug in sorted(structural_per_slug.keys()):
        for cond in sort_conditions(list(structural_per_slug[slug].keys())):
            s = structural_per_slug[slug][cond]
            lines.append(
                f"| `{slug}` | {cond} "
                f"| {s['features_grouped']} "
                f"| {s['n_groups']} "
                f"| {fmt_num(s['avg_group_size'], 2)} "
                f"| {fmt_pct(s['coverage'])} |"
            )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def write_csv(
    out_path: Path,
    min_size: int,
    structural_per_slug: dict,
    autointerp_for_size: dict,
) -> None:
    fieldnames = [
        "slug", "min_group_size", "condition",
        "features_grouped", "n_groups", "avg_group_size", "coverage",
        "m1_mean", "m1_stderr", "m2_mean", "m2_stderr",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for slug in sorted(structural_per_slug.keys()):
            ai = autointerp_for_size.get(slug, {})
            for cond, s in structural_per_slug[slug].items():
                m1 = ai.get(cond, {}).get("m1", (None, None))
                m2 = ai.get(cond, {}).get("m2", (None, None))
                w.writerow({
                    "slug": slug,
                    "min_group_size": min_size,
                    "condition": cond,
                    "features_grouped": s["features_grouped"],
                    "n_groups": s["n_groups"],
                    "avg_group_size": round(s["avg_group_size"], 4),
                    "coverage": round(s["coverage"], 4),
                    "m1_mean":   None if m1[0] is None else round(m1[0], 4),
                    "m1_stderr": None if m1[1] is None else round(m1[1], 4),
                    "m2_mean":   None if m2[0] is None else round(m2[0], 4),
                    "m2_stderr": None if m2[1] is None else round(m2[1], 4),
                })
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-sizes", default=None,
                        help="Comma-separated min_group_size values to summarise. "
                             "Default: every size present in the validation history files.")
    args = parser.parse_args()

    slugs = discover_slugs()
    if not slugs:
        print(f"No slugs with {OURS_FULL_NAME} found under {ARTIFACTS_DIR}")
        return

    # Structural: collect once (size-independent).
    structural_per_slug: dict[str, dict] = {s: collect_structural_per_slug(s) for s in slugs}
    structural_avg = aggregate_structural(structural_per_slug)
    closest_cap = pick_closest_cap(structural_avg)

    # Autointerp: bucketed by min_size from history files.
    autointerp_by_size = collect_autointerp_history()
    available_sizes = sorted(autointerp_by_size.keys())

    if args.min_sizes:
        requested = parse_int_list(args.min_sizes)
        missing = [s for s in requested if s not in autointerp_by_size]
        if missing:
            print(f"Skipping requested sizes with no validation history: {missing}. "
                  f"Available: {available_sizes}")
        wanted = [s for s in requested if s in autointerp_by_size]
        if not wanted:
            print(f"None of the requested min sizes have validation history. "
                  f"Available sizes: {available_sizes}")
            return
    else:
        wanted = available_sizes

    if not wanted:
        print("No validation history files found — run run_validation_sweep.py first, "
              "then re-run this script.")
        return

    print(f"Summarising for min_group_size values: {wanted}")
    if closest_cap:
        target = structural_avg["human"]["avg_group_size"]
        picked = structural_avg[closest_cap]["avg_group_size"]
        print(f"Closest-to-human cap (size-independent): {closest_cap} "
              f"(avg size {picked:.2f} vs human {target:.2f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for size in wanted:
        ai_for_size = autointerp_by_size.get(size, {})
        out_md  = RESULTS_DIR / f"phase2_cap_sweep_summary_min{size}.md"
        out_csv = RESULTS_DIR / f"phase2_cap_sweep_summary_min{size}.csv"
        write_markdown(out_md, size, structural_avg, structural_per_slug, ai_for_size, closest_cap)
        write_csv(out_csv, size, structural_per_slug, ai_for_size)


if __name__ == "__main__":
    main()
