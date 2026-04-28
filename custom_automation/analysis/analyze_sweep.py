"""
Analyze validation_sweep.csv and print a formatted summary to terminal.

Usage:
    python analyze_sweep.py
    python analyze_sweep.py --csv path/to/sweep.csv
    python analyze_sweep.py --size 3
    python analyze_sweep.py --per-slug
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV = RESULTS_DIR / "validation_sweep.csv"

CONDITIONS = ["random", "human", "ours-no-reconciliation", "ours-full"]
CONDITION_LABELS = {
    "random": "Random",
    "human": "Human",
    "ours-no-reconciliation": "Ours (no recon.)",
    "ours-full": "Ours (full)",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_int(v: str) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict], size: int) -> dict[str, dict]:
    """Return per-condition stats averaged across slugs."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if parse_int(r["min_group_size"]) != size:
            continue
        if r["skipped"].lower() == "true":
            continue
        m1 = parse_float(r["m1_mean"])
        m2 = parse_float(r["m2_mean"])
        trials = parse_int(r["m1_total_trials"])
        if m1 is None or trials is None or trials == 0:
            continue
        by_cond[r["condition"]].append(r)

    result: dict[str, dict] = {}
    for cond, crows in by_cond.items():
        m1_vals = [parse_float(r["m1_mean"]) for r in crows]
        m2_vals = [parse_float(r["m2_mean"]) for r in crows if parse_float(r["m2_mean"]) is not None]
        m1_mean = sum(m1_vals) / len(m1_vals) if m1_vals else None
        m2_mean = sum(m2_vals) / len(m2_vals) if m2_vals else None

        # SE of the mean across slugs
        def se(vals: list[float]) -> float | None:
            if len(vals) < 2:
                return None
            n = len(vals)
            mu = sum(vals) / n
            var = sum((x - mu) ** 2 for x in vals) / (n - 1)
            return (var / n) ** 0.5

        slugs = sorted({r["slug"] for r in crows})
        total_trials = sum(parse_int(r["m1_total_trials"]) or 0 for r in crows)
        result[cond] = {
            "n_slugs": len(crows),
            "slugs": slugs,
            "m1_mean": m1_mean,
            "m1_se": se(m1_vals),
            "m2_mean": m2_mean,
            "m2_se": se(m2_vals) if m2_vals else None,
            "total_trials": total_trials,
            "n_groups_valid_total": sum(parse_int(r["n_groups_valid"]) or 0 for r in crows),
            "attribution_coverage": (
                sum(parse_float(r["attribution_coverage"]) or 0 for r in crows) / len(crows)
                if crows else None
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Per-slug detail
# ---------------------------------------------------------------------------

def per_slug_table(rows: list[dict], size: int) -> None:
    filtered = [
        r for r in rows
        if parse_int(r["min_group_size"]) == size
        and r["skipped"].lower() != "true"
        and (parse_int(r["m1_total_trials"]) or 0) > 0
    ]

    by_slug: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in filtered:
        by_slug[r["slug"]][r["condition"]] = r

    slugs = sorted(by_slug)

    col_w = 18
    hdr = f"{'Slug':<30}" + "".join(f"{CONDITION_LABELS.get(c, c):>{col_w}}" for c in CONDITIONS)
    sep = "-" * len(hdr)
    print(f"\nPer-slug M1 accuracy  (min_group_size={size})")
    print(sep)
    print(hdr + "  (M1)")
    print(sep)
    for slug in slugs:
        row_str = f"{slug:<30}"
        for cond in CONDITIONS:
            r = by_slug[slug].get(cond)
            if r is None:
                row_str += f"{'—':>{col_w}}"
            else:
                m1 = parse_float(r["m1_mean"])
                se_v = parse_float(r["m1_stderr"])
                if m1 is None:
                    row_str += f"{'—':>{col_w}}"
                elif se_v is not None:
                    row_str += f"{m1*100:>6.1f}±{se_v*100:<4.1f}{'':>{col_w - 14}}"
                else:
                    row_str += f"{m1*100:>{col_w}.1f}"
        print(row_str)

    print(sep)

    # M2 table
    print(f"\nPer-slug M2 accuracy  (min_group_size={size})")
    print(sep)
    print(hdr.replace("(M1)", "(M2)"))
    print(sep)
    for slug in slugs:
        row_str = f"{slug:<30}"
        for cond in CONDITIONS:
            r = by_slug[slug].get(cond)
            if r is None:
                row_str += f"{'—':>{col_w}}"
            else:
                m2 = parse_float(r["m2_mean"])
                se_v = parse_float(r["m2_stderr"])
                if m2 is None:
                    row_str += f"{'—':>{col_w}}"
                elif se_v is not None:
                    row_str += f"{m2*100:>6.1f}±{se_v*100:<4.1f}{'':>{col_w - 14}}"
                else:
                    row_str += f"{m2*100:>{col_w}.1f}"
        print(row_str)
    print(sep)


# ---------------------------------------------------------------------------
# Size-stability table
# ---------------------------------------------------------------------------

def size_stability_table(rows: list[dict], sizes: list[int]) -> None:
    col_w = 14
    hdr = f"{'Condition':<22}" + "".join(f"size={s:>{col_w - 6}}" for s in sizes)
    sep = "-" * len(hdr)

    for metric, key in [("M1", "m1_mean"), ("M2", "m2_mean")]:
        print(f"\nSize stability: mean {metric} by condition × min_group_size")
        print(sep)
        print(hdr)
        print(sep)
        for cond in CONDITIONS:
            label = CONDITION_LABELS.get(cond, cond)
            row_str = f"{label:<22}"
            for s in sizes:
                agg = aggregate(rows, s)
                stats = agg.get(cond)
                val = stats.get(key) if stats else None
                if val is None:
                    row_str += f"{'—':>{col_w}}"
                else:
                    row_str += f"{val*100:>{col_w}.1f}"
            print(row_str)
        print(sep)


# ---------------------------------------------------------------------------
# Main summary table
# ---------------------------------------------------------------------------

def summary_table(rows: list[dict], size: int) -> None:
    agg = aggregate(rows, size)

    print(f"\n{'='*70}")
    print(f"  Validation sweep summary  |  min_group_size = {size}")
    print(f"{'='*70}")

    fmt_pct = lambda v, se: (
        f"{v*100:5.1f}%  ±{se*100:.1f}" if v is not None and se is not None else
        f"{v*100:5.1f}%      " if v is not None else
        "  —        "
    )

    hdr = f"  {'Condition':<22}  {'N slugs':>7}  {'M1 mean':>10}  {'M2 mean':>10}  {'Attr. cov':>10}  {'Trials':>8}"
    print(hdr)
    print(f"  {'-'*22}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for cond in CONDITIONS:
        label = CONDITION_LABELS.get(cond, cond)
        s = agg.get(cond)
        if s is None:
            print(f"  {label:<22}  {'—':>7}")
            continue
        m1_str = fmt_pct(s["m1_mean"], s["m1_se"])
        m2_str = fmt_pct(s["m2_mean"], s["m2_se"])
        cov = f"{s['attribution_coverage']*100:.1f}%" if s["attribution_coverage"] else "—"
        print(
            f"  {label:<22}  {s['n_slugs']:>7}  {m1_str:>16}  {m2_str:>16}  {cov:>10}  {s['total_trials']:>8}"
        )

    print(f"{'='*70}")
    print("  Note: M1 = 1-in-10 feature identification; M2 = 5-in-10 snippet match")
    print("  ±SE computed across slugs (macro-average).")
    print()


# ---------------------------------------------------------------------------
# Missing slugs report
# ---------------------------------------------------------------------------

def missing_report(rows: list[dict], size: int) -> None:
    slugs_per_cond: dict[str, set[str]] = defaultdict(set)
    skipped_per_cond: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in rows:
        if parse_int(r["min_group_size"]) != size:
            continue
        if r["skipped"].lower() == "true":
            skipped_per_cond[r["condition"]].append((r["slug"], r["skip_reason"]))
        else:
            slugs_per_cond[r["condition"]].add(r["slug"])

    all_slugs = {r["slug"] for r in rows}
    issues: list[str] = []
    for cond in CONDITIONS:
        present = slugs_per_cond[cond]
        missing = all_slugs - present - {s for s, _ in skipped_per_cond[cond]}
        if missing:
            issues.append(f"  {CONDITION_LABELS[cond]}: missing {sorted(missing)}")
        if skipped_per_cond[cond]:
            for slug, reason in skipped_per_cond[cond]:
                issues.append(f"  {CONDITION_LABELS[cond]}: {slug} skipped — {reason}")

    if issues:
        print(f"Missing / skipped at size={size}:")
        for line in issues:
            print(line)
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to validation_sweep.csv")
    parser.add_argument("--size", type=int, default=2, help="Primary min_group_size to report (default: 2)")
    parser.add_argument("--per-slug", action="store_true", help="Print per-slug breakdown table")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(csv_path)
    print(f"\nLoaded {len(rows)} rows from {csv_path.name}")

    all_sizes = sorted({parse_int(r["min_group_size"]) for r in rows if parse_int(r["min_group_size"])})
    all_slugs = sorted({r["slug"] for r in rows})
    print(f"Slugs ({len(all_slugs)}): {', '.join(all_slugs)}")
    print(f"Sizes: {all_sizes}")

    missing_report(rows, args.size)
    summary_table(rows, args.size)

    if len(all_sizes) > 1:
        size_stability_table(rows, all_sizes)

    if args.per_slug:
        per_slug_table(rows, args.size)


if __name__ == "__main__":
    main()
