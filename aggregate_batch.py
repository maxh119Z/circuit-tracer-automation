"""
aggregate_batch.py — Average validation scores across all prompts in a batch run.

After running batch_reproduce.sh with validation, each slug has its own
validation_report_*.json. This script reads all of them, groups by
(desc_variant, grouping_variant), and averages macro scores across prompts.

Writes: artifacts/batch_summary.md

Usage:
    python aggregate_batch.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_FILE = ARTIFACTS_DIR / "batch_summary.md"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _parse_variant(stem: str) -> tuple[str, str]:
    """'validation_report_v2_a0' → ('v2', 'a0')"""
    without = stem[len("validation_report_"):]
    parts = without.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (without, "?")


def load_all_reports() -> list[dict]:
    """Load every validation_report_*.json found under artifacts/."""
    entries = []
    for p in sorted(ARTIFACTS_DIR.rglob("validation_report_*.json")):
        try:
            with open(p) as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        desc, grouping = _parse_variant(p.stem)
        # Derive slug from directory name (parent of the report file)
        rel = p.relative_to(ARTIFACTS_DIR)
        slug = rel.parts[0] if len(rel.parts) > 1 else "root"
        entries.append({
            "slug": slug,
            "desc": desc,
            "grouping": grouping,
            "prompt": report.get("prompt", ""),
            "report": report,
        })
    return entries


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _mac(report: dict, condition: str, neg_source: str, method: str) -> float:
    return (
        report.get(condition, {})
        .get(neg_source, {})
        .get(method, {})
        .get("macro_avg", {})
        .get("mean_accuracy", 0.0)
    )


def _stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


def aggregate(entries: list[dict]) -> dict[tuple[str, str], dict]:
    """
    Group entries by (desc, grouping), collect per-prompt macro scores,
    return averaged stats.
    """
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in entries:
        buckets[(e["desc"], e["grouping"])].append(e)

    results = {}
    for (desc, grouping), group in sorted(buckets.items()):
        def _collect(fn):
            return [fn(e["report"]) for e in group]

        m1e  = _collect(lambda r: _mac(r, "auto", "easy",   "method1"))
        m2e  = _collect(lambda r: _mac(r, "auto", "easy",   "method2"))
        m1m  = _collect(lambda r: _mac(r, "auto", "medium", "method1"))
        m2m  = _collect(lambda r: _mac(r, "auto", "medium", "method2"))
        m1h  = _collect(lambda r: _mac(r, "auto", "hard",   "method1"))
        m2h  = _collect(lambda r: _mac(r, "auto", "hard",   "method2"))
        d1   = _collect(lambda r: r.get("description_accuracy", {}).get("macro_avg", {}).get("mean_accuracy", 0.0))
        d2   = _collect(lambda r: r.get("description_snippet_accuracy", {}).get("macro_avg", {}).get("mean_accuracy", 0.0))
        cov  = _collect(lambda r: r.get("auto", {}).get("attribution_coverage", 0.0))

        def _stat(vals):
            n = len(vals)
            mean = sum(vals) / n if n else 0.0
            return {"mean": mean, "stderr": _stderr(vals), "n": n}

        results[(desc, grouping)] = {
            "desc": desc,
            "grouping": grouping,
            "n_prompts": len(group),
            "slugs": [e["slug"] for e in group],
            "m1_easy":   _stat(m1e),
            "m2_easy":   _stat(m2e),
            "m1_medium": _stat(m1m),
            "m2_medium": _stat(m2m),
            "m1_hard":   _stat(m1h),
            "m2_hard":   _stat(m2h),
            "d1":        _stat(d1),
            "d2":        _stat(d2),
            "coverage":  _stat(cov),
        }

    return results


# ---------------------------------------------------------------------------
# Write markdown
# ---------------------------------------------------------------------------

def _fmt(stat: dict) -> str:
    return f"{stat['mean']:.1%} ± {stat['stderr']:.1%}"


def write_summary(results: dict, output: Path) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n_prompts = max((v["n_prompts"] for v in results.values()), default=0)

    L: list[str] = [
        "# Batch Validation Summary",
        "",
        f"**Generated:** {timestamp}",
        f"**Prompts:** {n_prompts}  |  **Variants:** {len(results)}",
        "",
        "Scores are **mean ± stderr across prompts** (each prompt contributes its macro score).  ",
        "Higher is better. Chance baselines — M1: 10% | M2: 50% | D1: 10% | D2: 50%.",
        "",
        "---",
        "",
        "## Summary — Easy Difficulty",
        "",
        "| Desc | Grouping | M1 Easy | M2 Easy | D1 | D2 | Coverage | N |",
        "|:-----|:---------|--------:|--------:|---:|---:|---------:|--:|",
    ]

    for (desc, grouping), r in results.items():
        L.append(
            f"| `{desc}` | `{grouping}` "
            f"| {_fmt(r['m1_easy'])} | {_fmt(r['m2_easy'])} "
            f"| {_fmt(r['d1'])} | {_fmt(r['d2'])} "
            f"| {_fmt(r['coverage'])} "
            f"| {r['n_prompts']} |"
        )

    L += ["", "*N = number of prompts averaged.*", ""]

    # Harder difficulties
    has_hard = any(r["m1_hard"]["mean"] > 0 for r in results.values())
    if has_hard:
        L += [
            "## Harder Difficulties",
            "",
            "| Desc | Grouping | M1 Medium | M2 Medium | M1 Hard | M2 Hard |",
            "|:-----|:---------|----------:|----------:|--------:|--------:|",
        ]
        for (desc, grouping), r in results.items():
            L.append(
                f"| `{desc}` | `{grouping}` "
                f"| {_fmt(r['m1_medium'])} | {_fmt(r['m2_medium'])} "
                f"| {_fmt(r['m1_hard'])} | {_fmt(r['m2_hard'])} |"
            )
        L += [""]

    # Per-prompt breakdown
    L += [
        "## Per-Prompt Breakdown",
        "",
        "Individual prompt scores (M1 easy / M2 easy) for each variant.",
        "",
    ]

    # Group entries by (desc, grouping) for per-prompt table
    all_entries = load_all_reports()
    by_variant: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for e in all_entries:
        key = (e["desc"], e["grouping"])
        by_variant[key][e["slug"]] = e["report"]

    all_slugs = sorted({e["slug"] for e in all_entries})

    for (desc, grouping), r in results.items():
        L += [
            f"### `{desc}` / `{grouping}`",
            "",
            "| Prompt | M1 Easy | M2 Easy | D1 | D2 | Coverage |",
            "|:-------|--------:|--------:|---:|---:|---------:|",
        ]
        slug_reports = by_variant.get((desc, grouping), {})
        for slug in all_slugs:
            rep = slug_reports.get(slug)
            if rep is None:
                L.append(f"| `{slug}` | — | — | — | — | — |")
            else:
                m1 = _mac(rep, "auto", "easy", "method1")
                m2 = _mac(rep, "auto", "easy", "method2")
                d1 = rep.get("description_accuracy", {}).get("macro_avg", {}).get("mean_accuracy", 0.0)
                d2 = rep.get("description_snippet_accuracy", {}).get("macro_avg", {}).get("mean_accuracy", 0.0)
                cov = rep.get("auto", {}).get("attribution_coverage", 0.0)
                L.append(
                    f"| `{slug}` "
                    f"| {m1:.1%} | {m2:.1%} "
                    f"| {d1:.1%} | {d2:.1%} "
                    f"| {cov:.1%} |"
                )
        L += [""]

    L += [
        "---",
        "",
        f"*Generated {timestamp}*",
    ]

    output.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Batch summary written → {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    entries = load_all_reports()
    if not entries:
        print(f"No validation_report_*.json found under {ARTIFACTS_DIR}")
        return
    print(f"Found {len(entries)} report(s) across {len({e['slug'] for e in entries})} slug(s).")
    results = aggregate(entries)
    write_summary(results, OUTPUT_FILE)


if __name__ == "__main__":
    main()
