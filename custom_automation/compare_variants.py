"""
compare_variants.py — Aggregate all validation reports into a single comparison markdown.

Scans artifacts/ recursively for every validation_report_*.json, reads each one,
and writes artifacts/comparison.md with all variants in flat, readable tables.

Run this after any batch that produces multiple validation reports:
    python compare_variants.py

Works for all-groups runs (same desc, different grouping),
all-variants runs (different desc, same grouping), and mixed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
OUTPUT_FILE = ARTIFACTS_DIR / "comparison.md"


# ---------------------------------------------------------------------------
# Load reports
# ---------------------------------------------------------------------------

def _parse_variant_from_stem(stem: str) -> tuple[str, str]:
    """Extract (desc, grouping) from 'validation_report_v2_a0' → ('v2', 'a0')."""
    without_prefix = stem[len("validation_report_"):]
    parts = without_prefix.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return without_prefix, "?"


def find_reports() -> list[dict]:
    """
    Scan artifacts/ recursively for validation_report_*.json.
    Returns list of dicts with: desc, grouping, label, report, timestamp_str.
    Deduplicates by (desc, grouping), keeping the most recently generated report.
    """
    seen: dict[tuple[str, str], dict] = {}

    for p in sorted(ARTIFACTS_DIR.rglob("validation_report_*.json")):
        try:
            with open(p) as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        desc, grouping = _parse_variant_from_stem(p.stem)
        key = (desc, grouping)
        ts = report.get("timestamp", "")

        if key not in seen or ts > seen[key]["timestamp_str"]:
            seen[key] = {
                "desc": desc,
                "grouping": grouping,
                "label": f"{desc} / {grouping}",
                "report": report,
                "timestamp_str": ts,
            }

    # Sort: by desc first, then grouping
    return sorted(seen.values(), key=lambda x: (x["desc"], x["grouping"]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mac(report: dict, condition: str, neg_source: str, method: str) -> dict:
    return (
        report.get(condition, {})
        .get(neg_source, {})
        .get(method, {})
        .get("macro_avg", {"mean_accuracy": 0.0, "stderr_accuracy": 0.0})
    )


def _fmt(d: dict) -> str:
    return f"{d.get('mean_accuracy', 0.0):.1%} ± {d.get('stderr_accuracy', 0.0):.1%}"


# ---------------------------------------------------------------------------
# Write markdown
# ---------------------------------------------------------------------------

def write_comparison(entries: list[dict], output: Path) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    n = len(entries)

    L: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    L += [
        "# Variant Comparison",
        "",
        f"**Generated:** {timestamp}  |  **Variants found:** {n}",
        "",
        "Scores are **mean ± stderr** across runs. Higher is better.",
        "Sorted by description variant, then grouping variant.",
        "",
        "---",
        "",
    ]

    if not entries:
        L.append("No validation reports found under `artifacts/`.")
        output.write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"Comparison written → {output}")
        return

    # ── Summary table ────────────────────────────────────────────────────────
    # Desc | Grouping | M1 Easy | M2 Easy | D1 | D2 | Coverage
    # D1 and D2 only depend on descriptions, so identical across grouping variants —
    # that pattern is visible naturally when rows share the same desc.

    L += [
        "## Summary",
        "",
        "**Easy** negatives = other named groups (best case for the grouper).  ",
        "**D1** and **D2** are description-only — they will be the same across grouping variants "
        "that share the same description variant.",
        "",
        "| Desc | Grouping | M1 Easy | M2 Easy | D1 | D2 | Coverage |",
        "|:-----|:---------|--------:|--------:|---:|---:|---------:|",
    ]

    for e in entries:
        r = e["report"]
        m1e = _mac(r, "auto", "easy", "method1")
        m2e = _mac(r, "auto", "easy", "method2")
        d1  = r.get("description_accuracy", {}).get("macro_avg", {"mean_accuracy": 0.0, "stderr_accuracy": 0.0})
        d2  = r.get("description_snippet_accuracy", {}).get("macro_avg", {"mean_accuracy": 0.0, "stderr_accuracy": 0.0})
        cov = r.get("auto", {}).get("attribution_coverage")
        cov_str = f"{cov:.1%}" if cov is not None else "—"
        L.append(
            f"| `{e['desc']}` | `{e['grouping']}` "
            f"| {_fmt(m1e)} | {_fmt(m2e)} "
            f"| {_fmt(d1)} | {_fmt(d2)} "
            f"| {cov_str} |"
        )

    L += [
        "",
        "*Chance baselines — M1: 10% | M2: 50% | D1: 10% | D2: 50%*",
        "",
    ]

    # ── Harder difficulties ──────────────────────────────────────────────────
    # Only include if at least one report has non-zero medium/hard scores
    has_hard = any(
        _mac(e["report"], "auto", "hard", "method1").get("mean_accuracy", 0) > 0
        for e in entries
    )
    if has_hard:
        L += [
            "## Harder Difficulties",
            "",
            "**Medium** negatives = Ungrouped features. "
            "**Hard** negatives = features outside the Phase 1 seed window.",
            "",
            "| Desc | Grouping | M1 Medium | M2 Medium | M1 Hard | M2 Hard |",
            "|:-----|:---------|----------:|----------:|--------:|--------:|",
        ]
        for e in entries:
            r = e["report"]
            m1m = _mac(r, "auto", "medium", "method1")
            m2m = _mac(r, "auto", "medium", "method2")
            m1h = _mac(r, "auto", "hard",   "method1")
            m2h = _mac(r, "auto", "hard",   "method2")
            L.append(
                f"| `{e['desc']}` | `{e['grouping']}` "
                f"| {_fmt(m1m)} | {_fmt(m2m)} "
                f"| {_fmt(m1h)} | {_fmt(m2h)} |"
            )
        L += ["", "*Chance baselines — M1: 10% | M2: 50%*", ""]

    # ── Random baseline ──────────────────────────────────────────────────────
    random_rows = [
        (e, _mac(e["report"], "random", "easy", "method1"),
             _mac(e["report"], "random", "easy", "method2"))
        for e in entries
        if _mac(e["report"], "random", "easy", "method1").get("mean_accuracy", 0) > 0
    ]
    if random_rows:
        L += [
            "## Random Baseline",
            "",
            "Shuffled group assignments. Theoretical chance: M1 = 10%, M2 = 50%.",
            "",
            "| Desc | Grouping | M1 Random | M2 Random |",
            "|:-----|:---------|----------:|----------:|",
        ]
        for e, rm1, rm2 in random_rows:
            L.append(
                f"| `{e['desc']}` | `{e['grouping']}` "
                f"| {_fmt(rm1)} | {_fmt(rm2)} |"
            )
        L += [""]

    # ── Per-variant detail links ─────────────────────────────────────────────
    L += [
        "## Individual Reports",
        "",
        "Full per-group breakdowns are in the individual markdown reports:",
        "",
    ]
    for e in entries:
        desc, grouping = e["desc"], e["grouping"]
        # Root-level or slug-based path
        root_md = ARTIFACTS_DIR / f"validation_report_{desc}_{grouping}.md"
        slug_md = ARTIFACTS_DIR / f"{desc}-{grouping}" / f"validation_report_{desc}_{grouping}.md"
        if slug_md.exists():
            md_path = slug_md.relative_to(ARTIFACTS_DIR.parent)
        elif root_md.exists():
            md_path = root_md.relative_to(ARTIFACTS_DIR.parent)
        else:
            md_path = None
        if md_path:
            L.append(f"- `{e['label']}` → [{md_path}]({md_path})")
        else:
            L.append(f"- `{e['label']}` → *(markdown not found)*")
    L += [""]

    # ── How to read ──────────────────────────────────────────────────────────
    L += [
        "---",
        "",
        "## How to read this",
        "",
        "| Method | Question it answers | Chance |",
        "|:-------|:--------------------|-------:|",
        "| **M1** Feature ID (1-in-10) | Given a group name, can the model pick the right feature description? Measures group coherence and naming quality. | 10% |",
        "| **M2** Text Match (5-in-10) | Given a group name, can the model pick the 5 activating text snippets? Measures how well the name captures real input patterns. | 50% |",
        "| **D1** Description Accuracy (1-in-10) | Given raw feature evidence, can the model identify its description? Measures description quality. | 10% |",
        "| **D2** Description Snippets (5-in-10) | Given a description, can the model pick the 5 correct activating snippets? Also measures description quality. | 50% |",
        "",
        "**Coverage** = fraction of total influence score flowing through annotated (non-Ungrouped) nodes. "
        "Higher means more of the circuit is explained.",
        "",
        f"*Generated {timestamp}*",
    ]

    output.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Comparison written → {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    entries = find_reports()
    if not entries:
        print(f"No validation_report_*.json files found under {ARTIFACTS_DIR}")
    else:
        labels = [e["label"] for e in entries]
        print(f"Found {len(entries)} report(s): {labels}")
    write_comparison(entries, OUTPUT_FILE)


if __name__ == "__main__":
    main()
