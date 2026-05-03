"""
summarize_neuronpedia_validation.py — Aggregate every neuronpedia validation
into per-min_group_size comparison tables.

Scans artifacts/*/validation_history_v2_a2_neuropedia.json (which appends a new
entry every time validation runs at any size, so all sizes are preserved) and
emits one .md / .csv pair per min_group_size:

  custom_automation/analysis/neuronpedia_validation_results/
    neuronpedia_summary_min2.md   neuronpedia_summary_min2.csv
    neuronpedia_summary_min3.md   neuronpedia_summary_min3.csv
    neuronpedia_summary_min4.md   neuronpedia_summary_min4.csv
    ...

Each summary file contains:
  - M1 (per prompt × condition)
  - M2 (per prompt × condition)
  - Macro mean across prompts (where condition has valid groups)
  - Coverage & valid-group counts (per prompt × condition)
  - Cross-size comparison table (mean by min_group_size — same content in every file)

Run:
    # All min sizes that the validation history covers:
    python custom_automation/analysis/summarize_neuronpedia_validation.py

    # Or explicitly:
    python custom_automation/analysis/summarize_neuronpedia_validation.py --min-sizes 2,3
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
RESULTS_DIR = Path(__file__).resolve().parent / "neuronpedia_validation_results"
HISTORY_NAME = "validation_history_v2_a2_neuropedia.json"
DIFFICULTY = "medium"
CONDITIONS = ["random", "human", "ours-no-reconciliation", "ours-full"]
SHORT_LABELS = {
    "random": "random",
    "human": "human",
    "ours-no-reconciliation": "ours-no-rec",
    "ours-full": "ours-full",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_bos(prompt: str) -> str:
    return prompt.removeprefix("<bos>").strip()


def _macro(report: dict, condition: str, method: str) -> tuple[float | None, float | None]:
    cond = report.get("conditions", {}).get(condition, {})
    diff = cond.get(DIFFICULTY, {})
    macro = diff.get(method, {}).get("macro_avg", {})
    if not macro:
        return None, None
    n_valid = cond.get("n_groups_valid", 0)
    if n_valid == 0:
        return None, None
    return macro.get("mean_accuracy"), macro.get("stderr_accuracy")


def _coverage(report: dict, condition: str) -> tuple[float | None, int, int]:
    cond = report.get("conditions", {}).get(condition, {})
    cov = cond.get("attribution_coverage")
    valid = cond.get("n_groups_valid", 0)
    total = cond.get("n_groups_total", 0)
    return cov, valid, total


def _fmt(value: float | None, stderr: float | None = None) -> str:
    if value is None:
        return "—"
    if stderr is None:
        return f"{value * 100:.1f}%"
    return f"{value * 100:.1f}% ± {stderr * 100:.1f}%"


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _description_metrics(entry: dict) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (d1_mean, d1_stderr, d2_mean, d2_stderr) from a report entry, or Nones."""
    desc = entry.get("description_metrics") or {}
    d1 = desc.get("d1", {}).get("macro_avg", {})
    d2 = desc.get("d2", {}).get("macro_avg", {})
    return (
        d1.get("mean_accuracy"), d1.get("stderr_accuracy"),
        d2.get("mean_accuracy"), d2.get("stderr_accuracy"),
    )


def collect_by_size() -> tuple[dict[int, list[dict]], dict[int, dict[str, dict[str, list[float]]]], dict[str, dict]]:
    """
    Scan every history file once. Returns:
      per_size_rows:        {size: [row, ...]}      one row per (slug, size); slug-level D1/D2 fields included
      cross_size:           {size: {cond: {m1: [means], m2: [means]}}}  for the cross-size table
      desc_metrics_by_slug: {slug: {d1_mean, d1_stderr, d2_mean, d2_stderr}}  one entry per slug
    """
    per_size_rows: dict[int, list[dict]] = {}
    cross_size: dict[int, dict[str, dict[str, list[float]]]] = {}
    desc_metrics_by_slug: dict[str, dict] = {}

    for path in sorted(ARTIFACTS_DIR.glob(f"*/{HISTORY_NAME}")):
        slug = path.parent.name
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
        for entry in history:
            size = entry.get("min_group_size")
            if not isinstance(size, int):
                continue

            prompt = _strip_bos(entry.get("prompt", ""))
            d1_m, d1_s, d2_m, d2_s = _description_metrics(entry)
            # Cache the latest non-null description metrics seen for this slug
            # (D1/D2 don't depend on size — same number every entry that has them).
            if d1_m is not None or d2_m is not None:
                desc_metrics_by_slug[slug] = {
                    "d1_mean": d1_m, "d1_stderr": d1_s,
                    "d2_mean": d2_m, "d2_stderr": d2_s,
                }

            row: dict = {"slug": slug, "prompt": prompt, "min_group_size": size,
                         "d1_mean": d1_m, "d1_stderr": d1_s,
                         "d2_mean": d2_m, "d2_stderr": d2_s}
            for cond in CONDITIONS:
                m1_mean, m1_se = _macro(entry, cond, "method1")
                m2_mean, m2_se = _macro(entry, cond, "method2")
                cov, valid, total = _coverage(entry, cond)
                row[f"{cond}__m1_mean"] = m1_mean
                row[f"{cond}__m1_stderr"] = m1_se
                row[f"{cond}__m2_mean"] = m2_mean
                row[f"{cond}__m2_stderr"] = m2_se
                row[f"{cond}__coverage"] = cov
                row[f"{cond}__valid"] = valid
                row[f"{cond}__total"] = total

                cs = cross_size.setdefault(size, {}).setdefault(cond, {"m1": [], "m2": []})
                if m1_mean is not None:
                    cs["m1"].append(m1_mean)
                if m2_mean is not None:
                    cs["m2"].append(m2_mean)

            per_size_rows.setdefault(size, []).append(row)

    # Backfill slug-level D1/D2 onto every row (so a size that didn't include
    # description_metrics still shows the cached value in the per-prompt table).
    for rows in per_size_rows.values():
        for r in rows:
            if r["d1_mean"] is None and r["d2_mean"] is None:
                cached = desc_metrics_by_slug.get(r["slug"])
                if cached:
                    r.update(cached)

    # Sort rows within each size by prompt for stable output.
    for size, rows in per_size_rows.items():
        rows.sort(key=lambda r: (r["prompt"], r["slug"]))
    return per_size_rows, cross_size, desc_metrics_by_slug


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def write_markdown(out_path: Path, size: int, rows: list[dict], cross_size: dict, desc_by_slug: dict) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = []
    lines.append(f"# Neuronpedia Validation Summary — min_group_size={size}")
    lines.append("")
    lines.append(f"**Generated:** {now}  ")
    lines.append(f"**Prompts:** {len(rows)}  |  **Difficulty:** {DIFFICULTY}  |  **min_group_size:** {size}  ")
    lines.append("**Variant:** desc=`v2`, grouping=`a2`")
    lines.append("")
    lines.append("M1 = 1-in-10 feature ID (chance 10%). M2 = 5-in-10 text match (chance 50%). "
                 "Cells show mean ± stderr across 5 runs. `—` = no valid groups for that condition at this min_group_size.")
    lines.append("")

    header = ["Prompt"] + [SHORT_LABELS[c] for c in CONDITIONS]

    # M1 per-prompt
    lines.append(f"## Method 1 — feature identification (chance 10%, min_group_size={size})")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([":--"] + ["---:"] * len(CONDITIONS)) + "|")
    for row in rows:
        cells = [f"`{row['prompt']}`"]
        for cond in CONDITIONS:
            cells.append(_fmt(row[f"{cond}__m1_mean"], row[f"{cond}__m1_stderr"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # M2 per-prompt
    lines.append(f"## Method 2 — text snippet matching (chance 50%, min_group_size={size})")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join([":--"] + ["---:"] * len(CONDITIONS)) + "|")
    for row in rows:
        cells = [f"`{row['prompt']}`"]
        for cond in CONDITIONS:
            cells.append(_fmt(row[f"{cond}__m2_mean"], row[f"{cond}__m2_stderr"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Macro means (this size only)
    lines.append(f"## Mean across prompts (min_group_size={size})")
    lines.append("")
    lines.append("Each cell = mean of per-prompt macro accuracies for prompts that had valid groups for the condition.")
    lines.append("")
    lines.append("| Method | " + " | ".join(SHORT_LABELS[c] for c in CONDITIONS) + " |")
    lines.append("|" + "|".join([":--"] + ["---:"] * len(CONDITIONS)) + "|")
    for method, label in [("m1", "M1"), ("m2", "M2")]:
        cells = [label]
        for cond in CONDITIONS:
            vals = [r[f"{cond}__{method}_mean"] for r in rows if r[f"{cond}__{method}_mean"] is not None]
            if not vals:
                cells.append("—")
            else:
                cells.append(f"{(sum(vals) / len(vals)) * 100:.1f}% (n={len(vals)})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Cross-size comparison (same content in every file, included for context)
    lines.append("## Mean across prompts by `min_group_size` (cross-size view)")
    lines.append("")
    lines.append("Same data appears in every per-size summary file — included so you can see how the means shift with the threshold.")
    lines.append("")
    sizes_present = sorted(cross_size.keys())
    for method, label in [("m1", "M1 — feature ID (chance 10%)"),
                          ("m2", "M2 — text match (chance 50%)")]:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| min_group_size | " + " | ".join(SHORT_LABELS[c] for c in CONDITIONS) + " |")
        lines.append("|" + "|".join([":--"] + ["---:"] * len(CONDITIONS)) + "|")
        for s in sizes_present:
            cells = [str(s) + (" *(this file)*" if s == size else "")]
            for cond in CONDITIONS:
                vals = cross_size.get(s, {}).get(cond, {}).get(method, [])
                if not vals:
                    cells.append("—")
                else:
                    cells.append(f"{(sum(vals) / len(vals)) * 100:.1f}% (n={len(vals)})")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Description quality (D1 / D2) — slug-level, size-independent
    lines.append("## Description Quality (D1 / D2)")
    lines.append("")
    lines.append("Per-feature tests of how well the auto-generated descriptions fit each feature's evidence. "
                 "Slug-level — does **not** depend on grouping condition or `min_group_size`. "
                 "D1 = feature evidence → pick correct description (1-in-10, chance 10%). "
                 "D2 = description → pick activating snippets (5-in-10, chance 50%).")
    lines.append("")
    lines.append("| Prompt | D1 | D2 |")
    lines.append("|:--|---:|---:|")
    for row in rows:
        slug = row["slug"]
        d = desc_by_slug.get(slug, {})
        d1 = _fmt(d.get("d1_mean"), d.get("d1_stderr"))
        d2 = _fmt(d.get("d2_mean"), d.get("d2_stderr"))
        lines.append(f"| `{row['prompt']}` | {d1} | {d2} |")
    # Aggregate row across slugs that have description metrics.
    d1_vals = [d["d1_mean"] for d in desc_by_slug.values() if d.get("d1_mean") is not None]
    d2_vals = [d["d2_mean"] for d in desc_by_slug.values() if d.get("d2_mean") is not None]
    d1_mean = sum(d1_vals) / len(d1_vals) if d1_vals else None
    d2_mean = sum(d2_vals) / len(d2_vals) if d2_vals else None
    d1_str = f"{d1_mean*100:.1f}% (n={len(d1_vals)})" if d1_mean is not None else "—"
    d2_str = f"{d2_mean*100:.1f}% (n={len(d2_vals)})" if d2_mean is not None else "—"
    lines.append(f"| **Mean across prompts** | **{d1_str}** | **{d2_str}** |")
    lines.append("")

    # Coverage & valid-group counts
    lines.append(f"## Coverage & valid-group counts (min_group_size={size})")
    lines.append("")
    lines.append("Coverage = fraction of attribution-graph influence captured by the condition's groups. "
                 "Valid/Total = groups large enough to score at this min_group_size.")
    lines.append("")
    cov_header = ["Prompt"]
    for cond in CONDITIONS:
        cov_header.extend([f"{SHORT_LABELS[cond]} cov", f"{SHORT_LABELS[cond]} grp"])
    lines.append("| " + " | ".join(cov_header) + " |")
    lines.append("|" + "|".join([":--"] + ["---:"] * (2 * len(CONDITIONS))) + "|")
    for row in rows:
        cells = [f"`{row['prompt']}`"]
        for cond in CONDITIONS:
            cov = row[f"{cond}__coverage"]
            cov_s = f"{cov * 100:.0f}%" if cov is not None else "—"
            cells.append(cov_s)
            cells.append(f"{row[f'{cond}__valid']}/{row[f'{cond}__total']}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} prompts)")


def write_csv(out_path: Path, rows: list[dict]) -> None:
    fieldnames = ["slug", "prompt", "min_group_size",
                  "d1_mean", "d1_stderr", "d2_mean", "d2_stderr"]
    for cond in CONDITIONS:
        fieldnames.extend([
            f"{cond}__m1_mean", f"{cond}__m1_stderr",
            f"{cond}__m2_mean", f"{cond}__m2_stderr",
            f"{cond}__coverage", f"{cond}__valid", f"{cond}__total",
        ])
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
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

    per_size_rows, cross_size, desc_by_slug = collect_by_size()
    if not per_size_rows:
        print(f"No history files found matching {ARTIFACTS_DIR}/*/{HISTORY_NAME}")
        return

    available = sorted(per_size_rows.keys())
    if args.min_sizes:
        requested = parse_int_list(args.min_sizes)
        missing = [s for s in requested if s not in per_size_rows]
        if missing:
            print(f"Skipping requested sizes with no data: {missing}. Available: {available}")
        wanted = [s for s in requested if s in per_size_rows]
        if not wanted:
            print(f"None of the requested sizes have data. Available: {available}")
            return
    else:
        wanted = available

    print(f"Summarising for min_group_size values: {wanted}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for size in wanted:
        rows = per_size_rows[size]
        out_md  = RESULTS_DIR / f"neuronpedia_summary_min{size}.md"
        out_csv = RESULTS_DIR / f"neuronpedia_summary_min{size}.csv"
        write_markdown(out_md, size, rows, cross_size, desc_by_slug)
        write_csv(out_csv, rows)


if __name__ == "__main__":
    main()