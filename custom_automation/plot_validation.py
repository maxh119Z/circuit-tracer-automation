"""
Plot validation results from validation_report.json or validation_history.json.

The report structure is nested by condition → neg_source → method:
  report["auto"]["easy"]["method1"]["groups"]
  report["auto"]["easy"]["method1"]["macro_avg"]
  report["auto"]["attribution_coverage"]

Neg sources: easy (other named groups) | medium (Ungrouped features) | hard (held-out 100-200)

Usage:
    python plot_validation.py                        # Per-group bar chart, M1, easy negatives
    python plot_validation.py --method m2            # Per-group bar chart, M2
    python plot_validation.py --source medium        # Use medium negatives
    python plot_validation.py --difficulty           # Macro accuracy across easy/medium/hard
    python plot_validation.py --history              # Macro accuracy trend over history
    python plot_validation.py --coverage             # Attribution coverage bar chart
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from config import VALIDATION_HISTORY_FILE, VALIDATION_REPORT_FILE, setup_logging

log = setup_logging()

COLORS = {
    "auto":   "#2196F3",
    "random": "#9E9E9E",
    "manual": "#4CAF50",
}

SOURCE_COLORS = {
    "easy":   "#2196F3",
    "medium": "#FF9800",
    "hard":   "#F44336",
}

CHANCE = {
    "method1": 0.10,   # 1-in-10
    "method2": 0.50,   # 5-in-10
}

METHOD_LABELS = {
    "method1": "Method 1 — Feature ID (1-in-10)",
    "method2": "Method 2 — Text Match (5-in-10)",
}

NEG_SOURCES = ("easy", "medium", "hard")


def _get_method_data(report: dict, cond: str, source: str, method: str) -> dict:
    """Safe accessor: report[cond][source][method], returns {} on missing keys."""
    return report.get(cond, {}).get(source, {}).get(method, {})


# ---------------------------------------------------------------------------
# Per-group bar chart (single report, one neg source)
# ---------------------------------------------------------------------------

def plot_report(report: dict, method: str = "method1", source: str = "easy", show: bool = True) -> None:
    """Bar chart of per-group accuracy for each condition at a given neg source."""
    method_label = METHOD_LABELS.get(method, method)
    conditions = [k for k in ("auto", "random", "manual") if k in report]

    auto_groups = _get_method_data(report, "auto", source, method).get("groups", [])
    group_names = [s["group"] for s in auto_groups]
    n_groups = len(group_names)

    if n_groups == 0:
        log.warning("No groups to plot for %s / %s / %s.", "auto", source, method)
        return

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.2), 6))
    fig.suptitle(
        f"{method_label} — Per-group Accuracy  [{source} negatives]\n"
        f"Prompt: {report.get('prompt', '')[:70]}",
        fontsize=10,
    )

    x = np.arange(n_groups)
    width = 0.8 / len(conditions)

    for i, cond in enumerate(conditions):
        groups_data = _get_method_data(report, cond, source, method).get("groups", [])
        stats_by_group = {s["group"]: s for s in groups_data}
        means   = [stats_by_group.get(g, {}).get("mean_accuracy",   0.0) for g in group_names]
        stderrs = [stats_by_group.get(g, {}).get("stderr_accuracy", 0.0) for g in group_names]

        offset = (i - len(conditions) / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width,
            yerr=stderrs, capsize=3,
            label=cond.capitalize(),
            color=COLORS.get(cond, "#FF9800"),
            alpha=0.85,
            error_kw={"elinewidth": 1.2},
        )

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([g[:22] for g in group_names], rotation=40, ha="right", fontsize=8)

    chance = CHANCE.get(method, 0.5)
    ax.axhline(chance, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
               label=f"chance ({chance:.0%})")
    ax.legend(loc="upper right")

    for cond in conditions:
        ma = _get_method_data(report, cond, source, method).get("macro_avg", {})
        cov = report[cond].get("attribution_coverage")
        cov_str = f"  cov={cov:.1%}" if cov is not None else ""
        log.info("%s/%s macro: %.3f ± %.3f%s",
                 cond, source, ma.get("mean_accuracy", 0), ma.get("stderr_accuracy", 0), cov_str)

    plt.tight_layout()
    out_path = VALIDATION_REPORT_FILE.parent / f"validation_plot_{method}_{source}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", out_path)
    plt.show() if show else plt.close()


# ---------------------------------------------------------------------------
# Difficulty comparison (easy vs medium vs hard) for one report
# ---------------------------------------------------------------------------

def plot_difficulty(report: dict, show: bool = True) -> None:
    """
    For each condition (auto, manual), show macro accuracy across easy/medium/hard
    for both methods. Reveals how much harder the task gets with tougher negatives.
    """
    conditions = [k for k in ("auto", "manual") if k in report]
    if not conditions:
        log.warning("No auto or manual condition found in report.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Accuracy vs Negative Source Difficulty\n"
        f"Prompt: {report.get('prompt', '')[:70]}",
        fontsize=10,
    )

    for ax, (method_key, method_label) in zip(axes, METHOD_LABELS.items()):
        chance = CHANCE[method_key]

        for cond in conditions:
            means, stderrs, labels = [], [], []
            for src in NEG_SOURCES:
                ma = _get_method_data(report, cond, src, method_key).get("macro_avg")
                if ma and ma.get("mean_accuracy", 0) > 0:
                    means.append(ma["mean_accuracy"])
                    stderrs.append(ma["stderr_accuracy"])
                    labels.append(src)

            if not means:
                continue

            xs = np.arange(len(labels))
            color = COLORS.get(cond, "#FF9800")
            ax.errorbar(
                xs, means, yerr=stderrs,
                marker="o", linewidth=2, capsize=4,
                label=cond.capitalize(), color=color,
            )

        ax.set_title(method_label, fontsize=9)
        ax.set_ylabel("Macro Accuracy (mean ± stderr)")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(np.arange(len(NEG_SOURCES)))
        ax.set_xticklabels([s.capitalize() for s in NEG_SOURCES])
        ax.set_xlabel("Negative Source")
        ax.axhline(chance, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
                   label=f"chance ({chance:.0%})")
        ax.legend()

    plt.tight_layout()
    out_path = VALIDATION_REPORT_FILE.parent / "validation_difficulty_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", out_path)
    plt.show() if show else plt.close()


# ---------------------------------------------------------------------------
# Macro accuracy trend over history
# ---------------------------------------------------------------------------

def plot_history(history: list[dict], source: str = "easy", show: bool = True) -> None:
    """Line plot of macro accuracy trend across history entries, both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Validation History — Macro Accuracy Trend  [{source} negatives]",
        fontsize=11,
    )

    for ax, (method_key, method_label) in zip(axes, METHOD_LABELS.items()):
        for cond in ("auto", "random", "manual"):
            means, stderrs = [], []
            for entry in history:
                if cond not in entry:
                    continue
                ma = _get_method_data(entry, cond, source, method_key).get("macro_avg")
                if ma is None:
                    continue
                means.append(ma["mean_accuracy"])
                stderrs.append(ma["stderr_accuracy"])

            if not means:
                continue

            xs = list(range(len(means)))
            color = COLORS.get(cond, "#FF9800")
            ax.plot(xs, means, marker="o", label=cond.capitalize(), color=color)
            ax.fill_between(
                xs,
                [m - e for m, e in zip(means, stderrs)],
                [m + e for m, e in zip(means, stderrs)],
                alpha=0.2, color=color,
            )

        chance = CHANCE[method_key]
        ax.set_title(method_label)
        ax.set_ylabel("Macro Accuracy (mean ± stderr)")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("History Run Index")
        ax.axhline(chance, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
                   label=f"chance ({chance:.0%})")
        ax.legend()

    plt.tight_layout()
    out_path = VALIDATION_HISTORY_FILE.parent / f"validation_history_plot_{source}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", out_path)
    plt.show() if show else plt.close()


# ---------------------------------------------------------------------------
# Attribution coverage bar chart
# ---------------------------------------------------------------------------

def plot_coverage(report: dict, show: bool = True) -> None:
    """Simple bar chart comparing attribution coverage across conditions."""
    conditions = [
        k for k in ("auto", "manual") if k in report
        and report[k].get("attribution_coverage") is not None
    ]
    if not conditions:
        log.warning("No attribution coverage data in report.")
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.suptitle("Attribution Coverage\n(fraction of influence score in annotated groups)")

    coverages = [report[cond]["attribution_coverage"] for cond in conditions]
    colors = [COLORS.get(cond, "#FF9800") for cond in conditions]

    ax.bar([c.capitalize() for c in conditions], coverages, color=colors, alpha=0.85)
    ax.set_ylabel("Coverage Fraction")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(coverages):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)

    plt.tight_layout()
    out_path = VALIDATION_REPORT_FILE.parent / "validation_coverage_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", out_path)
    plt.show() if show else plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation results.")
    parser.add_argument(
        "--history", action="store_true",
        help="Plot macro accuracy trend across all history runs.",
    )
    parser.add_argument(
        "--method", choices=["m1", "m2"], default="m1",
        help="Which method to show in per-group chart (default: m1).",
    )
    parser.add_argument(
        "--source", choices=["easy", "medium", "hard"], default="easy",
        help="Which negative source to use for per-group and history plots (default: easy).",
    )
    parser.add_argument(
        "--difficulty", action="store_true",
        help="Plot macro accuracy across easy/medium/hard neg sources.",
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Plot attribution coverage comparison.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate and save all plots to disk (no interactive windows).",
    )
    args = parser.parse_args()

    if not VALIDATION_REPORT_FILE.exists():
        log.error("No report file at %s — run validate_groups.py first.", VALIDATION_REPORT_FILE)
        return

    with open(VALIDATION_REPORT_FILE) as f:
        report = json.load(f)

    if args.all:
        for method_key in ("method1", "method2"):
            for src in ("easy", "medium", "hard"):
                plot_report(report, method=method_key, source=src, show=False)
        plot_difficulty(report, show=False)
        plot_coverage(report, show=False)
        if VALIDATION_HISTORY_FILE.exists():
            with open(VALIDATION_HISTORY_FILE) as f:
                history = json.load(f)
            for src in ("easy", "medium", "hard"):
                plot_history(history, source=src, show=False)
        log.info("All plots saved.")
        return

    if args.history:
        if not VALIDATION_HISTORY_FILE.exists():
            log.error("No history file at %s", VALIDATION_HISTORY_FILE)
            return
        with open(VALIDATION_HISTORY_FILE) as f:
            history = json.load(f)
        plot_history(history, source=args.source)
        return

    if args.difficulty:
        plot_difficulty(report)
        return

    method_key = "method1" if args.method == "m1" else "method2"
    plot_report(report, method=method_key, source=args.source)

    if args.coverage:
        plot_coverage(report)


if __name__ == "__main__":
    main()
