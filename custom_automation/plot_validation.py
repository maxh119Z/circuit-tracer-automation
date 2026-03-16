"""
Plot validation results from validation_report.json or validation_history.json.

Usage:
    python plot_validation.py                # Latest report — per-group bar chart
    python plot_validation.py --history      # History trend — macro accuracy over time
    python plot_validation.py --method m2    # Show Method 2 per-group instead of M1
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

# Chance baselines for each method
CHANCE = {
    "method1": 0.10,  # 1-in-10
    "method2": 0.50,  # 5-in-10
}

METHOD_LABELS = {
    "method1": "Method 1 — Feature ID (1-in-10)",
    "method2": "Method 2 — Text Match (5-in-10)",
}


# ---------------------------------------------------------------------------
# Per-group bar chart (single report)
# ---------------------------------------------------------------------------

def plot_report(report: dict, method: str = "method1") -> None:
    """Bar chart of per-group accuracy with error bars for each condition."""
    method_label = METHOD_LABELS.get(method, method)
    conditions = [k for k in ("auto", "random", "manual") if k in report]

    group_stats = report["auto"][method]["groups"]
    group_names = [s["group"] for s in group_stats]
    n_groups = len(group_names)

    if n_groups == 0:
        log.warning("No groups to plot for %s.", method)
        return

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.2), 6))
    fig.suptitle(
        f"{method_label} — Per-group Accuracy\n"
        f"Prompt: {report.get('prompt', '')[:70]}",
        fontsize=10,
    )

    x = np.arange(n_groups)
    width = 0.8 / len(conditions)

    for i, cond in enumerate(conditions):
        stats_by_group = {s["group"]: s for s in report[cond][method]["groups"]}
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
    ax.set_xticklabels(
        [g[:22] for g in group_names],
        rotation=40, ha="right", fontsize=8,
    )

    chance = CHANCE.get(method, 0.5)
    ax.axhline(chance, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label=f"chance ({chance:.0%})")
    ax.legend(loc="upper right")

    for cond in conditions:
        ma = report[cond][method]["macro_avg"]
        cov = report[cond].get("attribution_coverage")
        cov_str = f"  cov={cov:.1%}" if cov is not None else ""
        log.info(
            "%s macro accuracy: %.3f ± %.3f%s",
            cond, ma["mean_accuracy"], ma["stderr_accuracy"], cov_str,
        )

    plt.tight_layout()
    out_path = VALIDATION_REPORT_FILE.parent / f"validation_plot_{method}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", out_path)
    plt.show()


# ---------------------------------------------------------------------------
# Macro accuracy trend over history runs
# ---------------------------------------------------------------------------

def plot_history(history: list[dict]) -> None:
    """Line plot of macro accuracy trend across all history entries, both methods."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Validation History — Macro Accuracy Trend", fontsize=11)

    methods = [("method1", METHOD_LABELS["method1"]), ("method2", METHOD_LABELS["method2"])]

    for ax, (method_key, method_label) in zip(axes, methods):
        for cond in ("auto", "random", "manual"):
            means, stderrs = [], []
            for entry in history:
                if cond not in entry:
                    continue
                ma = entry[cond][method_key]["macro_avg"]
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

        chance = CHANCE.get(method_key, 0.5)
        ax.set_title(method_label)
        ax.set_ylabel("Macro Accuracy (mean ± stderr)")
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("History Run Index")
        ax.axhline(chance, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label=f"chance ({chance:.0%})")
        ax.legend()

    plt.tight_layout()
    out_path = VALIDATION_HISTORY_FILE.parent / "validation_history_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", out_path)
    plt.show()


# ---------------------------------------------------------------------------
# Attribution coverage bar chart
# ---------------------------------------------------------------------------

def plot_coverage(report: dict) -> None:
    """Simple bar chart comparing attribution coverage across conditions."""
    conditions = [k for k in ("auto", "manual") if k in report and report[k].get("attribution_coverage") is not None]
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
    plt.show()


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
        "--coverage", action="store_true",
        help="Plot attribution coverage comparison.",
    )
    args = parser.parse_args()

    if args.history:
        if not VALIDATION_HISTORY_FILE.exists():
            log.error("No history file at %s", VALIDATION_HISTORY_FILE)
            return
        with open(VALIDATION_HISTORY_FILE) as f:
            history = json.load(f)
        plot_history(history)
        return

    if not VALIDATION_REPORT_FILE.exists():
        log.error("No report file at %s — run validate_groups.py first.", VALIDATION_REPORT_FILE)
        return

    with open(VALIDATION_REPORT_FILE) as f:
        report = json.load(f)

    method_key = "method1" if args.method == "m1" else "method2"
    plot_report(report, method=method_key)

    if args.coverage:
        plot_coverage(report)


if __name__ == "__main__":
    main()