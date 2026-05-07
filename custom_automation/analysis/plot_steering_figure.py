"""
plot_steering_figure.py — Figure 2: causal steering on intermediate-hop supernodes.

Loads:
  analysis/amplify_experiment/amplify_middlehop_results.csv  (+2× constrained amplification)
  intervention/constrained_intervention_results.csv           (0× constrained suppression)

Both experiments use constrained patching.

p(correct) is approximated by the probability of the first subword token of the correct answer.
For multi-token answers this is an upper bound on the true sequence probability; we note this
in the paper.

Produces:
  analysis/results/steering_figure.pdf
  analysis/results/steering_figure.png

Usage:
    python analysis/plot_steering_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
PACKAGE_DIR = Path(__file__).resolve().parent.parent
AMP_CSV = PACKAGE_DIR / "analysis" / "amplify_experiment" / "amplify_middlehop_results.csv"
SUP_CSV = PACKAGE_DIR / "artifacts" / "constrained_intervention_results_ground_truth_capital.csv"
OUT_DIR = PACKAGE_DIR / "analysis" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "figure.figsize": (5.5, 2.4),
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DOT_COLOR = "#4878cf"


def _drop_invalid(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    mask = df["skipped"].isna() | (df["skipped"].astype(str).str.strip() == "")
    df = df[mask].copy()
    df = df[df["baseline_prob"].notna() & (df["baseline_prob"] > 0)]
    df = df[df[prob_col].notna()]
    return df


def load_amplify(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _drop_invalid(df, "amplified_prob")
    df["ratio"] = df["amplified_prob"] / df["baseline_prob"]
    df["model_correct"] = df["model_correct"].astype(str).str.lower() == "true"
    return df


def load_suppress(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _drop_invalid(df, "steered_prob")
    df["ratio"] = df["steered_prob"] / df["baseline_prob"]
    df["model_correct"] = df["baseline_rank"].notna() & (df["baseline_rank"].astype(float) == 1)
    return df



Y_MIN, Y_MAX = 0.125, 4.0
LABEL_FRAC_FLOOR = 0.07   # minimum axes-fraction for label position (avoids bottom tick overlap)


def plot_panel(
    ax: plt.Axes,
    amp_sub: pd.DataFrame,
    sup_sub: pd.DataFrame,
    title: str,
    show_ylabel: bool,
) -> None:
    rng = np.random.default_rng(42)

    for x, df in [(0, amp_sub), (1, sup_sub)]:
        ratios = np.asarray(df["ratio"].values, dtype=float)
        in_range = (ratios >= Y_MIN) & (ratios <= Y_MAX)
        n_below = int((ratios < Y_MIN).sum())
        n_above = int((ratios > Y_MAX).sum())

        jitter = rng.uniform(-0.10, 0.10, size=len(ratios))
        ax.scatter(x + jitter[in_range], ratios[in_range], color=DOT_COLOR,
                   alpha=0.55, s=12, linewidths=0, zorder=3)

        valid = ratios[ratios > 0]
        if len(valid):
            gm = float(np.exp(np.mean(np.log(valid))))
            gm_clipped = float(np.clip(gm, Y_MIN, Y_MAX))
            ax.plot([x - 0.18, x + 0.18], [gm_clipped, gm_clipped],
                    color="black", lw=1.8, ls="--", zorder=4)

            log_min = np.log10(Y_MIN)
            log_max = np.log10(Y_MAX)
            gm_frac = (np.log10(gm_clipped) - log_min) / (log_max - log_min)
            # Floor label position so it never overlaps the bottom tick
            label_base = max(gm_frac, LABEL_FRAC_FLOOR)
            line_h = 0.06
            xform = ax.get_xaxis_transform()

            ax.text(x + 0.20, label_base, f"gm={gm:.2g}×",
                    ha="left", va="center", fontsize=6, color="black",
                    transform=xform, zorder=5)
            stack = 1
            if n_below:
                ax.text(x + 0.20, label_base + stack * line_h,
                        f"↓{n_below} below",
                        ha="left", va="center", fontsize=6, color="#888888",
                        transform=xform, zorder=5)
                stack += 1
            if n_above:
                ax.text(x + 0.20, label_base + stack * line_h,
                        f"↑{n_above} above",
                        ha="left", va="center", fontsize=6, color="#888888",
                        transform=xform, zorder=5)

    ax.axhline(1.0, color="gray", lw=0.7, ls=":", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(Y_MIN, Y_MAX)
    ticks = [0.125, 0.25, 0.5, 1, 2, 4]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:g}\\times$"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$+2\times$", r"$0\times$"])
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel(
            r"$p_{\mathrm{steered}}(\mathrm{correct})\;/\;p_{\mathrm{baseline}}(\mathrm{correct})$"
        )
    ax.set_xlim(-0.5, 1.5)


def main() -> None:
    if not AMP_CSV.exists():
        raise FileNotFoundError(f"Run run_amplify_middlehop.py first: {AMP_CSV}")
    if not SUP_CSV.exists():
        raise FileNotFoundError(f"Run run_constrained_interventions.py first: {SUP_CSV}")

    amp_df = load_amplify(AMP_CSV)
    sup_df = load_suppress(SUP_CSV)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    plot_panel(ax1, amp_df[amp_df["model_correct"]], sup_df[sup_df["model_correct"]],
               "correct prompts", show_ylabel=True)
    plot_panel(ax2, amp_df[~amp_df["model_correct"]], sup_df[~sup_df["model_correct"]],
               "wrong prompts", show_ylabel=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"steering_figure.{ext}"
        fig.savefig(str(out), bbox_inches="tight")
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
