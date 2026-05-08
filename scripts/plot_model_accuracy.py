"""
Plot per-model test accuracy across the two negative-sampling regimes
(random vs hard). One small grouped bar chart, six models, two bars per
group. Sorted by random-neg accuracy so the eye reads left-to-right.

The story this figure carries: Siamese is best on random and worst on
hard, tree models cluster around 0.71-0.72 on hard. Hard negatives strip
the easy harmonic-BPM cue and force the model onto features the trees
handle better.

Output
    paper/figures/model_accuracy.png

Usage
    python scripts/plot_model_accuracy.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = REPO_ROOT / "data" / "clean" / "baseline" / "all_models"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def pick_sans_font():
    """Match plot_harmonic_bpm.py: prefer Helvetica, fall back through
    the same chain so figure typography looks consistent."""
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Helvetica", "Helvetica Neue", "Nimbus Sans",
                      "Liberation Sans", "Arial", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "sans-serif"


SANS_FONT = pick_sans_font()
plt.rcParams["font.family"] = SANS_FONT
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

RANDOM_COLOR = "#2E86AB"
HARD_COLOR = "#E07A3F"

# Short labels so the x-axis stays readable at small figure sizes.
SHORT_NAMES = {
    "RF default (100 trees)": "RF default",
    "RF tuned (500 trees, depth 20)": "RF tuned",
    "Gradient Boosting": "Grad. Boost",
    "LightGBM": "LightGBM",
    "PCA latent (10 PCs) + RF": "PCA + RF",
    "Siamese (early stopping)": "Siamese",
}


def load_results():
    rand = pd.read_csv(RESULTS_DIR / "random" / "all_models_results.csv")
    hard = pd.read_csv(RESULTS_DIR / "hard" / "all_models_results.csv")
    rand = rand.set_index("model")["test_accuracy"]
    hard = hard.set_index("model")["test_accuracy"]
    # sort by random-neg accuracy descending
    order = rand.sort_values(ascending=False).index.tolist()
    return order, rand.loc[order], hard.loc[order]


def _draw_panel(ax, labels, values, color, title):
    """One panel of bars for a single negative-sampling regime."""
    n = len(labels)
    x = np.arange(n)
    ax.bar(x, values, width=0.7, color=color, alpha=0.95,
           edgecolor="#333", linewidth=0.6)

    # value labels above each bar. Sized small so adjacent labels never
    # touch; black text reads cleanly against the white panel background.
    for xi, h in zip(x, values):
        ax.text(xi, h + 0.018, f"{h:.2f}",
                ha="center", va="bottom", fontsize=6.5, color="#000",
                fontweight="bold")

    # chance baseline
    ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.7,
               alpha=0.7, zorder=0)

    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.7, n - 0.3)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e6e6e6")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999")
        ax.spines[spine].set_linewidth(0.8)


def plot_grouped_bars(order, rand_acc, hard_acc, out_path):
    """Two side-by-side panels, random on the left and hard on the right.
    Same x-axis order in both so the eye can chase a single model across
    regimes. Sorted by random-negative accuracy descending."""
    labels = [SHORT_NAMES[m] for m in order]

    # Landscape aspect so the chart stays short vertically when stacked
    # with feature_importance_gbm.png in Figure 3.
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.0), sharey=True)

    _draw_panel(axes[0], labels, rand_acc.values,
                RANDOM_COLOR, "Random negatives")
    _draw_panel(axes[1], labels, hard_acc.values,
                HARD_COLOR, "Hard negatives ($\\pm$5 BPM)")

    # Only the leftmost panel keeps the y-axis label and tick labels;
    # sharey=True hides the duplicate ticks on the right panel
    # automatically but the axis spine still gets drawn, so we hint that
    # the shared scale is "Test accuracy".
    axes[0].set_ylabel("Test accuracy", fontsize=9)

    # "chance" annotation goes only on the left panel so it appears once.
    axes[0].text(-0.55, 0.51, "chance",
                 fontsize=6.5, color="#666", ha="left", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}  (font: {SANS_FONT})")


def main():
    order, rand_acc, hard_acc = load_results()
    print("model order (by random-neg accuracy):")
    for m in order:
        print(f"  {m:40s}  random={rand_acc[m]:.3f}  hard={hard_acc[m]:.3f}")
    plot_grouped_bars(order, rand_acc, hard_acc,
                      FIGURES_DIR / "model_accuracy.png")


if __name__ == "__main__":
    main()
