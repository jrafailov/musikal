"""
Side-by-side feature importance panels for the Gradient Boosting classifier
under random and hard negative regimes. Importances are aggregated to base
features (A_tempo + B_tempo + |diff|_tempo collapse into a single 'Tempo'
row), matching the style of the original poster.

The harmonic_bpm_distance bar is highlighted so its rank-1 → rank-5 collapse
between regimes is the dominant visual.

Reads importance CSVs written by train_all_models.py.

Output
    paper/figures/feature_importance_gbm.png

Usage
    python scripts/plot_feature_importance.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless WSL has no display
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RANDOM_CSV = REPO_ROOT / "data/clean/baseline/all_models/random/feature_importance_gbm.csv"
HARD_CSV = REPO_ROOT / "data/clean/baseline/all_models/hard/feature_importance_gbm.csv"

figures_dir = REPO_ROOT / "paper" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)


def _pick_sans_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Helvetica", "Helvetica Neue", "Nimbus Sans",
                 "Liberation Sans", "Arial", "DejaVu Sans"):
        if cand in available:
            return cand
    return "sans-serif"


# theme_pubr-style defaults: clean white background, no grid, black spines
# and ticks, sans-serif throughout. Font sizes pushed up so the figure stays
# legible after LaTeX downscales it into the side-by-side caption layout.
plt.rcParams.update({
    "font.family": _pick_sans_font(),
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelweight": "bold",
    "axes.linewidth": 1.0,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# harmonic BPM gets the same blue used for positives in plot_harmonic_bpm.py
HARMONIC_COLOR = "#2E86AB"
NEUTRAL_COLOR = "#7a7a7a"
PANEL_TITLE_COLOR = "black"

TOP_N = 10


def base_feature(raw):
    """Strip A_, B_, |diff|_ prefixes so e.g. A_tempogram_ratio_mean_5 collapses
    to 'tempogram_ratio_mean_5'. The numeric index stays since MFCC 0 and
    MFCC 6 are different features."""
    if raw.startswith("|diff|_"):
        return raw[len("|diff|_"):]
    if raw.startswith("A_") or raw.startswith("B_"):
        return raw[2:]
    return raw


def pretty_name(base):
    """Map a base feature name to a poster-friendly label."""
    if base == "harmonic_bpm_distance":
        return "Harmonic BPM dist."
    rules = [
        ("tempogram_ratio_mean_", "Tempogram "),
        ("tempogram_ratio_std_", "Tempogram σ "),
        ("mfcc_mean_", "MFCC "),
        ("mfcc_std_", "MFCC σ "),
        ("chroma_cens_mean_", "Chroma "),
        ("chroma_cens_std_", "Chroma σ "),
        ("tonnetz_mean_", "Tonnetz "),
        ("tonnetz_std_", "Tonnetz σ "),
        ("spectral_contrast_mean_", "Spec contrast "),
        ("spectral_contrast_std_", "Spec contrast σ "),
        ("spectral_centroid_mean", "Spec centroid"),
        ("spectral_centroid_std", "Spec centroid σ"),
        ("spectral_bandwidth_mean", "Spec bandwidth"),
        ("spectral_bandwidth_std", "Spec bandwidth σ"),
        ("spectral_rolloff_mean", "Spec rolloff"),
        ("spectral_rolloff_std", "Spec rolloff σ"),
        ("spectral_flatness_mean", "Spec flatness"),
        ("spectral_flatness_std", "Spec flatness σ"),
        ("zero_crossing_rate_mean", "ZCR"),
        ("zero_crossing_rate_std", "ZCR σ"),
        ("rms_mean", "RMS"),
        ("rms_std", "RMS σ"),
        ("tempo", "Tempo"),
    ]
    for pattern, replacement in rules:
        if base.startswith(pattern):
            return replacement + base[len(pattern):]
    return base


def aggregate_to_base(df):
    """Collapse A_x, B_x, |diff|_x rows into a single x row by summing."""
    df = df.copy()
    df["base"] = df.feature.map(base_feature)
    grouped = df.groupby("base", as_index=False).importance.sum()
    return grouped.sort_values("importance", ascending=False).reset_index(drop=True)


def plot_panel(ax, agg, title, title_color, x_max):
    top = agg.head(TOP_N).iloc[::-1].reset_index(drop=True)  # flip so #1 ends on top
    pretty = [pretty_name(b) for b in top.base]
    pcts = top.importance.values * 100.0

    colors = [HARMONIC_COLOR if name == "Harmonic BPM dist." else NEUTRAL_COLOR
              for name in pretty]

    y = np.arange(len(top))
    ax.barh(y, pcts, color=colors, edgecolor="black", linewidth=0.8)

    label_offset = x_max * 0.012
    for i, pct in enumerate(pcts):
        ax.text(pct + label_offset, i, f"{pct:.1f}%",
                ha="left", va="center", fontsize=12)

    ax.set_yticks(y)
    ax.set_yticklabels(pretty)
    ax.set_xlabel("Importance (%)")
    ax.set_xlim(0, x_max * 1.15)
    ax.set_title(title, color=title_color, pad=10)

    # theme_pubr: keep only bottom + left spines, hide top + right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", length=4, width=1.0)


def main():
    random_df = pd.read_csv(RANDOM_CSV)
    hard_df = pd.read_csv(HARD_CSV)

    random_agg = aggregate_to_base(random_df)
    hard_agg = aggregate_to_base(hard_df)

    # shared x-axis so bar widths are visually comparable across panels
    x_max = max(random_agg.head(TOP_N).importance.max(),
                hard_agg.head(TOP_N).importance.max()) * 100.0

    # Wider, shorter aspect: avoids the wasted vertical space the previous
    # 10.5 x 6.0 layout left between panel titles and bars. Suptitle is
    # dropped because the caption already names the figure.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    plot_panel(axes[0], random_agg, "Random negatives",
               PANEL_TITLE_COLOR, x_max)
    plot_panel(axes[1], hard_agg, "Hard negatives (±5 BPM)",
               PANEL_TITLE_COLOR, x_max)

    out_path = figures_dir / "feature_importance_gbm.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")

    print("\nrandom negatives, top 5:")
    print(random_agg.head(5).to_string(index=False))
    print("\nhard negatives, top 5:")
    print(hard_agg.head(5).to_string(index=False))

    rand_rank = (random_agg.base.values == "harmonic_bpm_distance").argmax() + 1
    hard_rank = (hard_agg.base.values == "harmonic_bpm_distance").argmax() + 1
    print(f"\nharmonic_bpm_distance rank — random: #{rand_rank}, hard: #{hard_rank}")


if __name__ == "__main__":
    main()
