"""
Plot mean harmonic BPM distance for positive vs negative pairs under both
random and hard negative sampling regimes. Single grouped bar chart with
two regimes side-by-side so the random→hard contrast is visible at a glance.

Reuses the data-loading + sampling helpers from train_all_models.py so the
numbers match exactly what the trained models actually saw.

Output
    paper/figures/harmonic_bpm_comparison.png

Usage
    python scripts/plot_harmonic_bpm.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_all_models import (  # noqa: E402
    FEATURES_CSV,
    TRANSITIONS_CSV,
    build_positives,
    filter_to_allow_list,
    harmonic_bpm_distance,
    sample_hard_negatives,
    sample_random_negatives,
)

figures_dir = REPO_ROOT / "paper" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)


def pick_sans_font():
    """Prefer Helvetica if it's installed; fall back to Nimbus Sans, the
    URW metric-identical Helvetica clone shipped with most Linux fontconfig
    distributions. Returning the first font matplotlib can actually render
    avoids the wall of 'font not found' warnings."""
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

POS_COLOR = "#2E86AB"
RAND_NEG_COLOR = "#E07A3F"
HARD_NEG_COLOR = "#A14B2A"


def harmonic_distances(pairs, tempo_by_track):
    out = np.empty(len(pairs))
    for i, (a, b) in enumerate(zip(pairs.track_1.values, pairs.track_2.values)):
        out[i] = harmonic_bpm_distance(tempo_by_track.get(a), tempo_by_track.get(b))
    return out


def plot_grouped_bars(pos_dist, rand_dist, hard_dist, out_path):
    """Grouped bars: each category (positives, random negs, hard negs) gets
    a mean bar and a median bar side by side. Same color per category, mean
    rendered solid and median rendered lighter so the eye reads them as a
    pair. This avoids the earlier version's issue of bar-height = mean and
    label = both, which conflates two statistics into one visual encoding."""
    import matplotlib.patches as mpatches

    means = np.array([pos_dist.mean(), rand_dist.mean(), hard_dist.mean()])
    medians = np.array([
        float(np.median(pos_dist)),
        float(np.median(rand_dist)),
        float(np.median(hard_dist)),
    ])
    colors = [POS_COLOR, RAND_NEG_COLOR, HARD_NEG_COLOR]
    labels = ["Positives", "Random\nnegatives", "Hard\nnegatives"]

    y_max = float(max(means.max(), medians.max())) * 1.30

    # Small native figsize, the figure renders at ~32% of \linewidth in
    # the paper so a small source size means LaTeX downscales less and
    # everything stays legible. Fonts are sized assuming this aspect.
    fig, ax = plt.subplots(figsize=(3.8, 2.8))

    group_centers = np.arange(3) * 1.4
    bar_w = 0.5
    mean_xs = group_centers - bar_w / 2 - 0.02
    med_xs = group_centers + bar_w / 2 + 0.02

    for x, h, c in zip(mean_xs, means, colors):
        ax.bar(x, h, width=bar_w, color=c, alpha=0.95,
               edgecolor="#333", linewidth=0.6)
    for x, h, c in zip(med_xs, medians, colors):
        ax.bar(x, h, width=bar_w, color=c, alpha=0.5,
               edgecolor="#333", linewidth=0.6)

    label_offset = y_max * 0.022
    for x, h in zip(mean_xs, means):
        ax.text(x, h + label_offset, f"{h:.2f}",
                ha="center", va="bottom", fontsize=12, color="#222",
                fontweight="medium")
    for x, h in zip(med_xs, medians):
        ax.text(x, h + label_offset, f"{h:.2f}",
                ha="center", va="bottom", fontsize=12, color="#222",
                fontweight="medium")

    ax.set_xticks(group_centers)
    ax.set_xticklabels(["Positives", "Random\nneg.", "Hard\nneg."],
                       fontsize=13)
    ax.set_ylabel("Harmonic BPM distance", fontsize=13)
    ax.set_ylim(0, y_max)
    ax.set_xlim(group_centers[0] - 0.85, group_centers[-1] + 0.85)
    ax.tick_params(axis="y", labelsize=11)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e6e6e6")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#999")
        ax.spines[spine].set_linewidth(0.8)

    # Legend in neutral gray so the per-category colors don't compete.
    mean_patch = mpatches.Patch(facecolor="#555", alpha=0.95,
                                edgecolor="#333", linewidth=0.6, label="Mean")
    med_patch = mpatches.Patch(facecolor="#555", alpha=0.5,
                               edgecolor="#333", linewidth=0.6, label="Median")
    ax.legend(handles=[mean_patch, med_patch], frameon=False,
              loc="upper right", fontsize=12, handlelength=1.3,
              borderpad=0.15, labelspacing=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}  (font: {SANS_FONT})")


def main():
    audio_features = pd.read_csv(FEATURES_CSV)
    transitions = pd.read_csv(TRANSITIONS_CSV)
    audio_features = filter_to_allow_list(audio_features)

    # sort the track pool so sampling is deterministic across runs
    # (set() iteration order varies with PYTHONHASHSEED)
    available_tracks = sorted(set(audio_features.track_id))
    tracks_by_id = audio_features.set_index("track_id")
    tempo_by_track = tracks_by_id["tempo"].to_dict()

    positives = build_positives(transitions, available_tracks)
    pos_dist = harmonic_distances(positives, tempo_by_track)
    print(f"positives: {len(positives)}  mean={pos_dist.mean():.2f}  median={np.median(pos_dist):.2f}")

    neg_random, _ = sample_random_negatives(
        positives, available_tracks, seed=42, ratio=1.0,
    )
    rand_dist = harmonic_distances(neg_random, tempo_by_track)
    print(f"random neg: {len(neg_random)}  mean={rand_dist.mean():.2f}  median={np.median(rand_dist):.2f}")

    neg_hard, _ = sample_hard_negatives(
        positives, tracks_by_id, available_tracks,
        seed=42, ratio=1.0, bpm_tolerance=5.0,
    )
    hard_dist = harmonic_distances(neg_hard, tempo_by_track)
    print(f"hard neg:   {len(neg_hard)}  mean={hard_dist.mean():.2f}  median={np.median(hard_dist):.2f}")

    plot_grouped_bars(
        pos_dist, rand_dist, hard_dist,
        figures_dir / "harmonic_bpm_comparison.png",
    )


if __name__ == "__main__":
    main()
