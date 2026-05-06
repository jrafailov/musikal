"""
Plot harmonic BPM distance distributions for positive vs negative pairs,
under both random and hard negative sampling regimes. Replaces the text-dump
PNGs that earlier showed train_all_models.py's printed output.

Reuses the data-loading + sampling helpers from train_all_models.py so the
distributions match exactly what the trained models actually saw.

Outputs
    paper/figures/harmonic_bpm_random.png
    paper/figures/harmonic_bpm_hard.png

Usage
    python scripts/plot_harmonic_bpm.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

plt.rcParams["font.family"] = ["Helvetica", "Arial", "Nimbus Sans", "sans-serif"]

POS_COLOR = "#2E86AB"
NEG_COLOR = "#E07A3F"


def harmonic_distances(pairs, tempo_by_track):
    out = np.empty(len(pairs))
    for i, (a, b) in enumerate(zip(pairs.track_1.values, pairs.track_2.values)):
        out[i] = harmonic_bpm_distance(tempo_by_track.get(a), tempo_by_track.get(b))
    return out


def plot_distributions(pos_dist, neg_dist, neg_label, out_path, y_max):
    """Violin plot of harmonic BPM distance for positives vs negatives. Each
    violin shows the full density shape; mean (diamond) and median (line)
    are overlaid. Y-axis range is shared across regimes so the random and
    hard plots can be compared directly."""

    pos_mean, neg_mean = pos_dist.mean(), neg_dist.mean()
    pos_med, neg_med = float(np.median(pos_dist)), float(np.median(neg_dist))

    fig, ax = plt.subplots(figsize=(6, 4))

    parts = ax.violinplot(
        [pos_dist, neg_dist],
        positions=[1, 2],
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], [POS_COLOR, NEG_COLOR]):
        body.set_facecolor(color)
        body.set_edgecolor("#333")
        body.set_alpha(0.65)
        body.set_linewidth(1.2)

    # median as a short horizontal bar inside each violin
    for x, med in zip([1, 2], [pos_med, neg_med]):
        ax.hlines(med, x - 0.18, x + 0.18, color="black", linewidth=2.5, zorder=4)

    # mean as a diamond
    ax.scatter([1, 2], [pos_mean, neg_mean],
               marker="D", s=70, color="black", zorder=5)

    # numeric annotations just above each violin
    annotation_y = y_max * 0.96
    ax.text(1, annotation_y, f"mean {pos_mean:.2f}\nmedian {pos_med:.2f}",
            ha="center", va="top", fontsize=10, color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=POS_COLOR, linewidth=1.2))
    ax.text(2, annotation_y, f"mean {neg_mean:.2f}\nmedian {neg_med:.2f}",
            ha="center", va="top", fontsize=10, color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=NEG_COLOR, linewidth=1.2))

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"Positives\n(n={len(pos_dist)})",
                        f"{neg_label}\n(n={len(neg_dist)})"], fontsize=11)
    ax.set_ylabel("Harmonic BPM distance\n(lower = closer tempo match)", fontsize=11)
    ax.set_ylim(0, y_max)
    ax.set_xlim(0.3, 2.7)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


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

    # shared y-axis so the two regimes are visually comparable. driven by
    # the random regime's tail since it's the wider distribution.
    y_max = float(np.percentile(
        np.concatenate([pos_dist, rand_dist, hard_dist]), 95,
    ))
    print(f"shared y_max (95th pct across all): {y_max:.2f}")

    plot_distributions(
        pos_dist, rand_dist, "Random negatives",
        figures_dir / "harmonic_bpm_random.png", y_max,
    )
    plot_distributions(
        pos_dist, hard_dist, "Hard negatives",
        figures_dir / "harmonic_bpm_hard.png", y_max,
    )


if __name__ == "__main__":
    main()
