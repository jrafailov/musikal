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
RAND_NEG_COLOR = "#E07A3F"
HARD_NEG_COLOR = "#A14B2A"


def harmonic_distances(pairs, tempo_by_track):
    out = np.empty(len(pairs))
    for i, (a, b) in enumerate(zip(pairs.track_1.values, pairs.track_2.values)):
        out[i] = harmonic_bpm_distance(tempo_by_track.get(a), tempo_by_track.get(b))
    return out


def plot_grouped_bars(pos_dist, rand_dist, hard_dist, out_path):
    """Three-bar comparison: positives, random negatives, hard negatives.
    The positives bar is identical across regimes so it only needs to appear
    once. Mean and median annotated above each bar."""

    pos_mean, pos_med = pos_dist.mean(), float(np.median(pos_dist))
    rand_mean, rand_med = rand_dist.mean(), float(np.median(rand_dist))
    hard_mean, hard_med = hard_dist.mean(), float(np.median(hard_dist))

    largest_mean = max(pos_mean, rand_mean, hard_mean)
    y_max = largest_mean * 1.45

    fig, ax = plt.subplots(figsize=(5, 3.8))

    xs = [1, 2, 3]
    means = [pos_mean, rand_mean, hard_mean]
    medians = [pos_med, rand_med, hard_med]
    colors = [POS_COLOR, RAND_NEG_COLOR, HARD_NEG_COLOR]
    labels = ["Positives", "Random\nnegatives", "Hard\nnegatives"]

    ax.bar(xs, means, width=0.6, color=colors,
           edgecolor="#333", linewidth=1.0, alpha=0.85)

    label_offset = y_max * 0.025
    for x, mean, median, color in zip(xs, means, medians, colors):
        ax.text(x, mean + label_offset,
                f"mean {mean:.2f}\nmed {median:.2f}",
                ha="center", va="bottom", fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=color, linewidth=1.0))

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Mean harmonic BPM distance\n(lower = closer tempo match)",
                  fontsize=10)
    ax.set_ylim(0, y_max)
    ax.set_xlim(0.4, 3.6)
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

    plot_grouped_bars(
        pos_dist, rand_dist, hard_dist,
        figures_dir / "harmonic_bpm_comparison.png",
    )


if __name__ == "__main__":
    main()
