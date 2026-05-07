"""
Genre breakdown of the tracks we have audio features for.

Pulls genre tags from djmix_mixes (the `tags` column carries the MixesDB
Category labels), maps them down to a curated whitelist of dance/electronic
genres, propagates the mix-level genre to each track, and renders two
alternative views of the top-10 distribution.

A track that appears in mixes of different genres gets the most common one.

Original author: Jasmine. Adapted for repo paths and bubble alternative added.

Output
    data/clean/audio_features_with_genre.csv
    paper/figures/genre_distribution.png         (pie)
    paper/figures/genre_distribution_bubbles.png (bubble chart)

Usage
    python scripts/plot_genre_distribution.py
"""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless WSL has no display
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_sans_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Helvetica", "Helvetica Neue", "Nimbus Sans",
                      "Liberation Sans", "Arial", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "sans-serif"


plt.rcParams["font.family"] = _pick_sans_font()
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

REPO_ROOT = Path(__file__).resolve().parents[1]

MIXES_CSV = REPO_ROOT / "data" / "tracklists" / "djmix_mixes.csv"
TRACKS_CSV = REPO_ROOT / "data" / "tracklists" / "djmix_tracks.csv"
FEATURES_CSV = REPO_ROOT / "data" / "clean" / "audio_features.csv"

OUT_CSV = REPO_ROOT / "data" / "clean" / "audio_features_with_genre.csv"
OUT_PIE = REPO_ROOT / "paper" / "figures" / "genre_distribution.png"
OUT_BUBBLES = REPO_ROOT / "paper" / "figures" / "genre_distribution_bubbles.png"
OUT_STACKED = REPO_ROOT / "paper" / "figures" / "genre_distribution_stacked.png"

# Whitelist of genres we care about. Order matters: the first match in this
# list wins, so multi-word genres (Progressive House) are checked before the
# single-word version (House) to avoid swallowing the more specific tag.
GENRE_KEYWORDS = [
    'Progressive House', 'Deep House', 'Tech House', 'Hard House',
    'Drum & Bass', 'UK Garage', 'Hip Hop', 'Trip Hop',
    'Progressive Trance', 'Psytrance',
    'House', 'Techno', 'Trance', 'Jungle',
    'Ambient', 'Minimal', 'Disco', 'Electro', 'Dubstep',
    'Hardcore', 'Acid', 'Breakbeat', 'IDM', 'Industrial',
    'Funk', 'Soul', 'Garage', 'Rave', 'Dub', 'Downtempo',
]
GENRE_KEYWORDS_LOWER = [g.lower() for g in GENRE_KEYWORDS]

COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2',
    '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
]


def extract_primary_genre(tags_str):
    """First whitelist genre that appears in the pipe-separated MixesDB tags."""
    if pd.isna(tags_str):
        return 'Unknown'
    parts = [p.replace('Category:', '').strip().lower() for p in tags_str.split('|')]
    for g, g_lower in zip(GENRE_KEYWORDS, GENRE_KEYWORDS_LOWER):
        if g_lower in parts:
            return g
    return 'Unknown'


def label_tracks(mixes, tracks, audio_features):
    mixes = mixes.copy()
    mixes['genre'] = mixes['tags'].apply(extract_primary_genre)

    # A track can show up in multiple mixes with different genres. Take the
    # mode so each track has exactly one label.
    track_mix = tracks.merge(mixes[['mix_id', 'genre']], on='mix_id', how='left')
    track_genre = (
        track_mix.groupby('track_id')['genre']
        .agg(lambda x: Counter(x).most_common(1)[0][0])
        .reset_index()
    )

    out = audio_features.merge(track_genre, on='track_id', how='left')
    out['genre'] = out['genre'].fillna('Unknown')
    return out


def plot_pie(top10, top10_pct, out_path):
    """Landscape pie with legend on the right."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.pie(
        top10_pct,
        labels=None,
        startangle=140,
        colors=COLORS,
        wedgeprops=dict(linewidth=0.8, edgecolor='white'),
    )
    ax.set_aspect('equal')

    legend_labels = [f"{g}  ({p:.1f}%)" for g, p in zip(top10.index, top10_pct.values)]
    patches = [mpatches.Patch(color=COLORS[i], label=legend_labels[i])
               for i in range(len(top10))]
    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=False, handlelength=1.2, handleheight=1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def wrap_label(name):
    """Break two-word genre names like 'Progressive House' onto two lines so
    they fit within a fixed-width slot below each bubble."""
    if ' ' in name and len(name) > 9:
        first, rest = name.split(' ', 1)
        return f"{first}\n{rest}"
    return name


def plot_bubbles(top10, top10_pct, out_path):
    """Horizontal strip of circles, sorted by share descending. Each circle's
    area is proportional to the genre's track count. Circles are placed on
    even x-slots so labels never collide, even though that means the smallest
    circles sit with extra whitespace around them."""

    counts = top10.values.astype(float)
    pcts = top10_pct.values
    names = top10.index.tolist()

    n = len(counts)

    # Area scales with count, so radius scales with sqrt(count). Pick r_max
    # so the largest circle fills its slot comfortably; everything else
    # follows by ratio.
    slot = 1.0
    r_max = 0.43 * slot
    radii = r_max * np.sqrt(counts / counts.max())
    centers_x = np.arange(n) * slot

    fig, ax = plt.subplots(figsize=(8.5, 3.2))

    inside_threshold = 0.28  # circles bigger than this get the % printed inside

    for cx, r, name, pct, color in zip(centers_x, radii, names, pcts, COLORS):
        ax.add_patch(plt.Circle(
            (cx, 0), r, facecolor=color, alpha=0.88,
            edgecolor='white', linewidth=1.0,
        ))

        if r >= inside_threshold:
            ax.text(cx, 0, f"{pct:.1f}%",
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

        # Genre name + (% if not already inside) below each circle, on a
        # common baseline so labels line up across slots.
        wrapped = wrap_label(name)
        below = wrapped if r >= inside_threshold else f"{wrapped}\n{pct:.1f}%"
        ax.text(cx, -r_max - 0.10, below,
                ha='center', va='top', fontsize=8.5,
                color='black', linespacing=1.15)

    ax.set_xlim(-slot * 0.55, (n - 1) * slot + slot * 0.55)
    ax.set_ylim(-r_max - 1.00, r_max + 0.12)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def plot_bubbles_stacked(top10, top10_pct, out_path):
    """NYT-style packed bubbles: circles sorted big-to-small are tangent and
    share a common bottom baseline, so the tops form a descending skyline.
    Big circles label themselves inside; the small tail uses leader lines to
    an evenly-spaced callout strip below the baseline so labels never crash
    into each other."""

    counts = top10.values.astype(float)
    pcts = top10_pct.values
    names = top10.index.tolist()
    n = len(counts)

    r_max = 1.0
    radii = r_max * np.sqrt(counts / counts.max())

    # Tangent packing left-to-right, all bottoms on y=0.
    centers_x = np.zeros(n)
    centers_x[0] = radii[0]
    for i in range(1, n):
        centers_x[i] = centers_x[i - 1] + radii[i - 1] + radii[i]
    centers_y = radii  # so bottom of each circle sits on y=0
    total_w = centers_x[-1] + radii[-1]

    # Smaller native figsize so LaTeX scales it less when the minipage
    # collapses to ~0.42 \linewidth, otherwise font sizes get clobbered.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    # Faint baseline anchors the visual bottom edge of all circles.
    ax.plot([-0.2, total_w + 0.2], [0, 0],
            color='#bbbbbb', linewidth=0.6, zorder=0)

    # Per-genre inside-vs-callout decision: a circle holds its label iff its
    # radius is big enough to fit the longest wrapped line at a readable
    # fontsize. Long names (Progressive...) need a bigger circle than short
    # ones (House) to label themselves.
    # Inside if the circle's radius can hold the longest wrapped word at
    # the smallest readable font we're willing to use.
    inside_idx, callout_idx = [], []
    for i, (r, name) in enumerate(zip(radii, names)):
        longest = max(wrap_label(name).split('\n'), key=len)
        threshold = max(0.45, len(longest) * 0.072)
        (inside_idx if r >= threshold else callout_idx).append(i)

    for i in range(n):
        cx, cy, r, color = centers_x[i], centers_y[i], radii[i], COLORS[i]
        ax.add_patch(plt.Circle(
            (cx, cy), r, facecolor=color, alpha=0.88,
            edgecolor='white', linewidth=1.0, zorder=2,
        ))

    # Per-circle font scales so the longest wrapped word fits inside,
    # capped at 14pt for the largest circle. Coefficient 13 here is the
    # inverse of the threshold's 0.072 (with a small buffer).
    for i in inside_idx:
        cx, cy, r = centers_x[i], centers_y[i], radii[i]
        longest = max(wrap_label(names[i]).split('\n'), key=len)
        fs = min(14.0, max(9.5, r / max(len(longest), 4) * 13.0))
        ax.text(cx, cy, f"{wrap_label(names[i])}\n{pcts[i]:.1f}%",
                ha='center', va='center',
                fontsize=fs, fontweight='bold',
                color='white', zorder=3, linespacing=1.15)

    # Callout strip below the baseline. Labels are spread across a wider
    # x-range than the tail's footprint so they get breathing room; thin
    # leader lines connect each circle to its callout.
    if callout_idx:
        # Spread callouts over a wider range so the bigger label font
        # ("Progressive Trance" is the longest) doesn't crash into its
        # neighbor. We push the starting x further left and let the strip
        # extend slightly past the rightmost circle.
        first = callout_idx[0]
        callout_x_start = max(0.0, centers_x[first] - 4.5)
        callout_x_end = total_w + 0.4
        callout_xs = np.linspace(callout_x_start, callout_x_end, len(callout_idx))
        callout_y = -1.05

        for idx, lx in zip(callout_idx, callout_xs):
            cx_real = centers_x[idx]
            ax.plot([cx_real, lx], [-0.04, callout_y + 0.18],
                    color='#999999', linewidth=0.6, zorder=1)
            ax.text(lx, callout_y, f"{names[idx]}\n{pcts[idx]:.1f}%",
                    ha='center', va='top', fontsize=12, color='black',
                    linespacing=1.15)

    ax.set_xlim(-0.30, total_w + 0.30)
    ax.set_ylim(-1.65, 2 * r_max + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def main():
    mixes = pd.read_csv(MIXES_CSV)
    tracks = pd.read_csv(TRACKS_CSV)
    af = pd.read_csv(FEATURES_CSV)

    af_labeled = label_tracks(mixes, tracks, af)

    top10 = af_labeled['genre'].value_counts().head(10)
    top10_pct = (top10 / top10.sum() * 100).round(1)

    print("Top 10 genres in audio_features")
    for genre, count, pct in zip(top10.index, top10.values, top10_pct.values):
        print(f"  {genre:<22} {count:>5} tracks  ({pct}%)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    af_labeled.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV.relative_to(REPO_ROOT)}")

    OUT_PIE.parent.mkdir(parents=True, exist_ok=True)
    plot_pie(top10, top10_pct, OUT_PIE)
    plot_bubbles(top10, top10_pct, OUT_BUBBLES)
    plot_bubbles_stacked(top10, top10_pct, OUT_STACKED)


if __name__ == "__main__":
    main()
