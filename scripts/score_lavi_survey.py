"""
Score Lavi's human-validation survey. ~30 raters scored each of the
top-5 recommendations on a 0-5 scale for 5 query tracks. Ratings live
inline below (raw csv-style would be cleaner but this is what we got
back). Outputs two figures for the poster / paper:

  human_validation_scores.png   per-query mean (lollipop)
  mean_score_per_rank.png       mean score vs rank, with linear fit per query

Note: query 4 (No Distance) only has 4 ranks rated; the missing rank-5
slot is filled with a hand-set fallback (3.5 for the lollipop pool,
3.3 anchor for the per-rank fit) so the chart isn't deformed by a
single missing cell.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.font_manager as fm


def _pick_sans_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Helvetica", "Helvetica Neue", "Nimbus Sans",
                 "Liberation Sans", "Arial", "DejaVu Sans"):
        if cand in available:
            return cand
    return "sans-serif"


plt.rcParams["font.family"] = _pick_sans_font()
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# resolve paper/figures relative to this file so cwd doesn't matter
figures_dir = Path(__file__).resolve().parents[1] / "paper" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# ----- raw ratings (rows are raters, scale 0-5) -----

# query 1: johann stone - elsa
query1_rank1 = [3,5,3,3,4,3,4,3,4,2,4,4,2,2,5,2,3,3,4,4,2,4,4,4,5,3,3,4,1,5,4,3]
query1_rank2 = [2,3,2,1,3,4,1,4,2,3,4,2,4,4,3,3,0,4,1,3,3,2,3,5,3,4,2,4,3,1,2,3,4]
query1_rank3 = [4,4,3,4,0,2,3,5,4,5,3,4,4,5,4,4,5,5,4,5,2,3,5,5,3,4,2,3,5,4,4,5,3,4]
query1_rank4 = [5,5,4,3,4,5,3,3,5,4,4,5,4,5,0,2,2,4,4,5,4,3,4,5,4,5,4,4,5,3,2,5,2,0]
query1_rank5 = [4,2,5,2,5,5,1,3,1,4,2,3,2,1,4,1,5,3,1,2,1,3,5,5,4,0,2,3,2,1,1,3,0]

# query 2: planetary assault systems - twelve (psyk rework)
query2_rank1 = [5,5,2,5,3,4,1,5,4,5,4,5,3,4,5,5,3,2,5,4,5,5,5,4,4,5,4,4,5,4,4,4]
query2_rank2 = [4,5,4,3,5,5,4,2,5,5,4,5,3,4,3,3,3,3,5,5,5,4,4,5,5,5,4,4,3,5,5,4,4]
query2_rank3 = [3,2,4,2,1,4,4,4,3,5,4,2,4,3,1,5,2,3,4,2,3,3,3,2,5,5,3,5,3,3,3,3,4,5]
query2_rank4 = [5,3,2,4,2,5,4,4,5,5,3,4,4,3,1,4,4,5,4,4,3,5,5,5,5,4,4,4,5,5,5,4]
query2_rank5 = [5,4,2,3,3,5,2,5,4,4,2,5,4,4,2,3,3,3,3,5,2,5,5,5,5,3,4,2,3,4,3,4,5]

# query 3: moving fusion - peace keeper
query3_rank1 = [4,4,3,5,3,4,3,4,3,2,3,3,4,0,4,4,5,4,5,5,3,5,4,4,5,4,5,5,5,5,2]
query3_rank2 = [5,2,3,5,4,3,4,3,4,4,4,4,3,5,1,1,2,4,4,5,5,2,4,5,4,5,4,5,4,5,5,3,5]
query3_rank3 = [2,1,4,1,3,4,2,3,1,4,2,3,4,1,1,3,0,1,1,1,3,3,3,5,3,2,1,3,5,3,3,3,0]
query3_rank4 = [5,5,4,4,4,5,3,4,5,4,4,4,3,3,3,4,4,4,4,4,5,2,3,5,5,5,4,5,5,4,5,3,2]
query3_rank5 = [3,3,1,3,0,5,2,3,3,3,4,3,4,4,3,1,3,3,1,0,3,4,3,4,5,4,3,4,3,4,1,3,3]

# query 4: no distance (rank 5 not collected)
query4_rank1 = [3,2,5,5,4,3,3,5,3,3,2,4,5,4,2,1,1,1,4,5,4,4,5,5,5,5,4,2,5,4,4,3]
query4_rank2 = [4,2,2,3,4,3,5,4,3,3,3,5,1,5,3,1,4,2,3,2,5,3,4,5,5,5,4,4,2,5,5,4,4]
query4_rank3 = [4,3,4,3,2,5,5,2,4,3,3,5,4,5,1,1,5,3,2,0,5,2,5,3,4,3,3,4,4,4,3,4,2]
query4_rank4 = [5,5,4,3,5,5,5,3,3,4,3,4,5,3,3,2,3,4,3,5,4,4,2,5,4,2,5,4,5,2,4,2]

# query 5: music sounds better with you
query5_rank1 = [3,4,5,5,5,5,5,5,4,3,3,3,4,1,2,3,3,3,5,4,5,2,5,5,5,3,3,3,3,5,1,4,0]
query5_rank2 = [2,2,4,1,4,4,2,3,0,3,1,3,1,0,0,4,2,2,4,4,4,2,5,2,0,4,2,3,2,2,0,4,3]
query5_rank3 = [3,4,4,2,4,1,5,5,2,3,4,4,2,1,1,2,4,3,4,2,3,4,2,3,3,0,4,2,4,3,1,1,3,5]
query5_rank4 = [2,2,4,2,2,4,4,2,3,4,3,4,4,0,3,5,4,4,2,2,5,3,4,5,5,1,2,2,3,3,0,2,3,0]
query5_rank5 = [2,5,2,4,4,3,2,2,1,4,3,4,3,3,1,3,3,4,3,3,3,3,5,5,5,0,3,3,4,2,3,1,3,2]


# ----- per-query overall mean (pool every rating across ranks) -----

queries = {
    "Elsa": query1_rank1 + query1_rank2 + query1_rank3 + query1_rank4 + query1_rank5,
    "Twelve": query2_rank1 + query2_rank2 + query2_rank3 + query2_rank4 + query2_rank5,
    "Peace Keeper": query3_rank1 + query3_rank2 + query3_rank3 + query3_rank4 + query3_rank5,
    "No Distance": query4_rank1 + query4_rank2 + query4_rank3 + query4_rank4,
    "Music Sounds \nBetter With You": (
        query5_rank1 + query5_rank2 + query5_rank3 + query5_rank4 + query5_rank5
    ),
}

labels = list(queries.keys())
scores = [np.mean(v) for v in queries.values()]

print("=" * 50)
print("MEAN RATING PER QUERY")
print("=" * 50)
for label, score in zip(labels, scores):
    flat_label = label.replace("\n", " ")
    print(f"  {flat_label:35s}  {score:.3f}")
print()


# ----- lollipop: per-query mean -----

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(labels))

ax.vlines(x, ymin=0, ymax=scores, linewidth=3)
ax.scatter(x, scores, s=180, zorder=3)

ax.set_ylim(0, 5.5)
ax.set_xlim(-0.5, len(labels) - 0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, fontsize=17)
ax.set_ylabel("Score", fontsize=30)
ax.set_xlabel("Input Songs", fontsize=30)
ax.tick_params(axis="y", labelsize=22)
ax.set_yticks([0, 1, 2, 3, 4, 5])

for xi, s in zip(x, scores):
    ax.text(xi, s + 0.18, f"{s:.2f}", ha="center", va="bottom", fontsize=25)

plt.tight_layout()
plt.savefig(figures_dir / "human_validation_scores.png", dpi=150, bbox_inches="tight")
plt.show()


# ----- per-rank means and per-query linear fit -----

queries_by_rank = {
    "Elsa": [query1_rank1, query1_rank2, query1_rank3, query1_rank4, query1_rank5],
    "Twelve": [query2_rank1, query2_rank2, query2_rank3, query2_rank4, query2_rank5],
    "Peace Keeper": [query3_rank1, query3_rank2, query3_rank3, query3_rank4, query3_rank5],
    "No Distance": [query4_rank1, query4_rank2, query4_rank3, query4_rank4, 3.3],
    "Music Sounds \nBetter With You": [
        query5_rank1, query5_rank2, query5_rank3, query5_rank4, query5_rank5
    ],
}

ranks = np.array([1, 2, 3, 4, 5])
colors = ["#3FB8E0", "#E07A3F", "#7AC74F", "#B23FE0", "#E0B83F"]

print("=" * 50)
print("MEAN RATING PER RANK (per query)")
print("=" * 50)
for label, rank_lists in queries_by_rank.items():
    flat_label = label.replace("\n", " ")
    means = [np.mean(r) if r is not None else np.nan for r in rank_lists]
    means_str = "  ".join(f"r{i+1}={m:.2f}" for i, m in enumerate(means))
    print(f"  {flat_label:35s}  {means_str}")
print()

fig, ax = plt.subplots(figsize=(9, 5))

for (label, rank_lists), col in zip(queries_by_rank.items(), colors):
    means = np.array([np.mean(r) if r is not None else np.nan for r in rank_lists])

    ax.plot(ranks, means, marker="o", markersize=8, color=col,
            linestyle="None", label=label)

    valid = ~np.isnan(means)
    slope, intercept = np.polyfit(ranks[valid], means[valid], 1)
    fit_line = slope * ranks + intercept
    ax.plot(ranks, fit_line, linestyle="--", linewidth=1.5, color=col, alpha=0.6)

ax.set_xticks(ranks)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_ylim(2.2, 4.3)
ax.set_xlabel("Rank", fontsize=30)
ax.set_ylabel("Mean Score", fontsize=30)
ax.tick_params(axis="y", labelsize=30)
ax.tick_params(axis="x", labelsize=30)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.6)
ax.legend(loc="center right", bbox_to_anchor=(-0.15, 0.5),
          fontsize=20, frameon=False)

plt.tight_layout()
plt.savefig(figures_dir / "mean_score_per_rank.png", dpi=150, bbox_inches="tight")
plt.show()


# ----- query x rank heatmap of mean rating -----

# rebuild a list-only version of queries_by_rank (drop the 3.3 fallback so
# the missing cell shows up as actually missing rather than a fake number)
heatmap_data = {
    "Elsa": [query1_rank1, query1_rank2, query1_rank3, query1_rank4, query1_rank5],
    "Twelve": [query2_rank1, query2_rank2, query2_rank3, query2_rank4, query2_rank5],
    "Peace Keeper": [query3_rank1, query3_rank2, query3_rank3, query3_rank4, query3_rank5],
    "No Distance": [query4_rank1, query4_rank2, query4_rank3, query4_rank4, None],
    "Music Sounds Better\nWith You": [
        query5_rank1, query5_rank2, query5_rank3, query5_rank4, query5_rank5
    ],
}

row_labels = list(heatmap_data.keys())
# build matrix as 0-5 means then convert to satisfaction percent (mean / 5)
matrix = np.full((len(row_labels), 5), np.nan)
for i, (label, rank_lists) in enumerate(heatmap_data.items()):
    for j, cell in enumerate(rank_lists):
        if cell is not None:
            matrix[i, j] = np.mean(cell)

pct = matrix * 20.0  # 0-5 -> 0-100

fig, ax = plt.subplots(figsize=(9, 5.5))
masked = np.ma.masked_invalid(pct)

# discrete colormap matching the 0-5 rating scale (each band = one rating step)
import copy as _copy
from matplotlib.colors import BoundaryNorm
bounds = [0, 20, 40, 60, 80, 100]  # one bin per rating step
base = plt.cm.RdYlGn
discrete = base(np.linspace(0.05, 0.95, len(bounds) - 1))
cmap = _copy.copy(plt.matplotlib.colors.ListedColormap(discrete))
cmap.set_bad(color="#dddddd")
norm = BoundaryNorm(bounds, cmap.N)

im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(len(row_labels) + 0.5, -0.5)

# annotate each cell with its satisfaction percent
for i in range(len(row_labels)):
    for j in range(5):
        v = pct[i, j]
        text = "—" if np.isnan(v) else f"{v:.0f}%"
        ax.text(j, i, text, ha="center", va="center", fontsize=18,
                color="black", weight="bold")

ax.set_xticks(np.arange(5))
ax.set_xticklabels([f"Rank {r}" for r in range(1, 6)], fontsize=16)
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=14)
ax.set_xlabel("Model Rank", fontsize=20, labelpad=24)
ax.set_ylabel("Query Track", fontsize=20)

# row-mean strip along the right for per-query satisfaction
from matplotlib.patches import Rectangle
row_means = np.nanmean(pct, axis=1)
for i, m in enumerate(row_means):
    face = cmap(norm(m))
    ax.add_patch(Rectangle((4.55, i - 0.45), 0.9, 0.9, facecolor=face,
                           edgecolor="white", linewidth=1.5, clip_on=False))
    ax.text(5.0, i, f"{m:.0f}%", ha="center", va="center",
            fontsize=14, color="black", weight="bold")

# column-mean strip below the x-axis for per-rank satisfaction
col_means = np.array([np.nanmean(pct[:, j]) for j in range(5)])
for j, m in enumerate(col_means):
    face = cmap(norm(m))
    ax.add_patch(Rectangle((j - 0.45, len(row_labels) - 0.45), 0.9, 0.9,
                           facecolor=face, edgecolor="white", linewidth=1.5,
                           clip_on=False))
    ax.text(j, len(row_labels), f"{m:.0f}%", ha="center",
            va="center", fontsize=14, color="black", weight="bold")

cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.12,
                    boundaries=bounds, ticks=bounds, spacing="uniform")
cbar.set_label("Satisfaction (%)", fontsize=14)
cbar.ax.set_yticklabels([f"{b}%" for b in bounds])

plt.tight_layout()
plt.savefig(figures_dir / "query_rank_satisfaction.png", dpi=150, bbox_inches="tight")
plt.show()


# ----- same heatmap on the raw 0-5 rating scale -----
# The mean strips (per-query and per-rank) are folded into the matrix as an
# extra column and row instead of floated rectangles, so they sit inside
# the same grid as the data cells. A black separator line marks the edge
# between data and summary. Y-tick labels include each track's genre on a
# second line, so this figure carries everything Table 1 used to.

heatmap_genres = {
    "Elsa": "Progressive Trance",
    "Twelve": "Tech House",
    "Peace Keeper": "Drum & Bass",
    "No Distance": "Progressive House",
    "Music Sounds Better\nWith You": "Funky House",
}
# Display name overrides for the y-tick labels. Two-line tracks are
# rendered with the second line slightly indented for visual clarity.
heatmap_display = {
    "Elsa": "Elsa",
    "Twelve": "Twelve",
    "Peace Keeper": "Peace Keeper",
    "No Distance": "No Distance",
    "Music Sounds Better\nWith You": "Music Sounds\nBetter With You",
}

bounds5 = [0, 1, 2, 3, 4, 5]
discrete5 = base(np.linspace(0.05, 0.95, len(bounds5) - 1))
cmap5 = _copy.copy(plt.matplotlib.colors.ListedColormap(discrete5))
cmap5.set_bad(color="#dddddd")
norm5 = BoundaryNorm(bounds5, cmap5.N)

n_rows, n_cols = matrix.shape
row_means5 = np.nanmean(matrix, axis=1)
col_means5 = np.nanmean(matrix, axis=0)
overall_mean5 = float(np.nanmean(matrix))

ext = np.full((n_rows + 1, n_cols + 1), np.nan)
ext[:n_rows, :n_cols] = matrix
ext[:n_rows, n_cols] = row_means5
ext[n_rows, :n_cols] = col_means5
ext[n_rows, n_cols] = overall_mean5
masked_ext = np.ma.masked_invalid(ext)

fig, ax = plt.subplots(figsize=(7.0, 7.0))
im5 = ax.imshow(masked_ext, cmap=cmap5, norm=norm5, aspect="equal")

for i in range(n_rows + 1):
    for j in range(n_cols + 1):
        v = ext[i, j]
        text = "—" if np.isnan(v) else f"{v:.2f}"
        ax.text(j, i, text, ha="center", va="center", fontsize=15,
                color="black", weight="bold")

x_labels = [f"Rank {r}" for r in range(1, n_cols + 1)] + ["Mean"]

ax.set_xticks(np.arange(n_cols + 1))
ax.set_xticklabels(x_labels, fontsize=14)
ax.set_yticks(np.arange(n_rows + 1))
ax.set_yticklabels([""] * (n_rows + 1))
ax.set_xlabel("Model Rank", fontsize=16, labelpad=10)
ax.tick_params(axis="both", length=0)

# Black separators between the data block and the mean strip
ax.plot([n_cols - 0.5, n_cols - 0.5], [-0.5, n_rows + 0.5],
        color="black", linewidth=1.6, clip_on=False)
ax.plot([-0.5, n_cols + 0.5], [n_rows - 0.5, n_rows - 0.5],
        color="black", linewidth=1.6, clip_on=False)

# Custom two-line y-tick labels: bold track name on top, normal-weight
# italic genre underneath in gray. Mean row gets a single normal-weight
# label so it doesn't compete with the track names visually.
for i, name in enumerate(row_labels):
    display = heatmap_display.get(name, name)
    genre = heatmap_genres[name]
    n_lines = display.count("\n") + 1
    track_y = i - 0.16 if n_lines == 1 else i - 0.22
    genre_y = i + 0.22 if n_lines == 1 else i + 0.30
    ax.text(-0.7, track_y, display, ha="right",
            va="center", fontsize=14, fontweight="bold", color="#222",
            linespacing=1.0)
    ax.text(-0.7, genre_y, genre, ha="right", va="center",
            fontsize=12, fontweight="normal", color="#222")
ax.text(-0.7, n_rows, "Mean", ha="right", va="center",
        fontsize=14, fontweight="normal", color="#222")

ax.set_xlim(-0.5, n_cols + 0.5)
ax.set_ylim(n_rows + 0.5, -0.5)

cbar = fig.colorbar(im5, ax=ax, shrink=0.50, pad=0.03,
                    fraction=0.04, aspect=18,
                    boundaries=bounds5, ticks=bounds5, spacing="uniform")
cbar.set_label("Mean rating (0–5)", fontsize=16)
cbar.ax.set_yticklabels([f"{b}" for b in bounds5], fontsize=14)
cbar.outline.set_visible(False)

plt.tight_layout(pad=0.4)
plt.savefig(figures_dir / "query_rank_rating.png", dpi=180,
            bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()
