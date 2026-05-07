"""
Plot Hit@k vs k for the test-corpus retrieval, alongside the random-baseline
chance curve. Hit@k = share of test (query, true_next) pairs whose true_next
ranks in the top k of the model's posterior-ordered list of test-corpus
candidates. The random baseline at k is k / (n_candidates - 1), the chance
that a uniformly random ranker puts the truth in its top k.

Reads
    data/clean/baseline/retrieval_per_pair.csv  (rank_full_corpus per pair)
    data/clean/baseline/retrieval_summary.json  (n_candidates)

Output
    paper/figures/hit_at_k_curve.png

Usage
    python scripts/plot_hit_at_k.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "data" / "clean" / "baseline"
PER_PAIR_PATH = RESULTS_DIR / "retrieval_per_pair.csv"
SUMMARY_PATH = RESULTS_DIR / "retrieval_summary.json"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def pick_sans_font():
    """Prefer Helvetica if installed; fall back to Nimbus Sans (URW's
    metric-identical Helvetica clone), then Liberation/Arial. Returns
    the first one matplotlib can actually render so we don't trigger
    a wall of 'font not found' warnings at runtime."""
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Helvetica", "Helvetica Neue", "Nimbus Sans",
                      "Liberation Sans", "Arial", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "sans-serif"


SANS_FONT = pick_sans_font()
plt.rcParams["font.family"] = SANS_FONT
plt.rcParams["axes.titleweight"] = "regular"
plt.rcParams["axes.labelweight"] = "regular"
plt.rcParams["pdf.fonttype"] = 42  # editable text in PDF
plt.rcParams["ps.fonttype"] = 42

MODEL_COLOR = "#1f5c8b"
RANDOM_COLOR = "#9aa0a6"
HIGHLIGHT_KS = (1, 5, 10, 20, 30, 40, 50)


def main():
    per_pair = pd.read_csv(PER_PAIR_PATH)
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    ranks = per_pair["rank_full_corpus"].values
    n_pairs = len(ranks)
    n_candidates = int(summary["n_candidates"])
    # candidates excluding the query itself, what the model actually ranks
    n_eff_candidates = n_candidates - 1

    ks = np.arange(1, 51)
    hit_curve = np.array([(ranks <= k).mean() for k in ks])
    random_curve = ks / n_eff_candidates

    # 95% CI on the chance baseline only. The chance line is genuinely
    # stochastic on this test set, each invocation of a random ranker
    # produces a different ordering and so a different Hit@k. The Wald
    # interval p ± 1.96 sqrt(p(1-p)/n) is the run-to-run variability of
    # a random ranker on n=n_pairs trials. The model curve is deterministic
    # for this test set so we don't put a band on it.
    z = 1.96
    rand_se = np.sqrt(random_curve * (1 - random_curve) / n_pairs)
    rand_lo = np.clip(random_curve - z * rand_se, 0, 1)
    rand_hi = random_curve + z * rand_se

    # Native size targets a ~0.50 linewidth render in the paper
    # (~3.3 inch wide on NeurIPS body text), so LaTeX downscales by less
    # and the source font sizes survive into the rendered PDF without
    # going tiny.
    fig, ax = plt.subplots(figsize=(3.6, 2.8))

    # gap fill between random expected and model, drawn first so it sits
    # under everything
    ax.fill_between(ks, random_curve, hit_curve,
                    where=hit_curve >= random_curve,
                    color=MODEL_COLOR, alpha=0.08, linewidth=0, zorder=1)

    # model line first so it appears first in the legend, raised zorder
    # keeps it on top of the chance band visually
    ax.plot(ks, hit_curve, color=MODEL_COLOR, linewidth=2.6,
            label="Model", zorder=5)

    # chance baseline + CI band underneath
    ax.fill_between(ks, rand_lo, rand_hi,
                    color=RANDOM_COLOR, alpha=0.25, linewidth=0, zorder=2)
    ax.plot(ks, random_curve, color=RANDOM_COLOR, linewidth=1.8,
            linestyle="--", label="Chance (95% CI shaded)", zorder=3)

    # markers + labels only at the headline k values. The curve rises
    # monotonically so labels sit above-left of each marker, where there's
    # empty space (the curve has already passed through that x to the left).
    # The last point (k=50) has no curve to its right, so put it above-right.
    for k in HIGHLIGHT_KS:
        y = hit_curve[k - 1]
        ax.scatter([k], [y], color=MODEL_COLOR, s=28,
                   edgecolor="white", linewidth=1.0, zorder=6)
        if k == HIGHLIGHT_KS[-1]:
            xt, yt, ha = 5, 4, "left"
        else:
            xt, yt, ha = -5, 5, "right"
        ax.annotate(
            f"{y:.1%}",
            xy=(k, y), xytext=(xt, yt), textcoords="offset points",
            fontsize=8, color=MODEL_COLOR, ha=ha, va="bottom",
            fontweight="bold",
        )

    ax.set_xlabel("k", fontsize=10, color="#222", fontweight="bold")
    ax.set_ylabel("Hit@k", fontsize=10, color="#222", fontweight="bold")

    y_top = max(hit_curve.max(), random_curve.max()) * 1.18
    ax.set_xlim(0, 51)
    ax.set_ylim(0, y_top)
    ax.set_xticks([1, 5, 10, 20, 30, 40, 50])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis="both", labelsize=8, color="#555",
                   labelcolor="#222", length=3, pad=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.grid(True, axis="y", linestyle="-", linewidth=0.5,
            color="#e6e6e6", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#555")
        ax.spines[spine].set_linewidth(0.9)

    leg = ax.legend(loc="upper left", frameon=False, fontsize=8,
                    handlelength=1.6, borderpad=0.15,
                    bbox_to_anchor=(0.01, 0.99))
    for text in leg.get_texts():
        text.set_color("#222")
        text.set_fontweight("bold")

    out_path = FIGURES_DIR / "hit_at_k_curve.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}  (font: {SANS_FONT})")

    # also dump the raw curve as a CSV in case you want it in the report
    curve_df = pd.DataFrame({
        "k": ks,
        "hit_at_k": hit_curve,
        "random_baseline": random_curve,
        "lift_over_random": hit_curve / random_curve,
    })
    curve_path = RESULTS_DIR / "hit_at_k_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    print(f"wrote {curve_path.relative_to(REPO_ROOT)}")

    print()
    print("Headline values:")
    for k in HIGHLIGHT_KS:
        h = hit_curve[k - 1]
        r = random_curve[k - 1]
        print(f"  Hit@{k:<3}  {h:.4f}  vs random {r:.5f}  ({h / r:.1f}x)")


if __name__ == "__main__":
    main()
