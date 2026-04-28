"""
Retrieval evaluation for musikal (DJ track recommendation).

Loads the best model, the held-out test split, and the exact feature
column order from train_baseline.py's outputs in data/clean/baseline/.
This guarantees the retrieval evaluation uses the same trained model,
same pairs, and same feature space as the reported pair-classification
numbers — no RNG drift, no allow-list mismatch, no within-mix leakage.

For each held-out positive (a real DJ transition where both tracks
have audio features), score the query against every candidate track,
rank, report Hit@1 / Hit@5 / Hit@10 / MRR. Also stratify by whether
the truth and query share an artist, to expose how much of the lift
comes from artist-style matching rather than genuine transition logic.

Run:
    python scripts/train_baseline.py     # produces the artifacts this script needs
    python scripts/retrieval_eval.py

Override features path if the CSV lives elsewhere:
    FEATURES_PATH=/some/other/path.csv python scripts/retrieval_eval.py

Outputs land in data/clean/baseline/:
    retrieval_per_query.csv   per-query ranks, top 5 IDs and artists, same-artist flags
    retrieval_summary.json    aggregate Hit@k, MRR, artist-stratified metrics
"""

import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = Path(os.environ.get(
    "FEATURES_PATH",
    REPO_ROOT / "data" / "clean" / "audio_features_v2.csv",
))
TRACKS_PATH = REPO_ROOT / "data" / "clean" / "tracks.csv"
OUT_DIR = REPO_ROOT / "data" / "clean" / "baseline"

PAIRS_PATH = OUT_DIR / "pairs.parquet"
MODEL_PATH = OUT_DIR / "best_model.joblib"
META_PATH = OUT_DIR / "feature_cols.json"


# ---- step 1, load saved artifacts from train_baseline.py ----

missing = [p for p in (PAIRS_PATH, MODEL_PATH, META_PATH) if not p.exists()]
if missing:
    raise SystemExit(
        "Missing artifact(s) from train_baseline.py:\n"
        + "\n".join(f"  {p}" for p in missing)
        + "\n\nRun: python scripts/train_baseline.py"
    )

with open(META_PATH) as f:
    meta = json.load(f)
feature_cols = meta["feature_cols"]
print(f"Loaded best model: {meta['best_class']} {meta['best_config']}")
print(f"  reported test pair accuracy: {meta['best_test_acc']:.4f}")

model = joblib.load(MODEL_PATH)
all_pairs = pd.read_parquet(PAIRS_PATH)
test_pairs = all_pairs[all_pairs["split"] == "test"].reset_index(drop=True)
test_positives = test_pairs[test_pairs["label"] == 1].reset_index(drop=True)
print(f"  loaded {len(all_pairs)} pairs, {len(test_pairs)} held out, "
      f"{len(test_positives)} held-out positive queries")


# ---- step 2, load features filtered to the same columns the model was trained on ----

print(f"\nLoading features from {FEATURES_PATH}")
features_raw = pd.read_csv(FEATURES_PATH).set_index("track_id")
missing_cols = [c for c in feature_cols if c not in features_raw.columns]
if missing_cols:
    raise SystemExit(
        f"Features file is missing {len(missing_cols)} columns the model expects:\n"
        f"  first few: {missing_cols[:5]}"
    )
features = features_raw[feature_cols].dropna(axis=0, how="any")
print(f"  {len(features)} tracks, {len(feature_cols)} feature columns (allow-listed)")


# ---- step 3, load track metadata for artist lookup and human-readable names ----

print(f"Loading tracks from {TRACKS_PATH}")
tracks = pd.read_csv(TRACKS_PATH).set_index("track_id")
print(f"  {len(tracks)} track metadata rows")


def get_artist(tid):
    if tid in tracks.index:
        a = tracks.loc[tid].get("artist")
        if isinstance(a, str) and a.strip():
            return a.strip().lower()
    return None


def fmt_track(tid):
    if tid in tracks.index:
        row = tracks.loc[tid]
        artist = row["artist"] if isinstance(row.get("artist"), str) else "?"
        title = row["title"] if isinstance(row.get("title"), str) else "?"
        return f"{artist} | {title}"
    return f"[{tid}]"


# ---- step 4, retrieval loop ----

print(f"\nRunning retrieval over {len(test_positives)} queries against "
      f"{len(features)} candidate tracks")

all_track_ids_arr = features.index.values
all_features_matrix = features.values
n_tracks = len(all_track_ids_arr)
track_id_to_idx = {tid: i for i, tid in enumerate(all_track_ids_arr)}

ranks = []
hits_at_1, hits_at_5, hits_at_10 = [], [], []
reciprocal_ranks = []
per_query_records = []

for i, row in test_positives.iterrows():
    q_id = row["track_1"]
    truth_id = row["track_2"]

    if q_id not in track_id_to_idx or truth_id not in track_id_to_idx:
        continue

    q_idx = track_id_to_idx[q_id]
    truth_idx = track_id_to_idx[truth_id]

    q_vec = all_features_matrix[q_idx]
    diff_vecs = np.abs(q_vec[None, :] - all_features_matrix)
    q_repeated = np.broadcast_to(q_vec, all_features_matrix.shape)
    pair_matrix = np.concatenate([q_repeated, all_features_matrix, diff_vecs], axis=1)

    scores = model.predict_proba(pair_matrix)[:, 1]
    scores[q_idx] = -np.inf  # exclude the query itself

    order = np.argsort(-scores)
    rank_of_truth = int(np.where(order == truth_idx)[0][0]) + 1

    ranks.append(rank_of_truth)
    hits_at_1.append(rank_of_truth <= 1)
    hits_at_5.append(rank_of_truth <= 5)
    hits_at_10.append(rank_of_truth <= 10)
    reciprocal_ranks.append(1.0 / rank_of_truth)

    top5 = order[:5]
    top5_ids = [all_track_ids_arr[idx] for idx in top5]

    query_artist = get_artist(q_id)
    truth_artist = get_artist(truth_id)
    top5_artists = [get_artist(tid) for tid in top5_ids]

    truth_same_artist_as_query = (
        query_artist is not None
        and truth_artist is not None
        and query_artist == truth_artist
    )
    top1_same_artist_as_query = (
        query_artist is not None
        and top5_artists[0] is not None
        and top5_artists[0] == query_artist
    )
    top5_same_artist_count = sum(
        1 for a in top5_artists
        if a is not None and query_artist is not None and a == query_artist
    )

    per_query_records.append({
        "query_id": q_id,
        "truth_id": truth_id,
        "query_artist": query_artist,
        "truth_artist": truth_artist,
        "rank_of_truth": rank_of_truth,
        "truth_same_artist_as_query": truth_same_artist_as_query,
        "top1_same_artist_as_query": top1_same_artist_as_query,
        "top5_same_artist_count": top5_same_artist_count,
        "top1_id": top5_ids[0],
        "top2_id": top5_ids[1],
        "top3_id": top5_ids[2],
        "top4_id": top5_ids[3],
        "top5_id": top5_ids[4],
        "top1_artist": top5_artists[0],
    })

    if (i + 1) % 100 == 0:
        running_hit5 = np.mean(hits_at_5)
        print(f"  {i + 1} / {len(test_positives)} queries, running Hit@5 {running_hit5:.4f}")


# ---- step 5, aggregate ----

per_query_df = pd.DataFrame(per_query_records)

hit1 = float(np.mean(hits_at_1))
hit5 = float(np.mean(hits_at_5))
hit10 = float(np.mean(hits_at_10))
mrr = float(np.mean(reciprocal_ranks))

random_hit1 = 1 / n_tracks
random_hit5 = 5 / n_tracks
random_hit10 = 10 / n_tracks

print()
print("=" * 60)
print(f"RETRIEVAL METRICS  ({len(per_query_df)} queries vs {n_tracks} candidates)")
print("=" * 60)
print(f"  Hit@1   {hit1:.4f}   (random baseline {random_hit1:.5f})")
print(f"  Hit@5   {hit5:.4f}   (random baseline {random_hit5:.5f})")
print(f"  Hit@10  {hit10:.4f}   (random baseline {random_hit10:.5f})")
print(f"  MRR     {mrr:.4f}")
print(f"  median rank {int(np.median(ranks))}, mean rank {np.mean(ranks):.1f}")


# ---- step 6, same-artist analysis ----

same_artist_truth = per_query_df[per_query_df["truth_same_artist_as_query"]]
diff_artist_truth = per_query_df[~per_query_df["truth_same_artist_as_query"]]

n_same = len(same_artist_truth)
n_diff = len(diff_artist_truth)

share_same_artist_truth = n_same / max(len(per_query_df), 1)
share_top1_same_artist = float(per_query_df["top1_same_artist_as_query"].mean())
mean_top5_same_artist = float(per_query_df["top5_same_artist_count"].mean())

def _hit_rate(df, k):
    if len(df) == 0:
        return float("nan")
    return float((df["rank_of_truth"] <= k).mean())

print()
print("=" * 60)
print("ARTIST ANALYSIS")
print("=" * 60)
print(f"  Truth shares artist with query: {n_same}/{len(per_query_df)} "
      f"({share_same_artist_truth:.1%})")
print(f"  Top-1 shares artist with query: {share_top1_same_artist:.1%} of queries")
print(f"  Avg same-artist tracks in top 5: {mean_top5_same_artist:.2f} (out of 5)")
print()
print("  Hit@k stratified by whether truth shares artist with query:")
print(f"                                    same-artist truth   different-artist truth")
print(f"    n queries                       {n_same:>17d}   {n_diff:>22d}")
print(f"    Hit@1                           {_hit_rate(same_artist_truth, 1):>17.4f}   "
      f"{_hit_rate(diff_artist_truth, 1):>22.4f}")
print(f"    Hit@5                           {_hit_rate(same_artist_truth, 5):>17.4f}   "
      f"{_hit_rate(diff_artist_truth, 5):>22.4f}")
print(f"    Hit@10                          {_hit_rate(same_artist_truth, 10):>17.4f}   "
      f"{_hit_rate(diff_artist_truth, 10):>22.4f}")


# ---- step 7, persist ----

per_query_path = OUT_DIR / "retrieval_per_query.csv"
per_query_df.to_csv(per_query_path, index=False)

summary = {
    "n_queries": int(len(per_query_df)),
    "n_candidates": int(n_tracks),
    "best_model_class": meta["best_class"],
    "best_model_config": meta["best_config"],
    "pair_test_accuracy": float(meta["best_test_acc"]),
    "hit_at_1": hit1,
    "hit_at_5": hit5,
    "hit_at_10": hit10,
    "mrr": mrr,
    "median_rank": int(np.median(ranks)),
    "mean_rank": float(np.mean(ranks)),
    "random_hit_at_1": random_hit1,
    "random_hit_at_5": random_hit5,
    "random_hit_at_10": random_hit10,
    "artist": {
        "share_same_artist_truth": share_same_artist_truth,
        "share_top1_same_artist": share_top1_same_artist,
        "mean_top5_same_artist": mean_top5_same_artist,
        "n_same_artist_truth": int(n_same),
        "n_diff_artist_truth": int(n_diff),
        "same_artist_hit_at_1": _hit_rate(same_artist_truth, 1),
        "same_artist_hit_at_5": _hit_rate(same_artist_truth, 5),
        "same_artist_hit_at_10": _hit_rate(same_artist_truth, 10),
        "diff_artist_hit_at_1": _hit_rate(diff_artist_truth, 1),
        "diff_artist_hit_at_5": _hit_rate(diff_artist_truth, 5),
        "diff_artist_hit_at_10": _hit_rate(diff_artist_truth, 10),
    },
}
summary_path = OUT_DIR / "retrieval_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved per-query results to {per_query_path}")
print(f"Saved summary to {summary_path}")


# ---- step 8, print one example for the poster ----

# Pick a query whose truth landed in top 5, around the median of those, so
# the example isn't cherry-picked too aggressively.
hit_examples = per_query_df[per_query_df["rank_of_truth"] <= 5].sort_values("rank_of_truth")
if len(hit_examples) > 0:
    example = hit_examples.iloc[len(hit_examples) // 2]
    print()
    print("=" * 60)
    print("EXAMPLE QUERY (poster artifact)")
    print("=" * 60)
    print(f"\nQuery, currently playing")
    print(f"  {fmt_track(example['query_id'])}")
    print(f"\nReal next track from the DJ mix")
    print(f"  {fmt_track(example['truth_id'])}  "
          f"(ranked #{int(example['rank_of_truth'])} in our recommendations)")
    print(f"  same-artist as query: {bool(example['truth_same_artist_as_query'])}")
    print(f"\nOur top 5 recommendations")
    for k in range(1, 6):
        tid = example[f"top{k}_id"]
        marker = "  <-- match" if tid == example["truth_id"] else ""
        print(f"  {k}. {fmt_track(tid)}{marker}")
else:
    print("\nNo queries had truth in top 5. Look at retrieval_per_query.csv to see where ranks land.")