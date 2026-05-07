"""
Retrieval evaluation for musikal (DJ track recommendation).

Loads the best model, the held-out test split, and the exact feature
column order from train_baseline.py's outputs in data/clean/baseline/.
This guarantees the retrieval evaluation uses the same trained model,
same pairs, and same feature space as the reported pair-classification
numbers, no RNG drift, no allow-list mismatch, no within-mix leakage.

The evaluation is grouped by query track (track_1). A query's relevant
set is every track_2 that follows it in a real test-split transition,
so a query with two real next-tracks gets credit for finding both.
When ranking candidates we exclude (a) the query itself and (b) any
train-split next-tracks for the same query, so already-seen positives
don't compete with the held-out ones. For each query we record the
rank of every relevant truth and compute Precision@k, Recall@k, Hit@k
for k in {1, 5, 10, 20, 50}, plus MRR using the rank of the first
relevant. Same-artist stratification stays for diagnostics.

Run:
    python scripts/train_baseline.py     # produces the artifacts this script needs
    python scripts/retrieval_eval.py

Override features path if the CSV lives elsewhere:
    FEATURES_PATH=/some/other/path.csv python scripts/retrieval_eval.py

Outputs land in data/clean/baseline/:
    retrieval_per_query.csv   per-query: ranks of each truth, top 5, artist flags
    retrieval_summary.json    aggregate Precision@k, Recall@k, Hit@k, MRR
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
test_positives = all_pairs[(all_pairs["split"] == "test") & (all_pairs["label"] == 1)].reset_index(drop=True)
train_positives = all_pairs[(all_pairs["split"] == "train") & (all_pairs["label"] == 1)].reset_index(drop=True)
print(f"  loaded {len(all_pairs)} pairs, {len(test_positives)} test positives, "
      f"{len(train_positives)} train positives")


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


# ---- step 4, retrieval loop, grouped by query track ----

all_track_ids_arr = features.index.values
all_features_matrix = features.values
n_tracks = len(all_track_ids_arr)
track_id_to_idx = {tid: i for i, tid in enumerate(all_track_ids_arr)}
tempo_idx = feature_cols.index("tempo")


def harmonic_bpm_distance(t_a, t_b):
    if pd.isna(t_a) or pd.isna(t_b):
        return 0.0
    return min(abs(t_a - t_b), abs(t_a - 2 * t_b), abs(t_a - t_b / 2))


# Group test ground-truth by query track. Most queries have one truth; some
# have several when the same track appeared as a transition source in
# multiple test mixes. Train positives for the same query are excluded from
# the candidate pool so already-seen good transitions don't crowd out the
# held-out ones at the top of the ranking.
test_truth_by_query = (
    test_positives.groupby("track_1")["track_2"].apply(set).to_dict()
)
train_pos_by_query = (
    train_positives.groupby("track_1")["track_2"].apply(set).to_dict()
)

# Candidate pool is the test corpus only, every track that shows up in a
# test transition (as query or as truth). Train-only tracks aren't valid
# distractors here since the test set never names them. We still feed all
# 3827 features through the model in one batch then mask the non-test
# tracks before ranking.
test_corpus_track_ids = set(test_positives["track_1"]).union(
    test_positives["track_2"]
)
test_corpus_track_ids = {
    tid for tid in test_corpus_track_ids if tid in track_id_to_idx
}
test_corpus_idxs = np.array(
    [track_id_to_idx[t] for t in test_corpus_track_ids], dtype=int
)
in_test_corpus = np.zeros(n_tracks, dtype=bool)
in_test_corpus[test_corpus_idxs] = True
n_test_corpus = int(in_test_corpus.sum())

K_VALUES = [1, 5, 10, 20, 50]
TOP_N_RECORDED = max(K_VALUES)

print(f"\nCandidate pool: test corpus only, {n_test_corpus} tracks")
print(f"  (filtered from {n_tracks} corpus tracks; train-only tracks dropped)")
print(f"\nRunning retrieval over {len(test_truth_by_query)} unique queries "
      f"against {n_test_corpus} test-corpus candidates")
print(f"  reporting Precision@k / Recall@k / Hit@k for k in {K_VALUES}")

ranks_first_relevant = []
reciprocal_ranks = []
precision_at_k = {k: [] for k in K_VALUES}
recall_at_k = {k: [] for k in K_VALUES}
hit_at_k = {k: [] for k in K_VALUES}
per_query_records = []
# One row per (query, true_next) test pair; rank against the full
# corpus (only the query itself masked) and against the corpus with
# train-split positives also masked. Most reporting is done on the
# masked-train view but the user asked for the raw ranks too.
per_pair_records = []

skipped_q_no_features = 0
skipped_no_truth_in_pool = 0

for q_count, (q_id, truth_set) in enumerate(test_truth_by_query.items()):
    if q_id not in track_id_to_idx:
        skipped_q_no_features += 1
        continue

    truth_set_in_pool = {t for t in truth_set if t in track_id_to_idx}
    if not truth_set_in_pool:
        skipped_no_truth_in_pool += 1
        continue

    q_idx = track_id_to_idx[q_id]
    truth_idxs = {track_id_to_idx[t] for t in truth_set_in_pool}
    n_relevant = len(truth_idxs)

    q_vec = all_features_matrix[q_idx]
    diff_vecs = np.abs(q_vec[None, :] - all_features_matrix)
    q_repeated = np.broadcast_to(q_vec, all_features_matrix.shape)
    q_tempo = q_vec[tempo_idx]
    harmonic_col = np.array([
        harmonic_bpm_distance(q_tempo, b_tempo)
        for b_tempo in all_features_matrix[:, tempo_idx]
    ]).reshape(-1, 1)
    pair_matrix = np.concatenate(
        [q_repeated, all_features_matrix, diff_vecs, harmonic_col], axis=1
    )

    raw_scores = model.predict_proba(pair_matrix)[:, 1]

    # restrict candidate pool to the test corpus, mask everything else.
    # within that pool, drop the query itself.
    base_mask = ~in_test_corpus
    scores_full = raw_scores.copy()
    scores_full[base_mask] = -np.inf
    scores_full[q_idx] = -np.inf
    order_full = np.argsort(-scores_full)
    # only the test-corpus positions are valid ranks
    valid_count_full = int(np.isfinite(scores_full).sum())
    rank_in_full = {idx: r for r, idx in enumerate(order_full[:valid_count_full], start=1)}
    top_score_full = float(scores_full[order_full[0]])

    # extra-masked ranking: also remove train-split positives for this
    # query (so already-seen good transitions don't compete). Skip train
    # positives that also happen to be test truths, since the same DJ
    # pair can appear in multiple mixes.
    train_pos_idxs = [
        track_id_to_idx[t]
        for t in train_pos_by_query.get(q_id, ())
        if t in track_id_to_idx and t not in truth_set_in_pool
    ]
    scores = raw_scores.copy()
    scores[base_mask] = -np.inf
    scores[q_idx] = -np.inf
    for ti in train_pos_idxs:
        scores[ti] = -np.inf
    order = np.argsort(-scores)
    valid_count = int(np.isfinite(scores).sum())
    pos_in_order = {idx: r for r, idx in enumerate(order[:valid_count], start=1)}
    truth_ranks = sorted(pos_in_order[t] for t in truth_idxs)
    first_rank = truth_ranks[0]

    # one record per (query, individual truth) pair
    for t_idx in truth_idxs:
        per_pair_records.append({
            "query_id": q_id,
            "truth_id": all_track_ids_arr[t_idx],
            "score_truth": float(raw_scores[t_idx]),
            "rank_full_corpus": rank_in_full[t_idx],
            "rank_masked_train": pos_in_order[t_idx],
            "top_score_full": top_score_full,
            "n_train_positives_for_query": len(train_pos_idxs),
            "n_relevant_in_test": n_relevant,
            "query_artist": get_artist(q_id),
            "truth_artist": get_artist(all_track_ids_arr[t_idx]),
        })

    ranks_first_relevant.append(first_rank)
    reciprocal_ranks.append(1.0 / first_rank)

    for k in K_VALUES:
        hits_in_top_k = sum(1 for r in truth_ranks if r <= k)
        precision_at_k[k].append(hits_in_top_k / k)
        recall_at_k[k].append(hits_in_top_k / n_relevant)
        hit_at_k[k].append(1.0 if hits_in_top_k > 0 else 0.0)

    top_ids = [all_track_ids_arr[idx] for idx in order[:TOP_N_RECORDED]]
    top5_ids = top_ids[:5]

    query_artist = get_artist(q_id)
    top5_artists = [get_artist(tid) for tid in top5_ids]
    truth_artists = [get_artist(t) for t in truth_set_in_pool]
    truth_same_artist_as_query = (
        query_artist is not None
        and any(a == query_artist for a in truth_artists if a is not None)
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
        "n_relevant": n_relevant,
        "truth_ids": ",".join(map(str, sorted(truth_set_in_pool))),
        "rank_first_relevant": first_rank,
        "ranks_all_relevant": ",".join(map(str, truth_ranks)),
        "reciprocal_rank": 1.0 / first_rank,
        "query_artist": query_artist,
        "truth_same_artist_as_query": truth_same_artist_as_query,
        "top1_same_artist_as_query": top1_same_artist_as_query,
        "top5_same_artist_count": top5_same_artist_count,
        "top1_id": top5_ids[0],
        "top2_id": top5_ids[1],
        "top3_id": top5_ids[2],
        "top4_id": top5_ids[3],
        "top5_id": top5_ids[4],
        "top1_artist": top5_artists[0],
        **{f"precision_at_{k}": precision_at_k[k][-1] for k in K_VALUES},
        **{f"recall_at_{k}": recall_at_k[k][-1] for k in K_VALUES},
    })

    if (q_count + 1) % 100 == 0:
        running_r5 = np.mean(recall_at_k[5])
        print(f"  {q_count + 1} / {len(test_truth_by_query)} queries, "
              f"running Recall@5 {running_r5:.4f}")

if skipped_q_no_features:
    print(f"  skipped {skipped_q_no_features} queries: query track not in feature pool")
if skipped_no_truth_in_pool:
    print(f"  skipped {skipped_no_truth_in_pool} queries: no truth in feature pool")


# ---- step 5, aggregate ----

per_query_df = pd.DataFrame(per_query_records)

mean_precision = {k: float(np.mean(precision_at_k[k])) for k in K_VALUES}
mean_recall = {k: float(np.mean(recall_at_k[k])) for k in K_VALUES}
mean_hit = {k: float(np.mean(hit_at_k[k])) for k in K_VALUES}
mrr = float(np.mean(reciprocal_ranks))
median_rank = int(np.median(ranks_first_relevant))
mean_rank = float(np.mean(ranks_first_relevant))

# Random-pick baseline: a uniformly random ranker over the test corpus
# (excluding the query itself) puts any one relevant item in the top-k
# with probability k / (n_test_corpus - 1).
random_recall = {k: k / max(n_test_corpus - 1, 1) for k in K_VALUES}

print()
print("=" * 60)
print(f"RETRIEVAL METRICS  ({len(per_query_df)} queries vs {n_test_corpus} test-corpus candidates)")
print("=" * 60)
print(f"  {'k':>3}  {'Precision@k':>12}  {'Recall@k':>10}  {'Hit@k':>8}  "
      f"{'random R@k':>10}")
for k in K_VALUES:
    print(f"  {k:>3}  {mean_precision[k]:>12.4f}  {mean_recall[k]:>10.4f}  "
          f"{mean_hit[k]:>8.4f}  {random_recall[k]:>10.5f}")
print()
print(f"  MRR  {mrr:.4f}")
print(f"  rank of first relevant: median {median_rank}, mean {mean_rank:.1f}")


# ---- step 6, same-artist analysis ----
# A query is "same-artist" if at least one of its truths is by the same
# artist as the query track. Hit@k stratified by this flag tells us how
# much retrieval lift comes from artist-style matching vs genuine
# transition logic.

same_artist_q = per_query_df[per_query_df["truth_same_artist_as_query"]]
diff_artist_q = per_query_df[~per_query_df["truth_same_artist_as_query"]]

n_same = len(same_artist_q)
n_diff = len(diff_artist_q)

share_same_artist_truth = n_same / max(len(per_query_df), 1)
share_top1_same_artist = float(per_query_df["top1_same_artist_as_query"].mean())
mean_top5_same_artist = float(per_query_df["top5_same_artist_count"].mean())


def _hit_rate(df, k):
    if len(df) == 0:
        return float("nan")
    return float((df["rank_first_relevant"] <= k).mean())


print()
print("=" * 60)
print("ARTIST ANALYSIS")
print("=" * 60)
print(f"  Query has same-artist truth:    {n_same}/{len(per_query_df)} "
      f"({share_same_artist_truth:.1%})")
print(f"  Top-1 shares artist with query: {share_top1_same_artist:.1%} of queries")
print(f"  Avg same-artist tracks in top 5: {mean_top5_same_artist:.2f} (out of 5)")
print()
print("  Hit@k stratified by whether the query has any same-artist truth:")
print(f"                          same-artist truth   different-artist truth")
print(f"    n queries             {n_same:>17d}   {n_diff:>22d}")
for k in (1, 5, 10, 20, 50):
    print(f"    Hit@{k:<3}              {_hit_rate(same_artist_q, k):>17.4f}   "
          f"{_hit_rate(diff_artist_q, k):>22.4f}")


# ---- step 7, persist ----

per_query_path = OUT_DIR / "retrieval_per_query.csv"
per_query_df.to_csv(per_query_path, index=False)

# Per-pair view, one row per actual (query, true_next) test transition,
# showing the rank of that specific truth in the corpus ranking.
per_pair_df = pd.DataFrame(per_pair_records).sort_values("rank_full_corpus")
per_pair_path = OUT_DIR / "retrieval_per_pair.csv"
per_pair_df.to_csv(per_pair_path, index=False)

# Distribution of the rank of the matched true-next song among the
# test corpus (only the query itself excluded), the closest answer to
# "where does the truth land when we rank the test corpus by posterior".
print()
print("=" * 60)
print(f"RANK OF TRUE NEXT-TRACK IN TEST-CORPUS RANKING  "
      f"({len(per_pair_df)} test pairs vs {n_test_corpus - 1} candidates)")
print("=" * 60)
ranks_full = per_pair_df["rank_full_corpus"].values
ranks_masked = per_pair_df["rank_masked_train"].values
print(f"                              full corpus    train-pos masked")
print(f"  min rank                  {int(np.min(ranks_full)):>13d}    {int(np.min(ranks_masked)):>16d}")
print(f"  25th percentile           {int(np.percentile(ranks_full, 25)):>13d}    {int(np.percentile(ranks_masked, 25)):>16d}")
print(f"  median                    {int(np.median(ranks_full)):>13d}    {int(np.median(ranks_masked)):>16d}")
print(f"  75th percentile           {int(np.percentile(ranks_full, 75)):>13d}    {int(np.percentile(ranks_masked, 75)):>16d}")
print(f"  mean                      {float(np.mean(ranks_full)):>13.1f}    {float(np.mean(ranks_masked)):>16.1f}")
print(f"  max                       {int(np.max(ranks_full)):>13d}    {int(np.max(ranks_masked)):>16d}")
print()
print(f"  share with rank <= k (full corpus / train-pos masked):")
for k in K_VALUES:
    share_f = float((ranks_full <= k).mean())
    share_m = float((ranks_masked <= k).mean())
    print(f"    k={k:<3}                 {share_f:>13.4f}    {share_m:>16.4f}")

summary = {
    "n_queries": int(len(per_query_df)),
    "n_candidates": int(n_test_corpus),
    "n_corpus_total": int(n_tracks),
    "candidate_pool": "test_corpus_only",
    "best_model_class": meta["best_class"],
    "best_model_config": meta["best_config"],
    "pair_test_accuracy": float(meta["best_test_acc"]),
    "k_values": K_VALUES,
    "precision_at_k": {str(k): mean_precision[k] for k in K_VALUES},
    "recall_at_k": {str(k): mean_recall[k] for k in K_VALUES},
    "hit_at_k": {str(k): mean_hit[k] for k in K_VALUES},
    "random_recall_at_k": {str(k): random_recall[k] for k in K_VALUES},
    "mrr": mrr,
    "median_rank_first_relevant": median_rank,
    "mean_rank_first_relevant": mean_rank,
    "n_test_positives": int(len(test_positives)),
    "n_train_positives_excluded": int(len(train_positives)),
    "artist": {
        "share_same_artist_truth": share_same_artist_truth,
        "share_top1_same_artist": share_top1_same_artist,
        "mean_top5_same_artist": mean_top5_same_artist,
        "n_same_artist_queries": int(n_same),
        "n_diff_artist_queries": int(n_diff),
        "same_artist_hit_at_k": {
            str(k): _hit_rate(same_artist_q, k) for k in K_VALUES
        },
        "diff_artist_hit_at_k": {
            str(k): _hit_rate(diff_artist_q, k) for k in K_VALUES
        },
    },
}
summary_path = OUT_DIR / "retrieval_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved per-query results to {per_query_path}")
print(f"Saved per-pair results to {per_pair_path}")
print(f"Saved summary to {summary_path}")


# ---- step 8, print one example for the poster ----

# Pick a query whose first relevant truth landed in top 5, around the median
# of those, so the example isn't cherry-picked too aggressively.
hit_examples = per_query_df[per_query_df["rank_first_relevant"] <= 5].sort_values(
    "rank_first_relevant"
)
if len(hit_examples) > 0:
    example = hit_examples.iloc[len(hit_examples) // 2]
    truth_ids = [int(t) if t.isdigit() else t for t in example["truth_ids"].split(",")]
    print()
    print("=" * 60)
    print("EXAMPLE QUERY (poster artifact)")
    print("=" * 60)
    print(f"\nQuery, currently playing")
    print(f"  {fmt_track(example['query_id'])}")
    print(f"\nReal next tracks from the DJ mix(es) ({example['n_relevant']} truth(s))")
    truth_ranks = [int(r) for r in example["ranks_all_relevant"].split(",")]
    for tid, r in zip(truth_ids, truth_ranks):
        print(f"  {fmt_track(tid)}  (ranked #{r})")
    print(f"  same-artist truth: {bool(example['truth_same_artist_as_query'])}")
    print(f"\nOur top 5 recommendations")
    truth_set = set(truth_ids)
    for k in range(1, 6):
        tid = example[f"top{k}_id"]
        marker = "  <-- match" if tid in truth_set else ""
        print(f"  {k}. {fmt_track(tid)}{marker}")
else:
    print("\nNo queries had a truth in top 5. Look at retrieval_per_query.csv "
          "to see where ranks land.")