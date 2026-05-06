"""
Survey-ready selection for Lavi: 2 queries per genre — one where the
model nailed it (lowest rank_of_truth) and one near the genre's median
rank (a typical case). Each query lists the currently-playing track,
the model's top 5 picks with confidence scores, and the real next track
the DJ actually played, for context.

Output is structured so each query is self-contained and pasteable into
a survey form.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MIXES = REPO_ROOT / "data" / "tracklists" / "djmix_mixes.csv"
PAIRS = REPO_ROOT / "data" / "clean" / "baseline" / "pairs.parquet"
PER_QUERY = REPO_ROOT / "data" / "clean" / "baseline" / "retrieval_per_query.csv"
TRACKS = REPO_ROOT / "data" / "clean" / "tracks.csv"
FEATURES = REPO_ROOT / "data" / "clean" / "audio_features_v2.csv"
MODEL = REPO_ROOT / "data" / "clean" / "baseline" / "best_model.joblib"
META = REPO_ROOT / "data" / "clean" / "baseline" / "feature_cols.json"
OUT = REPO_ROOT / "data" / "clean" / "baseline" / "survey_selection.md"

GENRES = [
    "House",
    "Tech House",
    "Deep House",
    "Techno",
    "Detroit Techno",
    "Minimal",
    "Progressive House",
    "Trance",
    "Progressive Trance",
    "Psytrance",
    "Drum & Bass",
    "Disco",
]


def yt(tid):
    return f"https://youtube.com/watch?v={tid}"


def fmt(tid, tracks):
    if tid in tracks.index:
        row = tracks.loc[tid]
        artist = row.get("artist") if isinstance(row.get("artist"), str) else "?"
        title = row.get("title") if isinstance(row.get("title"), str) else "?"
        return f"{artist} — {title}"
    return f"[{tid}]"


def parse_tags(s):
    if not isinstance(s, str):
        return set()
    out = set()
    for tok in s.split("|"):
        tok = tok.strip()
        if tok.startswith("Category:"):
            out.add(tok[len("Category:"):])
    return out


def harmonic_bpm_distance(t_a, t_b):
    if pd.isna(t_a) or pd.isna(t_b):
        return 0.0
    return min(abs(t_a - t_b), abs(t_a - 2 * t_b), abs(t_a - t_b / 2))


def score_pair(model, feats, feature_cols, tempo_idx, a_id, b_id):
    a = feats.loc[a_id].values
    b = feats.loc[b_id].values
    diff = np.abs(a - b)
    h = np.array([harmonic_bpm_distance(a[tempo_idx], b[tempo_idx])])
    vec = np.concatenate([a, b, diff, h]).reshape(1, -1)
    return float(model.predict_proba(vec)[0, 1])


mixes = pd.read_csv(MIXES)
mixes["tag_set"] = mixes.tags.apply(parse_tags)
mix_titles = mixes.set_index("mix_id")["title"]

pairs = pd.read_parquet(PAIRS)
test_pos = pairs[(pairs.split == "test") & (pairs.label == 1)]

pq = pd.read_csv(PER_QUERY)
test_with_mix = test_pos.merge(
    pq, left_on=["track_1", "track_2"], right_on=["query_id", "truth_id"], how="inner"
)
print(f"loaded {len(test_with_mix)} test queries with retrieval results")

tracks = pd.read_csv(TRACKS).set_index("track_id")

print("loading model and features for scoring")
with open(META) as f:
    meta = json.load(f)
feature_cols = meta["feature_cols"]
model = joblib.load(MODEL)
feats = pd.read_csv(FEATURES).set_index("track_id")[feature_cols].dropna(axis=0, how="any")
tempo_idx = feature_cols.index("tempo")

selected = []
seen_query_ids = set()

for genre in GENRES:
    genre_mix_ids = set(mixes[mixes.tag_set.apply(lambda s: genre in s)].mix_id)
    slice_ = test_with_mix[test_with_mix.mix_id.isin(genre_mix_ids)].copy()
    if slice_.empty:
        print(f"  {genre:25s}: skipped (no queries)")
        continue
    slice_ = slice_.sort_values("rank_of_truth").reset_index(drop=True)

    best = slice_.iloc[0]
    if len(slice_) >= 2:
        median_rank = slice_.rank_of_truth.median()
        typical_idx = (slice_.rank_of_truth - median_rank).abs().idxmin()
        if typical_idx == 0:
            typical_idx = min(1, len(slice_) - 1)
        typical = slice_.iloc[typical_idx]
        picks = [("best", best), ("typical", typical)]
    else:
        picks = [("best", best)]

    for kind, row in picks:
        if row.query_id in seen_query_ids:
            continue
        seen_query_ids.add(row.query_id)
        selected.append((genre, kind, row))

    print(f"  {genre:25s}: {len(slice_):4d} queries, "
          f"picked {' + '.join(k for k, _ in picks)}")

print(f"\ntotal selected: {len(selected)} unique queries")

lines = []
lines.append("# DJ Recommender — Survey Selection")
lines.append("")
lines.append(
    f"{len(selected)} queries from the held-out test split, sampled across "
    f"{len({g for g, _, _ in selected})} genres. For each genre we picked one "
    f"query the model handled well (lowest rank-of-truth) and one near the "
    f"genre's median rank — so the survey sees both wins and typical cases."
)
lines.append("")
lines.append(
    "**Format per query:** the track currently playing, our model's top 5 "
    "next-track picks (with the model's confidence score, 0-1), and at the "
    "bottom for reference the track the DJ actually played next from the source mix."
)
lines.append("")

for i, (genre, kind, row) in enumerate(selected, 1):
    q_id = row.query_id
    truth_id = row.truth_id
    rank = int(row.rank_of_truth)
    mix_id = row.mix_id
    mix_title = mix_titles.get(mix_id, "?")

    lines.append(f"## {i}. {genre} ({kind})")
    lines.append("")
    lines.append(f"**Currently playing:** [{fmt(q_id, tracks)}]({yt(q_id)})")
    lines.append("")
    lines.append("**Our top 5 picks (model confidence 0-1):**")
    for k in range(1, 6):
        tid = row[f"top{k}_id"]
        score = score_pair(model, feats, feature_cols, tempo_idx, q_id, tid)
        lines.append(f"{k}. [{fmt(tid, tracks)}]({yt(tid)}) — {score:.3f}")
    lines.append("")
    truth_score = score_pair(model, feats, feature_cols, tempo_idx, q_id, truth_id)
    rank_note = "✓ in top 5" if rank <= 5 else f"ranked #{rank} of 3827 candidates"
    lines.append(
        f"<sub>Reference — DJ actually played: [{fmt(truth_id, tracks)}]({yt(truth_id)}), "
        f"model score {truth_score:.3f}, {rank_note}. Source mix `{mix_id}` — {mix_title}.</sub>"
    )
    lines.append("")

OUT.write_text("\n".join(lines))
print(f"\nWrote {OUT}")
