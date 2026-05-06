"""
Run the trained baseline model on an arbitrary mp3 (a track that isn't
in our corpus) and return its top-5 next-track recommendations from the
candidate pool the model was trained on.

Pipeline:
  1. extract v2 features for the input mp3 using the same compute_features
     used by extract_features_v2.py
  2. load the trained model and the exact feature column order from
     data/clean/baseline/
  3. build pair vectors [query | cand | |query-cand| | harmonic_bpm]
     against every candidate in audio_features_v2.csv
  4. rank and print top 5 with YouTube links

Usage
    python scripts/recommend_for_audio.py "/home/jr/NYU/Let Love In.mp3"
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from extract_features_v2 import compute_features

FEATURES_PATH = REPO_ROOT / "data" / "clean" / "audio_features_v2.csv"
TRACKS_PATH = REPO_ROOT / "data" / "clean" / "tracks.csv"
OUT_DIR = REPO_ROOT / "data" / "clean" / "baseline"
MODEL_PATH = OUT_DIR / "best_model.joblib"
META_PATH = OUT_DIR / "feature_cols.json"


def yt(tid):
    return f"https://youtube.com/watch?v={tid}"


def fmt(tid, tracks):
    if tid in tracks.index:
        row = tracks.loc[tid]
        artist = row.get("artist") if isinstance(row.get("artist"), str) else "?"
        title = row.get("title") if isinstance(row.get("title"), str) else "?"
        return f"{artist} — {title}"
    return f"[{tid}]"


def harmonic_bpm_distance(t_a, t_b):
    if pd.isna(t_a) or pd.isna(t_b):
        return 0.0
    return min(abs(t_a - t_b), abs(t_a - 2 * t_b), abs(t_a - t_b / 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.audio_path.exists():
        raise SystemExit(f"file not found: {args.audio_path}")

    print(f"loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    print(f"  best model: {meta['best_class']} {meta['best_config']}, "
          f"reported test accuracy {meta['best_test_acc']:.4f}")

    print(f"loading candidate features from {FEATURES_PATH}")
    cand = pd.read_csv(FEATURES_PATH).set_index("track_id")
    cand = cand[feature_cols].dropna(axis=0, how="any")
    print(f"  {len(cand)} candidate tracks, {len(feature_cols)} feature columns")

    tracks = pd.read_csv(TRACKS_PATH).set_index("track_id")

    print(f"\nextracting v2 features for {args.audio_path.name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(args.audio_path, sr=None)
    feats = compute_features(y, sr)
    missing = [c for c in feature_cols if c not in feats]
    if missing:
        raise SystemExit(f"feature schema mismatch: missing {missing[:5]}")
    q_vec = np.array([feats[c] for c in feature_cols], dtype=float)
    print(f"  tempo {feats['tempo']:.1f} bpm, duration {len(y) / sr:.1f}s")

    cand_matrix = cand.values
    cand_ids = cand.index.values
    tempo_idx = feature_cols.index("tempo")

    diff = np.abs(q_vec[None, :] - cand_matrix)
    q_repeat = np.broadcast_to(q_vec, cand_matrix.shape)
    q_tempo = q_vec[tempo_idx]
    harmonic_col = np.array([
        harmonic_bpm_distance(q_tempo, t) for t in cand_matrix[:, tempo_idx]
    ]).reshape(-1, 1)
    pair_matrix = np.concatenate([q_repeat, cand_matrix, diff, harmonic_col], axis=1)

    scores = model.predict_proba(pair_matrix)[:, 1]
    order = np.argsort(-scores)[: args.top_k]

    print()
    print("=" * 60)
    print(f"TOP {args.top_k} RECOMMENDATIONS for {args.audio_path.name}")
    print("=" * 60)
    for rank, idx in enumerate(order, 1):
        tid = cand_ids[idx]
        score = scores[idx]
        cand_tempo = cand_matrix[idx, tempo_idx]
        print(f"{rank}. {fmt(tid, tracks)}")
        print(f"   score {score:.4f}, candidate tempo {cand_tempo:.1f} bpm")
        print(f"   {yt(tid)}")


if __name__ == "__main__":
    main()