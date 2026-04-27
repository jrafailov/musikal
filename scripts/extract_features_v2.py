"""
Richer feature extraction over data/audio/ (Lavi's feature list).

Same parallel + resumable behavior as extract_features_batch.py but
computes a much wider feature vector and summarizes each time-series
matrix as per-row mean and std across time. Writes to a separate
output CSV so the existing audio_features.csv stays untouched.

Resumable: track_ids already in the output or errors CSV are skipped.

Usage
    python scripts/extract_features_v2.py
    python scripts/extract_features_v2.py --jobs 16 --limit 2

Detached run (survives SSH/WSL disconnect)
    nohup /home/jr/miniforge3/envs/musikal/bin/python \
        scripts/extract_features_v2.py --jobs 24 \
        > data/clean/extract_v2.log 2>&1 &
    disown
    tail -f data/clean/extract_v2.log
"""

import argparse
import csv
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import librosa
import numpy as np
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "data" / "audio"
OUT_PATH = REPO_ROOT / "data" / "clean" / "audio_features_v2.csv"
ERR_PATH = REPO_ROOT / "data" / "clean" / "audio_features_v2_errors.csv"

N_MFCC = 13
ERROR_COLS = ["track_id", "error"]


def summarize(mat):
    """mean, std per row of a 1D or 2D feature matrix. Complex inputs use magnitude."""
    arr = np.asarray(mat)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    means = np.mean(arr, axis=1).astype(float).tolist()
    stds = np.std(arr, axis=1).astype(float).tolist()
    return means, stds


def compute_features(y, sr):
    """
    Returns an ordered dict of {column_name: float}.
    Order is fixed so all rows share the same schema.
    """
    feats = {}

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    feats["tempo"] = float(np.asarray(tempo).item())
    feats["n_beats"] = int(len(beat_frames))

    one_d = [
        ("rms", librosa.feature.rms(y=y)),
        ("spectral_flatness", librosa.feature.spectral_flatness(y=y)),
        ("zero_crossing_rate", librosa.feature.zero_crossing_rate(y=y)),
        ("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr)),
        ("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        ("spectral_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr)),
    ]
    for name, vals in one_d:
        means, stds = summarize(vals)
        feats[f"{name}_mean"] = means[0]
        feats[f"{name}_std"] = stds[0]

    two_d = [
        ("chroma_stft", librosa.feature.chroma_stft(y=y, sr=sr)),
        ("chroma_cqt", librosa.feature.chroma_cqt(y=y, sr=sr)),
        ("chroma_cens", librosa.feature.chroma_cens(y=y, sr=sr)),
        ("chroma_vqt", librosa.feature.chroma_vqt(y=y, sr=sr, intervals="ji5")),
        ("melspectrogram", librosa.feature.melspectrogram(y=y, sr=sr)),
        ("mfcc", librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)),
        ("spectral_contrast", librosa.feature.spectral_contrast(y=y, sr=sr)),
        ("poly_features", librosa.feature.poly_features(y=y, sr=sr)),
        ("tonnetz", librosa.feature.tonnetz(y=y, sr=sr)),
        ("tempogram", librosa.feature.tempogram(y=y, sr=sr)),
        ("fourier_tempogram", librosa.feature.fourier_tempogram(y=y, sr=sr)),
        ("tempogram_ratio", librosa.feature.tempogram_ratio(y=y, sr=sr)),
    ]
    for name, mat in two_d:
        means, stds = summarize(mat)
        for k, v in enumerate(means):
            feats[f"{name}_mean_{k}"] = v
        for k, v in enumerate(stds):
            feats[f"{name}_std_{k}"] = v

    return feats


def extract_one(path):
    track_id = path.stem
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path, sr=None)
        feats = compute_features(y, sr)
        return "ok", track_id, feats
    except Exception as e:
        return "err", track_id, f"{type(e).__name__}: {e}"


def load_done_ids(path, key_col):
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or key_col not in header:
            return done
        key_idx = header.index(key_col)
        for row in reader:
            if row:
                done.add(row[key_idx])
    return done


def probe_schema(audio_path):
    """Run one track to determine the feature column order."""
    print(f"probing schema with {audio_path.name}", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(audio_path, sr=None)
    feats = compute_features(y, sr)
    cols = ["track_id"] + list(feats.keys())
    print(f"schema has {len(cols) - 1} feature columns", flush=True)
    return cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    all_files = sorted(AUDIO_DIR.glob("*.mp3"))

    done_ok = load_done_ids(OUT_PATH, "track_id")
    done_err = load_done_ids(ERR_PATH, "track_id")
    done = done_ok | done_err

    pending = [f for f in all_files if f.stem not in done]
    if args.limit:
        pending = pending[: args.limit]

    n_workers = args.jobs if args.jobs else os.cpu_count()
    print(
        f"corpus={len(all_files)}  done_ok={len(done_ok)}  prior_errors={len(done_err)}  "
        f"pending={len(pending)}  workers={n_workers}",
        flush=True,
    )
    if not pending:
        print("nothing to do", flush=True)
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            cols = next(csv.reader(f))
    else:
        cols = probe_schema(pending[0])
        with OUT_PATH.open("w", newline="") as f:
            csv.writer(f).writerow(cols)

    if not ERR_PATH.exists():
        with ERR_PATH.open("w", newline="") as f:
            csv.writer(f).writerow(ERROR_COLS)

    feature_cols = cols[1:]

    with OUT_PATH.open("a", newline="") as fout, ERR_PATH.open("a", newline="") as ferr:
        ok_writer = csv.writer(fout)
        err_writer = csv.writer(ferr)

        n_ok = 0
        n_err = 0
        parallel = Parallel(
            n_jobs=n_workers,
            backend="loky",
            return_as="generator_unordered",
            verbose=0,
        )
        results = parallel(delayed(extract_one)(p) for p in pending)
        for i, result in enumerate(results, 1):
            kind, track_id, payload = result
            if kind == "ok":
                feats = payload
                missing = [c for c in feature_cols if c not in feats]
                if missing:
                    err_writer.writerow([track_id, f"schema_mismatch: missing {missing[:3]}"])
                    n_err += 1
                else:
                    row = [track_id] + [feats[c] for c in feature_cols]
                    ok_writer.writerow(row)
                    n_ok += 1
            else:
                err_writer.writerow([track_id, payload])
                n_err += 1
            fout.flush()
            ferr.flush()
            if i % 10 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)}  ok={n_ok}  err={n_err}", flush=True)

    print(f"done: {n_ok} new rows written, {n_err} new failures", flush=True)


if __name__ == "__main__":
    main()
