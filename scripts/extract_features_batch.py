"""
Parallel feature extraction over data/audio/ with incremental CSV writes.

Walks every .mp3 in data/audio/ and emits one feature row per track keyed
on track_id (YouTube ID = filename stem). Each row is flushed to disk as
soon as its worker finishes, so a crash at track 3000 only loses that
one track, not the prior 2999.

Resumable: on startup, track_ids already present in the output CSV (or
the errors CSV) are skipped. Safe to kill and re-run.

Features match the starter script extract_features.py: tempo, n_beats,
rms_mean, centroid_mean, and 13 MFCC means.

Output
    data/clean/audio_features.csv           17 feature columns + track_id
    data/clean/audio_features_errors.csv    track_id, error for failures

Usage
    python scripts/extract_features_batch.py
    python scripts/extract_features_batch.py --jobs 32 --limit 50
"""

import argparse
import csv
import os
import warnings
from pathlib import Path

# Pin BLAS threads so 32 workers don't each spawn 32 BLAS threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import librosa
import numpy as np
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "data" / "audio"
OUT_PATH = REPO_ROOT / "data" / "clean" / "audio_features.csv"
ERR_PATH = REPO_ROOT / "data" / "clean" / "audio_features_errors.csv"

N_MFCC = 13

FEATURE_COLS = (
    ["track_id", "tempo", "rms_mean", "n_beats", "centroid_mean"]
    + [f"mfcc_{k}" for k in range(N_MFCC)]
)
ERROR_COLS = ["track_id", "error"]


def extract_one(path):
    track_id = path.stem
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path, sr=None)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        rms_mean = float(np.mean(librosa.feature.rms(y=y)))
        centroid_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        mfcc_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC), axis=1)

        row = [
            track_id,
            float(np.asarray(tempo).item()),
            rms_mean,
            int(len(beat_frames)),
            centroid_mean,
        ] + [float(x) for x in mfcc_mean]
        return "ok", row
    except Exception as e:
        return "err", [track_id, f"{type(e).__name__}: {e}"]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=None, help="parallel workers (default: all cores)")
    parser.add_argument("--limit", type=int, default=None, help="process only first N pending files (for testing)")
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
        f"pending={len(pending)}  workers={n_workers}"
    )
    if not pending:
        print("nothing to do")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header_out = not OUT_PATH.exists()
    write_header_err = not ERR_PATH.exists()

    with OUT_PATH.open("a", newline="") as fout, ERR_PATH.open("a", newline="") as ferr:
        ok_writer = csv.writer(fout)
        err_writer = csv.writer(ferr)
        if write_header_out:
            ok_writer.writerow(FEATURE_COLS)
        if write_header_err:
            err_writer.writerow(ERROR_COLS)

        n_ok = 0
        n_err = 0
        # loky backend isolates worker crashes (a segfault on one bad mp3
        # only kills that task, not the whole pool). return_as streams
        # results as they finish so we can write per-track.
        parallel = Parallel(
            n_jobs=n_workers,
            backend="loky",
            return_as="generator_unordered",
            verbose=0,
        )
        results = parallel(delayed(extract_one)(p) for p in pending)
        for i, (kind, row) in enumerate(results, 1):
            if kind == "ok":
                ok_writer.writerow(row)
                n_ok += 1
            else:
                err_writer.writerow(row)
                n_err += 1
            fout.flush()
            ferr.flush()
            if i % 25 == 0 or i == len(pending):
                print(f"  {i}/{len(pending)}  ok={n_ok}  err={n_err}", flush=True)

    print(f"done: {n_ok} new rows written, {n_err} new failures")


if __name__ == "__main__":
    main()
