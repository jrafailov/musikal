"""
Frame-resolved feature extraction. Preserves the time axis for every
feature so later analyses can slice into specific regions of a track
(intro, drop, outro, last 30s, etc.) without re-running librosa.

One .npz per track at data/clean/feats_v3/<track_id>.npz containing:
    sr, hop_length, n_samples, tempo, beat_frames, onset_env,
    rms, spectral_flatness, zero_crossing_rate,
    spectral_centroid, spectral_bandwidth, spectral_rolloff,
    chroma_stft, chroma_cqt, chroma_cens, chroma_vqt,
    melspectrogram, mfcc, spectral_contrast, poly_features, tonnetz,
    tempogram, fourier_tempogram (magnitude), tempogram_ratio

All matrices are float32 with shape (channels, n_frames). Convert
frame index to seconds via librosa.frames_to_time(idx, sr, hop_length).

Resumable: tracks whose .npz file already exists are skipped. Writes
are atomic (temp file + rename) so a kill mid-write doesn't poison
the resume logic.

Usage
    python scripts/extract_features_v3.py
    python scripts/extract_features_v3.py --jobs 8 --limit 2
    python scripts/extract_features_v3.py --jobs 24 --compress

Detached run
    nohup /home/jr/miniforge3/envs/musikal/bin/python \
        scripts/extract_features_v3.py --jobs 24 \
        > data/clean/extract_v3.log 2>&1 &
    disown
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
OUT_DIR = REPO_ROOT / "data" / "clean" / "feats_v3"
ERR_PATH = REPO_ROOT / "data" / "clean" / "feats_v3_errors.csv"

N_MFCC = 13
HOP_LENGTH = 512
TARGET_SR = 22050
ERROR_COLS = ["track_id", "error"]


def f32(x):
    return np.asarray(x, dtype=np.float32)


def compute_features(y, sr):
    """Return dict of {name: ndarray}, time axis preserved."""
    feats = {}
    feats["sr"] = np.int32(sr)
    feats["hop_length"] = np.int32(HOP_LENGTH)
    feats["n_samples"] = np.int32(len(y))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    feats["tempo"] = np.float32(np.asarray(tempo).item())
    feats["beat_frames"] = np.asarray(beat_frames, dtype=np.int32)
    feats["onset_env"] = f32(librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH))

    feats["rms"] = f32(librosa.feature.rms(y=y, hop_length=HOP_LENGTH))
    feats["spectral_flatness"] = f32(librosa.feature.spectral_flatness(y=y, hop_length=HOP_LENGTH))
    feats["zero_crossing_rate"] = f32(librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH))
    feats["spectral_centroid"] = f32(librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH))
    feats["spectral_bandwidth"] = f32(librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH))
    feats["spectral_rolloff"] = f32(librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH))

    feats["chroma_stft"] = f32(librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH))
    feats["chroma_cqt"] = f32(librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH))
    feats["chroma_cens"] = f32(librosa.feature.chroma_cens(y=y, sr=sr, hop_length=HOP_LENGTH))
    feats["chroma_vqt"] = f32(librosa.feature.chroma_vqt(y=y, sr=sr, intervals="ji5", hop_length=HOP_LENGTH))
    feats["melspectrogram"] = f32(librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP_LENGTH))
    feats["mfcc"] = f32(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH))
    feats["spectral_contrast"] = f32(librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=HOP_LENGTH))
    feats["poly_features"] = f32(librosa.feature.poly_features(y=y, sr=sr, hop_length=HOP_LENGTH))
    feats["tonnetz"] = f32(librosa.feature.tonnetz(y=y, sr=sr))
    feats["tempogram"] = f32(librosa.feature.tempogram(y=y, sr=sr, hop_length=HOP_LENGTH))
    feats["fourier_tempogram"] = f32(np.abs(librosa.feature.fourier_tempogram(y=y, sr=sr, hop_length=HOP_LENGTH)))
    feats["tempogram_ratio"] = f32(librosa.feature.tempogram_ratio(y=y, sr=sr, hop_length=HOP_LENGTH))

    return feats


def extract_one(path, out_dir, compress, target_sr):
    track_id = path.stem
    out_path = out_dir / f"{track_id}.npz"
    tmp_path = out_dir / f".{track_id}.tmp.npz"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(path, sr=target_sr)
        feats = compute_features(y, sr)
        if compress:
            np.savez_compressed(tmp_path, **feats)
        else:
            np.savez(tmp_path, **feats)
        tmp_path.rename(out_path)
        return "ok", track_id, out_path.stat().st_size
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return "err", track_id, f"{type(e).__name__}: {e}"


def load_done_errors(path):
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                done.add(row[0])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compress", action="store_true", help="np.savez_compressed (smaller, no mmap)")
    parser.add_argument("--sr", type=int, default=TARGET_SR, help=f"resample rate (default {TARGET_SR}, pass 0 for native)")
    args = parser.parse_args()
    target_sr = None if args.sr == 0 else args.sr

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_files = sorted(AUDIO_DIR.glob("*.mp3"))

    done_ok = {p.stem for p in OUT_DIR.glob("*.npz")}
    done_err = load_done_errors(ERR_PATH)
    done = done_ok | done_err

    pending = [f for f in all_files if f.stem not in done]
    if args.limit:
        pending = pending[: args.limit]

    n_workers = args.jobs if args.jobs else os.cpu_count()
    print(
        f"corpus={len(all_files)}  done_ok={len(done_ok)}  prior_errors={len(done_err)}  "
        f"pending={len(pending)}  workers={n_workers}  compress={args.compress}  "
        f"sr={target_sr if target_sr else 'native'}",
        flush=True,
    )
    if not pending:
        print("nothing to do", flush=True)
        return

    write_header = not ERR_PATH.exists()
    with ERR_PATH.open("a", newline="") as ferr:
        err_writer = csv.writer(ferr)
        if write_header:
            err_writer.writerow(ERROR_COLS)

        n_ok = 0
        n_err = 0
        total_bytes = 0
        parallel = Parallel(
            n_jobs=n_workers,
            backend="loky",
            return_as="generator_unordered",
            verbose=0,
        )
        results = parallel(
            delayed(extract_one)(p, OUT_DIR, args.compress) for p in pending
        )
        for i, (kind, track_id, payload) in enumerate(results, 1):
            if kind == "ok":
                n_ok += 1
                total_bytes += payload
            else:
                err_writer.writerow([track_id, payload])
                ferr.flush()
                n_err += 1
            if i % 10 == 0 or i == len(pending):
                avg_mb = total_bytes / max(n_ok, 1) / 1024 / 1024
                print(
                    f"  {i}/{len(pending)}  ok={n_ok}  err={n_err}  "
                    f"avg={avg_mb:.1f} MB/track",
                    flush=True,
                )

    print(f"done: {n_ok} new tracks written, {n_err} new failures", flush=True)


if __name__ == "__main__":
    main()
