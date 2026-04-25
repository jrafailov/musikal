from pathlib import Path

import librosa
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MUSIC_DIR = REPO_ROOT / "data" / "music"
OUT_PATH = REPO_ROOT / "data" / "clean" / "audio_features_sample.csv"

N_MFCC = 13

print("starting")

y1, sr1 = librosa.load(MUSIC_DIR / "Music Sounds Better With You.m4a", sr=None)
y2, sr2 = librosa.load(MUSIC_DIR / "Let Love In.mp3", sr=None)
y3, sr3 = librosa.load(MUSIC_DIR / "Lady.mp3", sr=None)
y4, sr4 = librosa.load(MUSIC_DIR / "Viva la vida.mp3", sr=None)

print("loaded")

names = ["Music Sounds Better With You", "Let Love In", "Lady", "Viva la vida"]
y = [y1, y2, y3, y4]
sr = [sr1, sr2, sr3, sr4]

rows = []
for i in range(len(names)):
    tempo, beat_frames = librosa.beat.beat_track(y=y[i], sr=sr[i])
    rms = librosa.feature.rms(y=y[i])
    avg_energy = np.mean(rms)
    centroid = librosa.feature.spectral_centroid(y=y[i], sr=sr[i])
    avg_centroid = np.mean(centroid)
    mfcc = librosa.feature.mfcc(y=y[i], sr=sr[i], n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)

    row = {
        "track_name": names[i],
        "tempo": float(np.asarray(tempo).item()),
        "rms_mean": float(avg_energy),
        "n_beats": int(len(beat_frames)),
        "centroid_mean": float(avg_centroid),
    }
    for k in range(N_MFCC):
        row[f"mfcc_{k}"] = float(mfcc_mean[k])
    rows.append(row)

features = pd.DataFrame(rows)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(OUT_PATH, index=False)

print(f"wrote {len(features)} rows x {features.shape[1] - 1} features to {OUT_PATH}")
print("done")
