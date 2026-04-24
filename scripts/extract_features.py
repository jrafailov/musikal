from pathlib import Path

import librosa
import numpy as np

MUSIC_DIR = Path(__file__).resolve().parents[1] / "data" / "music"

print("starting")

y1, sr1 = librosa.load(MUSIC_DIR / "Music Sounds Better With You.m4a", sr=None)
y2, sr2 = librosa.load(MUSIC_DIR / "Let Love In.mp3", sr=None)
y3, sr3 = librosa.load(MUSIC_DIR / "Lady.mp3", sr=None)
y4, sr4 = librosa.load(MUSIC_DIR / "Viva la vida.mp3", sr=None)

print("loaded")

names = [ "Music Sounds Better With You", "Let Love In", "Lady", "Viva la vida"]
y = [ y1,y2,y3,y4]
sr = [ sr1,sr2,sr3,sr4]

for i in range(len(names)):

    tempo, beat_frames = librosa.beat.beat_track(y=y[i], sr=sr[i])
    rms = librosa.feature.rms(y=y[i]) # RMS Energy (most common) over time
    avg_energy = np.mean(rms) # converting it into an average
    # centroid for "brightness" of track
    centroid = librosa.feature.spectral_centroid(y=y[i], sr=sr[i])
    avg_centroid = np.mean(centroid)
    # mfcc: timbre / "sound identity"
    mfcc = librosa.feature.mfcc(y=y[i], sr=sr[i], n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    print("-----------------------------")
    print("NAME: ", names[i])
    print("tempo: ", tempo)
    print("RMS Energy: ", avg_energy) # RMS is a relative amplitude measure so no units
    print("Number of beats: ", len(beat_frames)) # total number of detected beats in the entire audio files
    print("Average centroid: ", avg_centroid) # evaluates the brightness: similar brightness blend better in transitions: High centroid (for clubbing), low centroid ( mellow music)
    print("MFCC: ", *mfcc_mean)

print("done")
