import librosa
import numpy as np

print("starting")

y1, sr1 = librosa.load("/Users/laviniaelser/Desktop/Music Sounds Better With You.m4a", sr=None)
#y2, sr2 = librosa.load("/Users/laviniaelser/Desktop/Let Love In.mp3", sr=None)
#y3, sr3 = librosa.load("/Users/laviniaelser/Desktop/Lady.mp3", sr=None)
#y4, sr4 = librosa.load("/Users/laviniaelser/Desktop/Viva la vida.mp3", sr=None)

print("loaded")

names = [ "Music Sounds Better With You"]# , "Let Love In", "Lady", "Viva la vida"
y = [ y1] # ,y2,y3,y4
sr = [ sr1] # ,sr2,sr3,sr4

for i in range(len(names)): 

# CHROMA: 
    chroma_stft = librosa.feature.chroma_stft(y=y[i], sr=sr[i]) # Compute a chromagram from a waveform or power spectrogram.
    chroma_cqt  = librosa.feature.chroma_cqt(y=y[i], sr=sr[i]) # Constant-Q chromagram
    chroma_cens = librosa.feature.chroma_cens(y=y[i], sr=sr[i]) # Compute the chroma variant "Chroma Energy Normalized" (CENS)
    chroma_vqt = librosa.feature.chroma_vqt(y=y[i], sr=sr[i], intervals='ji5') # Variable-Q chromagram

# Spectral features
    melspectrogram = librosa.feature.melspectrogram(y=y[i], sr=sr[i]) # Compute a mel-scaled spectrogram.
    # mfcc: timbre / “sound identity”
    mfcc = librosa.feature.mfcc(y=y[i], sr=sr[i], n_mfcc=13) # Mel-frequency cepstral coefficients (MFCCs)

    rms = librosa.feature.rms(y=y[i]) # RMS Energy (most common) over time
    avg_energy = np.mean(rms) # converting it into an average
    # centroid for "brightness" of track
    centroid = librosa.feature.spectral_centroid(y=y[i], sr=sr[i]) # Compute the spectral centroid.
    avg_centroid = np.mean(centroid)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y[i], sr=sr[i]) # Compute p'th-order spectral bandwidth.
    spectral_contrast = librosa.feature.spectral_contrast(y=y[i], sr=sr[i]) # Compute spectral contrast
    spectral_flatness = librosa.feature.spectral_flatness(y=y[i]) # Compute spectral flatness
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y[i], sr=sr[i]) # Compute roll-off frequency.
    poly_features = librosa.feature.poly_features(y=y[i], sr=sr[i]) # Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
    tonnetz = librosa.feature.tonnetz(y=y[i], sr=sr[i]) #Compute the tonal centroid features (tonnetz)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y[i]) # Compute the zero-crossing rate of an audio time series.

# Rhythm features
    tempo, beat_frames = librosa.beat.beat_track(y=y[i], sr=sr[i]) 
    tempogram = librosa.feature.tempogram(y=y[i], sr=sr[i]) # Compute the tempogram: local autocorrelation of the onset strength envelope.
    fourier_tempogram = librosa.feature.fourier_tempogram(y=y[i], sr=sr[i]) # Compute the Fourier tempogram: the short-time Fourier transform of the onset strength envelope.
    tempogram_ratio = librosa.feature.tempogram_ratio(y=y[i], sr=sr[i]) # Tempogram ratio features, also known as spectral rhythm patterns.
    
    vector = [ chroma_stft, chroma_cqt, chroma_cens, chroma_vqt, melspectrogram, mfcc, rms, 
              avg_energy, centroid, avg_centroid,  spectral_bandwidth,spectral_contrast,
               spectral_flatness, spectral_rolloff, poly_features, tonnetz, zero_crossing_rate, 
               tempo, beat_frames, tempogram, fourier_tempogram, tempogram_ratio]
    print("-----------------------------")
    print("NAME: ", names[i])
    print("vector: ", vector)
    

print("done")