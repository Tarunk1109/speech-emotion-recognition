"""Feature extraction: MFCC vectors (for ANN) and Mel-spectrograms (for CNN / Transformer)."""

import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION = 3.0          # seconds, clips padded/truncated to this length
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048


def load_audio(filepath: str, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    y, _ = librosa.load(filepath, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_mfcc(filepath: str) -> np.ndarray:
    """Returns a fixed-length MFCC feature vector (mean over time) for the ANN baseline."""
    y = load_audio(filepath)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return np.mean(mfcc.T, axis=0)  # shape: (N_MFCC,)


def extract_mel_spectrogram(filepath: str) -> np.ndarray:
    """Returns a log-scaled Mel-spectrogram (n_mels, time) for CNN / Transformer input."""
    y = load_audio(filepath)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # shape: (N_MELS, T)


def augment_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def augment_pitch_shift(y: np.ndarray, sr: int = SAMPLE_RATE, n_steps: float = 2.0) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
