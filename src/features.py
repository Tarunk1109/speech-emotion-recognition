"""Feature extraction: MFCC vectors (for ANN) and Mel-spectrograms (for CNN / CNN-Transformer).

This file is the answer to "how does a sound file become numbers a model can learn from?"
Every clip goes through load_audio() first, then one of two paths:
  - extract_mfcc()          -> 40 numbers (one tone summary)      -> used by the ANN
  - extract_mel_spectrogram() -> a 128x130 picture (pitch over time) -> used by the CNN / CNN-Transformer
"""

import numpy as np
import librosa

# These 6 numbers control the whole conversion. Change any one of them and every
# cached feature file becomes out of date (would need to rerun build_dataset.py).
SAMPLE_RATE = 22050     # how many times per second we measure the sound wave
DURATION = 3.0           # every clip is forced to exactly this many seconds
N_MFCC = 40                # how many "tone summary" numbers the ANN gets
N_MELS = 128                # how many pitch buckets the spectrogram is split into
HOP_LENGTH = 512              # how far we slide the analysis window each step
N_FFT = 2048                    # how big each analysis window is


def load_audio(filepath: str, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """Step 1: read the file and make every clip the exact same length.

    Clips shorter than 3 seconds get zero-padded (silence added at the end);
    clips longer than 3 seconds get cut off. This is required because the
    models expect a fixed-size input every time.
    """
    y, _ = librosa.load(filepath, sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y


def extract_mfcc(filepath: str) -> np.ndarray:
    """Step 2a (ANN path): squeeze the whole clip down into 40 numbers.

    librosa.feature.mfcc() does the heavy lifting (FFT + Mel bands + a final
    compression step called a DCT). We then average across the whole clip,
    which is *why* the ANN has no sense of timing — everything gets flattened
    into one summary before the model ever sees it.
    """
    y = load_audio(filepath)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return np.mean(mfcc.T, axis=0)  # shape: (N_MFCC,) -- one 40-number vector per clip


def extract_mel_spectrogram(filepath: str) -> np.ndarray:
    """Step 2b (CNN / CNN-Transformer path): turn the clip into a picture instead.

    Same idea as MFCC, but we *don't* average it away — we keep the full
    128 (pitch) x ~130 (time) grid, so the CNN and CNN-Transformer can see how
    the sound changes across the 3 seconds, not just its overall average.
    """
    y = load_audio(filepath)
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # convert to decibels (log scale, like human hearing)
    return mel_db  # shape: (N_MELS, T) -- a picture, not a single vector


# --- Data augmentation: only used on the training set, to help the CNN / CNN-Transformer
# generalize instead of memorizing the exact 4,240 clips we have. Test clips never touch these.

def augment_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Makes a slightly noisy copy of a clip — like adding quiet background hiss."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def augment_pitch_shift(y: np.ndarray, sr: int = SAMPLE_RATE, n_steps: float = 2.0) -> np.ndarray:
    """Makes a copy of a clip with the pitch shifted slightly higher or lower."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
