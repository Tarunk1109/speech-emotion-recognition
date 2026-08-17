"""Audio augmentation (noise injection + pitch shifting) applied only to the training
split, to combat overfitting on the moderate-sized dataset without leaking augmented
copies of test samples into training (see proposal Challenge #3)."""

import numpy as np
from tqdm import tqdm

from features import load_audio, extract_mel_spectrogram, augment_noise, augment_pitch_shift, SAMPLE_RATE
from build_dataset import pad_or_truncate


def _mel_from_waveform(y: np.ndarray) -> np.ndarray:
    import librosa
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128, hop_length=512, n_fft=2048)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return pad_or_truncate(mel_db)


def build_augmented_mel_set(filepaths: list, labels: np.ndarray) -> tuple:
    """For each training file, adds a noise-augmented and a pitch-shifted version.
    Returns (X_aug, y_aug) to be concatenated onto the original training arrays."""
    aug_mels, aug_labels = [], []
    for path, label in tqdm(zip(filepaths, labels), total=len(filepaths), desc="Augmenting training set"):
        y = load_audio(path)

        y_noise = augment_noise(y)
        aug_mels.append(_mel_from_waveform(y_noise))
        aug_labels.append(label)

        y_pitch = augment_pitch_shift(y)
        aug_mels.append(_mel_from_waveform(y_pitch))
        aug_labels.append(label)

    X_aug = np.array(aug_mels)[..., np.newaxis]
    y_aug = np.array(aug_labels)
    return X_aug, y_aug
