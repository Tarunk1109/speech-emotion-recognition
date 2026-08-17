"""Extract MFCC + Mel-spectrogram features for every file in the manifest and cache to disk."""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from features import extract_mfcc, extract_mel_spectrogram

MANIFEST_PATH = "data/processed/manifest.csv"
OUT_DIR = "data/processed"
MEL_TIME_STEPS = 130  # ~3s at sr=22050, hop_length=512 -> pad/truncate for consistent shape


def pad_or_truncate(mel: np.ndarray, time_steps: int = MEL_TIME_STEPS) -> np.ndarray:
    if mel.shape[1] < time_steps:
        pad_width = time_steps - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=mel.min())
    else:
        mel = mel[:, :time_steps]
    return mel


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    mfcc_features, mel_features, labels = [], [], []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting features"):
        try:
            mfcc = extract_mfcc(row["filepath"])
            mel = pad_or_truncate(extract_mel_spectrogram(row["filepath"]))
        except Exception as e:
            print(f"Skipping {row['filepath']}: {e}")
            continue
        mfcc_features.append(mfcc)
        mel_features.append(mel)
        labels.append(row["emotion"])

    X_mfcc = np.array(mfcc_features)
    X_mel = np.array(mel_features)[..., np.newaxis]  # add channel dim
    y = np.array(labels)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X_mfcc.npy"), X_mfcc)
    np.save(os.path.join(OUT_DIR, "X_mel.npy"), X_mel)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)

    print(f"Saved: X_mfcc {X_mfcc.shape}, X_mel {X_mel.shape}, y {y.shape}")


if __name__ == "__main__":
    main()
