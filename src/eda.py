"""Exploratory data analysis: class balance, waveform, and Mel-spectrogram examples per emotion."""

import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

from features import load_audio, extract_mel_spectrogram, SAMPLE_RATE

MANIFEST_PATH = "data/processed/manifest.csv"
REPORT_DIR = "reports"


def plot_class_balance(manifest: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    manifest["emotion"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Sample count per emotion")
    axes[0].set_xlabel("Emotion")
    axes[0].set_ylabel("Count")

    pd.crosstab(manifest["emotion"], manifest["dataset"]).plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Emotion count by source dataset")
    axes[1].set_xlabel("Emotion")

    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "class_balance.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_waveforms_and_spectrograms(manifest: pd.DataFrame, seed: int = 42):
    random.seed(seed)
    emotions = sorted(manifest["emotion"].unique())

    fig, axes = plt.subplots(len(emotions), 2, figsize=(12, 3 * len(emotions)))
    for i, emotion in enumerate(emotions):
        sample_path = manifest[manifest["emotion"] == emotion].sample(1, random_state=seed)["filepath"].iloc[0]
        y = load_audio(sample_path)
        mel_db = extract_mel_spectrogram(sample_path)

        librosa.display.waveshow(y, sr=SAMPLE_RATE, ax=axes[i, 0])
        axes[i, 0].set_title(f"{emotion} — waveform")

        img = librosa.display.specshow(mel_db, sr=SAMPLE_RATE, x_axis="time", y_axis="mel", ax=axes[i, 1])
        axes[i, 1].set_title(f"{emotion} — Mel-spectrogram")

    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "waveform_spectrogram_examples.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def print_duration_stats(manifest: pd.DataFrame):
    durations = []
    sample = manifest.sample(min(200, len(manifest)), random_state=42)
    for path in sample["filepath"]:
        durations.append(librosa.get_duration(path=path))
    durations = np.array(durations)
    print("\nAudio duration stats (seconds), sample of 200 files:")
    print(f"  mean={durations.mean():.2f}  min={durations.min():.2f}  max={durations.max():.2f}  std={durations.std():.2f}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)

    print("Class balance:\n", manifest["emotion"].value_counts())
    print("\nDataset source breakdown:\n", manifest["dataset"].value_counts())

    plot_class_balance(manifest)
    plot_waveforms_and_spectrograms(manifest)
    print_duration_stats(manifest)


if __name__ == "__main__":
    main()
