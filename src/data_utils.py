"""Build a manifest (filepath, emotion label) for RAVDESS and TESS datasets."""

import os
import re
import pandas as pd

# RAVDESS emotion codes -> label (from filename position 3, e.g. 03-01-05-...)
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# TESS emotion tokens found in folder/file names -> unified label
TESS_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "ps": "surprised",          # "pleasant surprise"
    "surprise": "surprised",
    "sad": "sad",
}


def parse_ravdess(root_dir: str) -> pd.DataFrame:
    """root_dir contains Actor_01..Actor_24 subfolders of .wav files."""
    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".wav"):
                continue
            parts = fname.split("-")
            if len(parts) < 3:
                continue
            code = parts[2]
            label = RAVDESS_EMOTION_MAP.get(code)
            if label is None:
                continue
            rows.append({
                "filepath": os.path.join(dirpath, fname),
                "emotion": label,
                "dataset": "RAVDESS",
            })
    return pd.DataFrame(rows)


def parse_tess(root_dir: str) -> pd.DataFrame:
    """root_dir contains subfolders/files like OAF_angry, YAF_Sad, etc."""
    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".wav"):
                continue
            stem = fname.lower().replace(".wav", "")
            label = None
            for token, mapped in TESS_EMOTION_MAP.items():
                if re.search(rf"(^|_){token}($|_)", stem):
                    label = mapped
                    break
            if label is None:
                continue
            rows.append({
                "filepath": os.path.join(dirpath, fname),
                "emotion": label,
                "dataset": "TESS",
            })
    return pd.DataFrame(rows)


def build_manifest(ravdess_dir: str = None, tess_dir: str = None) -> pd.DataFrame:
    frames = []
    if ravdess_dir and os.path.isdir(ravdess_dir):
        frames.append(parse_ravdess(ravdess_dir))
    if tess_dir and os.path.isdir(tess_dir):
        frames.append(parse_tess(tess_dir))
    if not frames:
        raise ValueError("No valid dataset directories provided.")
    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest.drop_duplicates(subset="filepath").reset_index(drop=True)
    return manifest


if __name__ == "__main__":
    manifest = build_manifest(
        ravdess_dir="data/raw/RAVDESS",
        tess_dir="data/raw/TESS",
    )
    print(manifest["emotion"].value_counts())
    print(manifest["dataset"].value_counts())
    manifest.to_csv("data/processed/manifest.csv", index=False)
    print(f"Saved manifest with {len(manifest)} rows to data/processed/manifest.csv")
