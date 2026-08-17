"""Train and evaluate ANN, CNN, and CNN-Transformer models on the cached feature arrays."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

from models import build_ann, build_cnn, build_cnn_transformer
from augment import build_augmented_mel_set

DATA_DIR = "data/processed"
RANDOM_STATE = 42


def load_data():
    X_mfcc = np.load(os.path.join(DATA_DIR, "X_mfcc.npy"))
    X_mel = np.load(os.path.join(DATA_DIR, "X_mel.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"), allow_pickle=True)
    # manifest.csv is in the same row order as the cached feature arrays (build_dataset.py
    # iterates it in order and every file succeeded, so no rows were skipped).
    filepaths = pd.read_csv(os.path.join(DATA_DIR, "manifest.csv"))["filepath"].values
    return X_mfcc, X_mel, y, filepaths


def prepare_labels(y):
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    y_cat = to_categorical(y_int)
    return y_cat, le


def train_and_eval(model, X_train, y_train, X_test, y_test, label_encoder, name: str):
    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[{name}] Test accuracy: {acc:.4f}, Test loss: {loss:.4f}")

    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    os.makedirs("models", exist_ok=True)
    model.save(f"models/{name}.keras")
    return history, acc


def main(augment: bool = True):
    X_mfcc, X_mel, y, filepaths = load_data()
    y_cat, label_encoder = prepare_labels(y)
    num_classes = y_cat.shape[1]
    indices = np.arange(len(y))

    # --- ANN on MFCC ---
    scaler = StandardScaler()
    X_mfcc_scaled = scaler.fit_transform(X_mfcc)
    Xm_train, Xm_test, ya_train, ya_test = train_test_split(
        X_mfcc_scaled, y_cat, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cat
    )
    ann = build_ann(input_dim=X_mfcc.shape[1], num_classes=num_classes)
    train_and_eval(ann, Xm_train, ya_train, Xm_test, ya_test, label_encoder, "ann")

    # --- CNN / Transformer on Mel-spectrograms ---
    idx_train, idx_test = train_test_split(
        indices, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cat
    )
    Xs_train, Xs_test = X_mel[idx_train], X_mel[idx_test]
    ys_train, ys_test = y_cat[idx_train], y_cat[idx_test]

    if augment:
        # Noise + pitch-shift augmentation applied only to the training split (Challenge #3:
        # regularization against overfitting on a moderate-sized dataset). Test data is
        # never augmented, so evaluation stays on clean, unseen samples.
        print("Generating augmented training samples (noise + pitch shift)...")
        X_aug, y_aug = build_augmented_mel_set(filepaths[idx_train].tolist(), ys_train)
        Xs_train = np.concatenate([Xs_train, X_aug], axis=0)
        ys_train = np.concatenate([ys_train, y_aug], axis=0)
        print(f"Training set size after augmentation: {Xs_train.shape[0]} (was {len(idx_train)})")

    input_shape = X_mel.shape[1:]  # (n_mels, time, 1)

    cnn = build_cnn(input_shape=input_shape, num_classes=num_classes)
    train_and_eval(cnn, Xs_train, ys_train, Xs_test, ys_test, label_encoder, "cnn")

    cnn_transformer = build_cnn_transformer(input_shape=input_shape, num_classes=num_classes)
    train_and_eval(cnn_transformer, Xs_train, ys_train, Xs_test, ys_test, label_encoder, "cnn_transformer")


if __name__ == "__main__":
    main()
