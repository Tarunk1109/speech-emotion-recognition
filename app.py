"""Streamlit GUI: upload a speech clip, pick a model, see the predicted emotion."""

import os
import sys
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; the default GUI backend hangs off the main thread
import matplotlib.pyplot as plt
import librosa.display
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import extract_mfcc, extract_mel_spectrogram, load_audio, SAMPLE_RATE  # noqa: E402

MODELS_DIR = "models"
DATA_DIR = "data/processed"
MEL_TIME_STEPS = 130

EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

# Validated categorical palette (dataviz skill, dark-surface steps) — fixed assignment,
# never reshuffled, so each emotion keeps the same color everywhere in the app.
EMOTION_COLOR = {
    "angry": "#3987e5", "calm": "#d95926", "disgust": "#199e70", "fearful": "#c98500",
    "happy": "#d55181", "neutral": "#008300", "sad": "#9085e9", "surprised": "#e66767",
}
EMOJI = {
    "angry": "😠", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprised": "😲",
}
MODEL_FILES = {
    "ANN": ("ann.keras", "MFCC features"),
    "CNN": ("cnn.keras", "Mel-spectrogram"),
    "CNN-Transformer": ("cnn_transformer.keras", "Mel-spectrogram + self-attention"),
}
ENSEMBLE_LABEL = "Ensemble (all 3 averaged)"

INK = "#c3c2b7"
MUTED = "#898781"
SURFACE = "#1a1a19"


def pad_or_truncate(mel: np.ndarray, time_steps: int = MEL_TIME_STEPS) -> np.ndarray:
    if mel.shape[1] < time_steps:
        pad_width = time_steps - mel.shape[1]
        mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=mel.min())
    else:
        mel = mel[:, :time_steps]
    return mel


@st.cache_resource
def get_mfcc_scaler():
    X_mfcc = np.load(os.path.join(DATA_DIR, "X_mfcc.npy"))
    scaler = StandardScaler()
    scaler.fit(X_mfcc)
    return scaler


@st.cache_resource
def get_model(filename: str):
    return load_model(os.path.join(MODELS_DIR, filename))


def predict(model_name: str, audio_path: str):
    filename, _ = MODEL_FILES[model_name]
    if model_name == "ANN":
        mfcc = extract_mfcc(audio_path).reshape(1, -1)
        X = get_mfcc_scaler().transform(mfcc)
    else:
        mel = pad_or_truncate(extract_mel_spectrogram(audio_path))
        X = mel[np.newaxis, ..., np.newaxis]

    model = get_model(filename)
    # model.predict() spins up tf.function tracing machinery that can deadlock when
    # called from a non-main thread (Streamlit runs scripts in a worker thread) —
    # calling the model directly avoids that codepath for a single small batch.
    return np.asarray(model(X, training=False))[0]


def predict_ensemble(model_names, audio_path: str):
    """Averages the probability outputs of multiple models — a quick way to boost accuracy,
    since each model makes different mistakes on the same clip."""
    all_probs = [predict(name, audio_path) for name in model_names]
    return np.mean(all_probs, axis=0)


def render_waveform_and_spectrogram(audio_path: str):
    y = load_audio(audio_path)
    mel_db = extract_mel_spectrogram(audio_path)

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4.2), constrained_layout=True)
    librosa.display.waveshow(y, sr=SAMPLE_RATE, ax=ax1, color="#3987e5")
    ax1.set_title("Waveform", fontsize=10, color=INK)
    ax1.set_xlabel("")

    librosa.display.specshow(mel_db, sr=SAMPLE_RATE, x_axis="time", y_axis="mel", ax=ax2, cmap="magma")
    ax2.set_title("Mel-spectrogram (model input)", fontsize=10, color=INK)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_probability_chart(probs: np.ndarray):
    df = pd.DataFrame({
        "emotion": [e.capitalize() for e in EMOTIONS],
        "probability": probs,
        "color": [EMOTION_COLOR[e] for e in EMOTIONS],
    }).sort_values("probability", ascending=True)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, height=18)
        .encode(
            x=alt.X("probability:Q", axis=alt.Axis(format="%", title=None, gridColor="#2c2c2a", tickColor=MUTED, labelColor=MUTED)),
            y=alt.Y("emotion:N", sort=None, axis=alt.Axis(title=None, labelColor=INK, tickColor=MUTED, domainColor=MUTED)),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("emotion:N", title="Emotion"), alt.Tooltip("probability:Q", title="Probability", format=".1%")],
        )
        .properties(height=220)
        .configure_view(strokeWidth=0)
        .configure(background="transparent")
    )
    st.altair_chart(chart, width="stretch")


st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 2.5rem; max-width: 1000px; }
.ser-title { font-size: 2.1rem; font-weight: 800; margin-bottom: 0; }
.ser-subtitle { color: #898781; font-size: 0.95rem; margin-top: 0.2rem; margin-bottom: 1.6rem; }
.result-card {
  border: 1px solid rgba(255,255,255,0.10); border-radius: 14px; padding: 1.4rem 1.6rem;
  background: rgba(255,255,255,0.03);
}
.result-emoji { font-size: 3rem; line-height: 1; }
.result-label { font-size: 1.6rem; font-weight: 700; margin: 0.3rem 0 0.1rem 0; }
.result-conf { color: #898781; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎙️ SER Project")
    st.caption("Speech Emotion Recognition from Audio Using Deep Learning")
    st.markdown("---")
    st.markdown(
        "**Dataset:** RAVDESS + TESS\n\n"
        "**Emotions:** " + ", ".join(e.capitalize() for e in EMOTIONS) + "\n\n"
        "**Models:** ANN (MFCC baseline), CNN and CNN-Transformer (Mel-spectrograms)"
    )
    st.markdown("---")
    st.caption("Team: Tarun · Harmandeep · Kiranjeet · Jashandeep")

st.markdown('<div class="ser-title">🎙️ Speech Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="ser-subtitle">Upload a short speech clip and a trained model predicts the speaker\'s emotion.</div>', unsafe_allow_html=True)

available = {name: info for name, info in MODEL_FILES.items() if os.path.exists(os.path.join(MODELS_DIR, info[0]))}

if not available:
    st.warning("No trained models found in `models/` yet. Run `python src/train.py` first.")
else:
    model_options = list(available.keys())
    if len(available) > 1:
        model_options = model_options + [ENSEMBLE_LABEL]

    model_choice = st.radio(
        "Model", model_options, horizontal=True,
        format_func=lambda name: (
            f"{name} — {available[name][1]}" if name in available
            else f"{name} — averages all models' predictions"
        ),
    )

    uploaded = st.file_uploader("Upload a .wav file", type=["wav"])

    if uploaded is not None:
        tmp_path = os.path.join("data", "_uploaded_tmp.wav")
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Analyzing audio..."):
            if model_choice == ENSEMBLE_LABEL:
                probs = predict_ensemble(list(available.keys()), tmp_path)
            else:
                probs = predict(model_choice, tmp_path)
            top_idx = int(np.argmax(probs))
            top_emotion = EMOTIONS[top_idx]

        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            st.audio(uploaded)
            render_waveform_and_spectrogram(tmp_path)
        with col2:
            st.markdown(
                f"""<div class="result-card">
                    <div class="result-emoji">{EMOJI[top_emotion]}</div>
                    <div class="result-label" style="color:{EMOTION_COLOR[top_emotion]}">{top_emotion.capitalize()}</div>
                    <div class="result-conf">{probs[top_idx]*100:.1f}% confidence &middot; {model_choice}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.write("")
            render_probability_chart(probs)

        if len(available) > 1:
            with st.expander("Compare across all trained models"):
                rows = []
                all_probs = []
                for name in available:
                    p = predict(name, tmp_path)
                    all_probs.append(p)
                    idx = int(np.argmax(p))
                    rows.append({"Model": name, "Predicted emotion": EMOTIONS[idx].capitalize(), "Confidence": f"{p[idx]*100:.1f}%"})
                ens = np.mean(all_probs, axis=0)
                ens_idx = int(np.argmax(ens))
                rows.append({"Model": ENSEMBLE_LABEL, "Predicted emotion": EMOTIONS[ens_idx].capitalize(), "Confidence": f"{ens[ens_idx]*100:.1f}%"})
                st.table(pd.DataFrame(rows).set_index("Model"))

        os.remove(tmp_path)
    else:
        st.info("👆 Upload a .wav file to get a prediction.")
