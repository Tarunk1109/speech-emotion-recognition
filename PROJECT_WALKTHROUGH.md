# Speech Emotion Recognition — Project Walkthrough

**Deep Learning Project — Humber Polytechnic**
Team: Tarun Karnati · Harmandeep Kaur · Kiranjeet Kaur Deol · Jashandeep Kaur

---

## 1. What we built

We trained and compared three deep learning models that each take a short speech clip and classify
it into one of eight emotions: **angry, calm, disgust, fearful, happy, neutral, sad, surprised**.

Instead of hand-crafting acoustic rules, each model learns the patterns directly from audio features:

| Model | Input | Architecture | Test accuracy |
|---|---|---|---|
| **ANN** (baseline) | MFCC feature vector | Dense 256 → 128 → 64 | **88.7%** |
| **CNN** | Mel-spectrogram image | 3× Conv2D + MaxPool + Dropout → Dense 128 | 88.6% |
| **CNN-Transformer** | Mel-spectrogram image | 2× Conv2D → self-attention encoder ×2 → Dense 64 | 87.3% |

All three were evaluated on the same held-out test set of 848 clips (20% split, stratified by
emotion), so the comparison is apples-to-apples.

**All three land within ~1.4 points of each other** — the honest takeaway, and a good talking
point for the presentation. The CNN-Transformer's self-attention lets the model directly compare
*every* time step in the clip against *every other* time step in one shot (the same mechanism
behind modern language models, applied here to audio frames instead of words), which is a more
powerful mechanism in principle than the CNN's local filters. In practice it didn't come out
ahead here — attention-based models are typically data-hungry, and our ~4,240-clip dataset is
modest by the standards Transformer architectures were designed for (matches what we expected
going in: Singh et al., 2023 note attention models need more data to show a clear edge over
CNN/RNN baselines on SER-sized datasets).

---

## 2. Dataset

| Attribute | Details |
|---|---|
| Dataset | RAVDESS + TESS |
| Source | Kaggle (public) |
| Total samples | 4,240 audio clips |
| Emotions | Neutral, Calm, Happy, Sad, Angry, Fearful, Disgusted, Surprised |
| RAVDESS | 1,440 files — 24 actors (12 male, 12 female) |
| TESS | 2,800 files — 2 actresses, 7 emotions (no "calm") |
| Original format | .wav, 48kHz |
| Model input | Resampled to 22.05kHz, 3-second clips, padded/truncated |
| ANN input | 40-coefficient MFCC vector (mean over time) |
| CNN / CNN-Transformer input | 128-band Mel-spectrogram, log-scaled (dB) |

---

## 3. From raw audio to model input — the exact conversion

This is the step nothing above explains: **how does a `.wav` file actually become 40 MFCC numbers
or a 128×130 spectrogram grid?** This all happens in `src/features.py`, using these fixed settings:

| Setting | Value | What it means |
|---|---|---|
| Sample rate | 22,050 Hz | The clip is (re)sampled to 22,050 measurements of air pressure per second |
| Duration | 3.0 seconds | Every clip is forced to exactly 66,150 samples — shorter clips are zero-padded at the end, longer ones are cut off |
| Window size (`n_fft`) | 2048 samples (~93ms) | Audio is analyzed in small overlapping chunks, not all at once |
| Hop length | 512 samples (~23ms) | How far the window slides forward each step — this overlap is what produces ~130 time steps across 3 seconds |
| Mel bands (`n_mels`) | 128 | Number of pitch buckets the frequency axis is divided into |
| MFCC count (`n_mfcc`) | 40 | Number of "timbre summary" numbers extracted per window |

### Step by step

1. **Load & standardize length.** `librosa.load()` reads the `.wav` and resamples it to 22,050 Hz.
   `load_audio()` then pads or truncates it to exactly 3 seconds, so every clip — regardless of its
   original length — becomes the same fixed-size array of 66,150 numbers (raw air-pressure values
   over time).

2. **Slide a window across the clip and run an FFT.** The 66,150-number waveform is chopped into
   overlapping 2048-sample windows, stepping forward 512 samples each time (~130 windows fit across
   3 seconds). For **each window**, a Fast Fourier Transform (FFT) decomposes that tiny slice of
   sound into *how much energy is present at each frequency* — the same idea as splitting a chord
   into its individual notes. This turns "a wiggly waveform over time" into "energy-per-frequency,
   repeated for every time window."

3. **Group frequencies into Mel bands (→ the spectrogram).** Human hearing doesn't perceive pitch
   linearly — we're far more sensitive to differences in low pitches than high ones. The **Mel
   scale** groups the raw FFT frequencies into 128 bands that mirror this, giving finer resolution
   where the ear actually notices more. Stacking one column per time window produces the
   **128 (Mel bands) × ~130 (time steps)** grid — apply `librosa.power_to_db()` (converts to
   decibels, a log scale, because loudness perception is logarithmic too) and that grid **is** the
   Mel-spectrogram — exactly what the CNN and CNN-Transformer take as input, reshaped to
   `128 × 130 × 1` (the trailing 1 is a "channel" dimension, like a grayscale image).

4. **Compress each column further, then average over time (→ MFCC).** For the ANN, we go one step
   further: each time window's Mel-spectrum is compressed with a Discrete Cosine Transform (DCT)
   into just 40 numbers that capture the overall *shape* of the spectrum (the timbre/formants) while
   throwing away fine pitch detail — these are the MFCCs. We then **average all ~130 time windows
   into one single 40-number vector** (`np.mean(mfcc.T, axis=0)` in `features.py`). This averaging
   step is exactly why the ANN has no sense of timing — all 130 time windows get collapsed into one
   flat summary before the network ever sees it.

**In short:**
```
raw .wav  →  resample to 22,050 Hz, pad/truncate to 3s (66,150 samples)
          →  slide a 2048-sample window, step 512 samples at a time, FFT each window
          →  group into 128 Mel bands, convert to dB   →  128 × 130 spectrogram  (CNN / CNN-Transformer input)
          →  compress each window to 40 MFCCs, average across all ~130 windows →  40-number vector (ANN input)
```

---

## 4. Inside each model — inputs, layers, and how they process the data

### 3.1 ANN (baseline) — reads a "summary" of the voice's tone

**Input:** one clip → **40 numbers** (MFCC coefficients, averaged over the whole 3 seconds).
MFCCs describe the overall *timbre* of a sound — roughly, "how bright/dark, sharp/muffled" it is —
the same kind of feature older speech-recognition systems were built on. Averaging over time means
the ANN has **no idea what order things happened in**, only the general tonal fingerprint of the
clip.

| Layer | What it is | What it's doing |
|---|---|---|
| Input | 40 numbers | The MFCC vector for one clip, standardized (zero mean, unit variance) |
| Dense(256) + ReLU | fully-connected layer | Every one of the 40 numbers is combined (weighted sum) into 256 new numbers; ReLU zeroes out negative values so the network can build non-linear patterns |
| Dropout(0.3) | regularization | Randomly "turns off" 30% of neurons *during training only*, forcing the network not to over-rely on any single neuron |
| Dense(128) + ReLU | fully-connected | Same idea, compresses 256 → 128 |
| Dropout(0.3) | regularization | Same as above |
| Dense(64) + ReLU | fully-connected | Compresses 128 → 64 |
| Dense(8) + Softmax | output layer | Turns the final 64 numbers into 8 probabilities that sum to 100% — one per emotion |

**In one sentence:** the ANN is a chain of "combine and simplify" steps that gradually turns 40 tone
numbers into 8 emotion probabilities — no image, no sense of time, just overall tone.

---

### 3.2 CNN — reads the spectrogram like a picture

**Input:** one clip → a **128 × 130 × 1 image**. That's 128 Mel-frequency bands (pitch, low to
high) × 130 time steps (left to right across the 3 seconds) × 1 channel (like a grayscale image —
brightness = loudness at that pitch/time).

| Layer | What it is | What it's doing |
|---|---|---|
| Conv2D(32, 3×3) | convolution | Slides 32 different tiny 3×3 "detectors" across the image; each learns to fire on a specific local pattern (e.g. a burst of high-pitch energy) |
| BatchNorm | normalization | Rescales activations so training stays stable and fast |
| MaxPool(2×2) | downsampling | Shrinks the image by half, keeping only the strongest signal in each 2×2 block — makes the network less sensitive to tiny shifts |
| Dropout(0.25) | regularization | Randomly drops 25% of values during training |
| Conv2D(64, 3×3) → BatchNorm → MaxPool → Dropout | *(same pattern, doubled filters)* | Detects more complex combinations of the first layer's patterns, on a now-smaller image |
| Conv2D(128, 3×3) → BatchNorm → MaxPool → Dropout | *(same pattern again)* | Detects even higher-level patterns |
| Flatten | reshape | Turns the final small stack of feature maps into one long list of numbers |
| Dense(128) + ReLU, Dropout(0.4) | fully-connected | Combines all the detected patterns into a final feature summary |
| Dense(8) + Softmax | output layer | 8 emotion probabilities |

**In one sentence:** the CNN treats the spectrogram exactly like a photo, sliding filters over it
three times to build up from "small local textures" to "whole-picture patterns" — the same trick
used to tell cats from dogs in image classifiers.

---

### 3.3 CNN-Transformer — reads the spectrogram *and* attends across the whole clip at once

**Input:** the same 128 × 130 × 1 Mel-spectrogram as the CNN.

| Layer | What it is | What it's doing |
|---|---|---|
| Conv2D(32) → BatchNorm → MaxPool → Dropout | convolution | Same local-pattern detection as the CNN, but only **2 rounds** instead of 3 — kept shallower on purpose, to preserve more time resolution for the attention layers |
| Conv2D(64) → BatchNorm → MaxPool → Dropout | convolution | Second round of pattern detection |
| Permute + Reshape + Dense(128) | reorganizing | Turns the image-shaped output into a **sequence of 32 tokens** (one per remaining time step), each projected to a 128-number embedding — same idea as turning words into embeddings before a language model reads them |
| + Positional encoding | fixed sin/cos pattern | Self-attention has no built-in sense of order, so a fixed wave pattern is added to each token telling the model *where in time* it sits |
| Self-attention block ×2 (Multi-Head Attention + residual, Feed-Forward + residual) | Transformer encoder | Each of the 32 time-tokens looks at **every other token in the clip simultaneously** and learns how much to "attend to" each one — e.g. the token at the moment of a pitch spike can directly pull in context from a token 2 seconds later, no step-by-step relay needed |
| GlobalAveragePooling1D | reshape | Averages all 32 attended tokens into one 128-number summary of the whole clip |
| Dense(64) + ReLU, Dropout(0.3) | fully-connected | Combines the summary into a final feature vector |
| Dense(8) + Softmax | output layer | 8 emotion probabilities |

**In one sentence:** it's the CNN's pattern-spotting *plus* self-attention that lets every moment of
the clip directly compare itself to every other moment in one step — a more powerful mechanism in
principle than a step-by-step relay (like an LSTM), though on our dataset size it landed within a
hair of the simpler CNN and ANN rather than clearly ahead of them (see Results, section 7).

---

### 3.4 How "training" actually works (all three models)

- **Loss function — categorical cross-entropy:** after each prediction, this measures *how wrong*
  the 8 predicted probabilities were compared to the true label. Training is just: keep adjusting
  the network to make this number smaller.
- **Optimizer — Adam:** the algorithm that nudges every internal weight in the network, a small step
  at a time, in the direction that would have made the last batch of predictions more correct.
- **Batches & epochs:** the model doesn't see all 4,240 clips at once — it processes them in batches
  of 32, adjusting weights after each batch. One full pass through all the training data = 1 epoch.
- **Early stopping (patience = 8):** after each epoch we check accuracy on a held-out validation
  slice. If it hasn't improved in 8 epochs straight, training stops and we keep the *best* version
  seen — not the last one. This is what prevents the model from over-training on the training data.
- **Data augmentation (train set only):** to fight overfitting on a moderate-sized dataset (~4,240
  clips), we generate extra training copies with noise added or pitch shifted slightly. The test set
  is never augmented, so evaluation always happens on clean, unseen audio.

---

## 5. The pipeline — how it actually runs

There is **no single notebook** here (unlike Colab). The project is five separate Python scripts,
run one at a time from the terminal. Each script reads the *files* the previous one produced, does
its job, and writes new files for the next step to read.

**Colab comparison:** in a notebook, a cell's print/plot output stays visible right below it
forever. Here, each script's output is either printed to the terminal as it runs (progress bars,
accuracy numbers) or saved to disk as a file — a `.png`, a `.csv`, a trained `.keras` model — which
the next script then reads back in.

```
data/raw/  (RAVDESS + TESS .wav files)
    │
    ▼
① data_utils.py       →  scans every actor/emotion folder, writes manifest.csv
    │                     (filepath → emotion label)
    ▼
② build_dataset.py    →  reads manifest.csv, extracts MFCC + Mel-spectrogram
    │                     features for all 4,240 clips, caches as .npy files
    ▼
③ eda.py               →  reads the .npy files, saves class-balance and
    │                     waveform/spectrogram example charts as .png
    ▼
④ train.py              →  reads the .npy files, trains ANN / CNN / CNN-Transformer
    │                     with train-only data augmentation, prints accuracy
    │                     + classification reports, saves .keras models
    ▼
⑤ app.py                →  Streamlit GUI — loads the trained models, lets you
                          upload a .wav and see a live prediction
```

### Script reference

| Script | Reads | Produces |
|---|---|---|
| `src/data_utils.py` | `data/raw/RAVDESS/`, `data/raw/TESS/` | `data/processed/manifest.csv` |
| `src/build_dataset.py` | `manifest.csv` | `X_mfcc.npy`, `X_mel.npy`, `y.npy` |
| `src/eda.py` | `manifest.csv`, raw audio | `reports/class_balance.png`, `reports/waveform_spectrogram_examples.png` |
| `src/train.py` | `X_mfcc.npy`, `X_mel.npy`, `y.npy` | `models/ann.keras`, `models/cnn.keras`, `models/cnn_transformer.keras` |
| `app.py` | the three `.keras` models + an uploaded `.wav` | a live prediction in the browser |

Run them in order from the project root:

```bash
python src/data_utils.py
python src/build_dataset.py
python src/eda.py
python src/train.py
streamlit run app.py
```

---

## 6. Challenges we addressed

1. **Audio preprocessing** — converting raw `.wav` files into Mel-spectrograms and MFCC vectors with
   Librosa required learning audio signal processing fundamentals.
2. **CNN architecture selection** — spectrograms aren't natural images, so kernel sizes, pooling, and
   layer depth needed experimentation.
3. **Overfitting on a moderate dataset** — with ~4,240 samples, we used dropout throughout and
   train-only data augmentation (noise injection + pitch shifting) to regularize the CNN/CNN-Transformer.
4. **Distinguishing similar emotions** — calm vs. neutral and fearful vs. sad have very similar
   acoustic signatures; this shows up directly in the per-emotion results below.
5. **Team coordination** — integrating preprocessing, model-building, and evaluation into one
   reproducible pipeline across four people.

---

## 7. Results in detail — CNN-Transformer

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| Angry | 0.81 | 0.94 | 0.87 |
| Calm | 0.78 | 0.55 | 0.65 |
| Disgust | 0.79 | 0.89 | 0.83 |
| Fearful | 0.88 | 0.94 | 0.91 |
| Happy | 0.94 | 0.85 | 0.89 |
| Neutral | 0.96 | 0.90 | 0.93 |
| Sad | 0.88 | 0.75 | 0.81 |
| Surprised | 0.92 | 0.95 | 0.93 |
| **Overall accuracy** | | | **0.873** |

**Calm is the weak point** — same as with the other two models. It's acoustically close to neutral
and TESS doesn't include a "calm" class at all, so it has the fewest training examples of the
eight; the confusion matrix shows calm getting misread mostly as disgust and sad. This is a good
talking point for the presentation: the model's mistakes make intuitive sense, not random noise —
and it's consistent across all three architectures, which is itself evidence the models learned
something real about the data rather than overfitting to one architecture's quirks.

---

## 8. The live demo (`app.py`)

A Streamlit web app — this is the piece that gives an interactive, Colab-notebook-like experience
instead of scrolling terminal text. Run `streamlit run app.py` and it opens in the browser.

**Demo flow:**
1. Upload a `.wav` clip
2. Pick a model — ANN, CNN, or CNN-Transformer
3. See its waveform and Mel-spectrogram (the actual model input, visualized)
4. Get the predicted emotion + confidence, color-coded
5. Expand **"Compare across all trained models"** to see all three predictions side by side on the
   same clip

**Good demo moment:** upload one clip and show that all three models — trained completely
independently, on different feature types — usually agree. It's a clean way to show the models
learned something real and consistent, not just memorized noise.

---

## 9. Team & roles

| Member | Responsibilities |
|---|---|
| Tarun Karnati | Project coordination, literature review, CNN model implementation |
| Harmandeep Kaur | Audio preprocessing, MFCC & Mel-spectrogram generation, EDA |
| Kiranjeet Kaur Deol | ANN baseline and CNN-Transformer model building and training |
| Jashandeep Kaur | Model evaluation, confusion matrices, visualizations, final report |

---

## 10. References

- Singh, J., Saheer, L. B., & Faust, O. (2023). Speech emotion recognition using attention model.
  *International Journal of Environmental Research and Public Health, 20*(6), 5140.
- Atila, O., & Şengür, A. (2021). Attention guided 3D CNN-LSTM model for accurate speech based
  emotion recognition. *Applied Acoustics, 182*, 108260.
- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech
  and Song (RAVDESS). *PLOS ONE, 13*(5), e0196391.
- Toronto Emotional Speech Set (TESS). (2020). University of Toronto.
