"""Model architectures: ANN baseline, CNN, and CNN-Transformer hybrid."""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_ann(input_dim: int, num_classes: int) -> models.Sequential:
    """The simplest model: takes the 40-number MFCC vector and passes it through
    a stack of Dense layers, each one mixing the numbers together and shrinking
    them down, until we're left with 8 numbers -- one probability per emotion."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),          # 40 numbers in (the MFCC vector)
        layers.Dense(256, activation="relu"),        # mix into 256 numbers
        layers.Dropout(0.3),                           # randomly ignore 30% of them (prevents memorizing)
        layers.Dense(128, activation="relu"),        # squeeze down to 128
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),         # squeeze down to 64
        layers.Dense(num_classes, activation="softmax"),  # 8 numbers out (one per emotion, sums to 100%)
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn(input_shape: tuple, num_classes: int) -> models.Sequential:
    """Treats the 128x130 spectrogram like a picture. Three rounds of
    Conv2D -> BatchNorm -> MaxPooling -> Dropout each look for patterns, then
    shrink the picture in half; by the end we've gone from a big picture to a
    small, pattern-rich summary that the last two Dense layers turn into a guess."""
    model = models.Sequential([
        layers.Input(shape=input_shape),  # (128 pitch bands, 130 time steps, 1)

        # --- round 1: look for small, simple patterns ---
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),   # shrink the picture by half
        layers.Dropout(0.25),

        # --- round 2: look for combinations of round-1 patterns ---
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # --- round 3: look for bigger, more complex patterns ---
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),                                # turn the small picture into one long list of numbers
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),  # 8 numbers out (one per emotion)
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def positional_encoding(time_steps: int, d_model: int) -> tf.Tensor:
    """Fixed sin/cos positional encoding so the Transformer knows the order of time frames."""
    positions = tf.range(time_steps, dtype=tf.float32)[:, tf.newaxis]
    dims = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    angle_rates = 1.0 / tf.pow(10000.0, (2 * (dims // 2)) / tf.cast(d_model, tf.float32))
    angles = positions * angle_rates
    sines = tf.sin(angles[:, 0::2])
    cosines = tf.cos(angles[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    return pos_encoding[tf.newaxis, ...]  # (1, time_steps, d_model)


def transformer_encoder_block(x, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.3):
    """One Transformer encoder block: self-attention + feed-forward, each with a residual
    connection and LayerNorm (pre-norm style, standard for stabilizing small datasets)."""
    attn_input = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(attn_input, attn_input)
    attn_out = layers.Dropout(dropout)(attn_out)
    x = layers.Add()([x, attn_out])

    ff_input = layers.LayerNormalization(epsilon=1e-6)(x)
    ff_out = layers.Dense(ff_dim, activation="relu")(ff_input)
    ff_out = layers.Dense(d_model)(ff_out)
    ff_out = layers.Dropout(dropout)(ff_out)
    x = layers.Add()([x, ff_out])
    return x


def build_cnn_transformer(input_shape: tuple, num_classes: int, num_blocks: int = 2, num_heads: int = 4, d_model: int = 128) -> models.Model:
    """CNN front-end (same conv stack as the CNN model) extracts local time-frequency
    patterns from the Mel-spectrogram, then a Transformer encoder attends across time
    steps to model longer-range dependencies than the CNN's local receptive field can."""
    inputs = layers.Input(shape=input_shape)  # (n_mels, time, 1)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Reshape (freq, time, channels) -> (time, freq*channels): one token per time step,
    # then project down to a fixed embedding size so attention stays cheap on CPU.
    time_steps = input_shape[1] // 4  # two MaxPooling2D((2,2)) layers
    raw_dim = model_time_features(input_shape)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((time_steps, raw_dim))(x)
    x = layers.Dense(d_model)(x)

    x = x + positional_encoding(time_steps, d_model)
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, d_model=d_model, num_heads=num_heads, ff_dim=d_model * 2)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="cnn_transformer")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def model_time_features(input_shape: tuple) -> int:
    """Computes flattened feature size per timestep after two (2,2) pooling layers."""
    n_mels, _, channels = input_shape
    pooled_mels = n_mels // 4  # two MaxPooling2D((2,2)) layers
    return pooled_mels * 64  # 64 = channels out of last Conv2D before reshape
