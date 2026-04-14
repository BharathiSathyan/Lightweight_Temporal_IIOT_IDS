"""
<<<<<<< HEAD
models/cnn_gru.py
=================
Hybrid CNN-GRU architecture for IDS classification.

Architecture:
    Input(window_size, n_features)
        │
    Conv1D → BatchNorm → ReLU → MaxPool
        │
    Conv1D → BatchNorm → ReLU → MaxPool   (optional 2nd conv block)
        │
    GRU(units)
        │
    Dense(128) → Dropout
        │
    Dense(n_classes, softmax)

Design choices:
  - 1D CNN captures local temporal patterns in the flow window
  - GRU captures long-range dependencies across the CNN embeddings
  - Lightweight enough for CPU inference
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
=======
preprocessing/sequencer.py
==========================
Converts tabular network flow data into temporal sequences
using a sliding window approach.

Output shape: (n_sequences, window_size, n_features)
Labels: majority class within each window (or last-label strategy)
"""

import numpy as np
import logging
from scipy.stats import mode as scipy_mode
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b

logger = logging.getLogger(__name__)


<<<<<<< HEAD
def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Focal loss to address class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def loss_fn(y_true, y_pred):
        # y_true: one-hot (batch, n_classes)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        probs = tf.reduce_sum(y_true * y_pred, axis=-1)
        fl = alpha * tf.pow(1 - probs, gamma) * ce
        return tf.reduce_mean(fl)
    loss_fn.__name__ = "focal_loss"
    return loss_fn


def build_cnn_gru(
    n_timesteps: int,
    n_features: int,
    n_classes: int,
    cnn_filters: int = 64,
    cnn_kernel: int = 3,
    pool_size: int = 2,
    gru_units: int = 64,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
) -> keras.Model:
    """
    Build and compile the CNN-GRU hybrid model.

    Args:
        n_timesteps:    Sequence length (window size)
        n_features:     Number of input features per timestep
        n_classes:      Number of output classes
        cnn_filters:    Number of 1D CNN filters
        cnn_kernel:     Kernel size for convolutions
        pool_size:      Max-pooling pool size
        gru_units:      Hidden units in GRU layer
        dense_units:    Units in intermediate dense layer
        dropout_rate:   Dropout probability
        learning_rate:  Adam LR
        use_focal_loss: Use focal loss (True) or categorical crossentropy (False)
        focal_gamma:    Gamma for focal loss

    Returns:
        Compiled Keras model
    """
    inp = keras.Input(shape=(n_timesteps, n_features), name="input")

    # ── CNN Block 1 ──────────────────────────────────────────────────────────
    x = layers.Conv1D(cnn_filters, cnn_kernel, padding="same", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling1D(pool_size, name="pool1")(x)

    # ── CNN Block 2 (deeper representation) ─────────────────────────────────
    x = layers.Conv1D(cnn_filters * 2, cnn_kernel, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)

    # ── GRU Temporal Layer ───────────────────────────────────────────────────
    x = layers.GRU(gru_units, return_sequences=False, name="gru")(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)

    # ── Classification Head ──────────────────────────────────────────────────
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN_GRU_IDS")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_fn = focal_loss(gamma=focal_gamma) if use_focal_loss else "categorical_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    model.summary(print_fn=logger.info)
    return model


def get_model_size_mb(model: keras.Model) -> float:
    """Estimate model parameter size in megabytes."""
    total_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    size_mb = (total_params * 4) / (1024 ** 2)   # float32 = 4 bytes
    return round(size_mb, 3)
=======
def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int = 20,
    step_size: int = 10,
    label_strategy: str = "last"
) -> tuple:
    """
    Build sliding window sequences from tabular data.

    Args:
        X:               Feature matrix (n_samples, n_features)
        y:               Label array (n_samples,)
        window_size:     Number of flows per sequence (timesteps)
        step_size:       Stride between consecutive windows
        label_strategy:  "last"     → use label of last sample in window
                         "majority" → use most frequent label in window

    Returns:
        (X_seq, y_seq) where
            X_seq shape: (n_sequences, window_size, n_features)
            y_seq shape: (n_sequences,)
    """
    n_samples, n_features = X.shape
    sequences, labels = [], []

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window_X = X[start:end]          # (window_size, n_features)
        window_y = y[start:end]          # (window_size,)

        sequences.append(window_X)

        if label_strategy == "majority":
            label = int(scipy_mode(window_y, keepdims=True).mode[0])
        else:  # "last"
            label = int(window_y[-1])

        labels.append(label)

    X_seq = np.array(sequences, dtype=np.float32)   # (N, T, F)
    y_seq = np.array(labels, dtype=np.int32)         # (N,)

    logger.info(
        f"Sequences created: {X_seq.shape} | "
        f"Labels: {y_seq.shape} | "
        f"Window: {window_size} | Step: {step_size}"
    )
    return X_seq, y_seq


def stratified_sample(
    df,
    target_col: str,
    n_samples: int,
    random_seed: int = 42
):
    """
    Draw a stratified sample from a DataFrame, preserving class ratios.

    Args:
        df:          Full DataFrame
        target_col:  Name of label column
        n_samples:   Desired total sample count
        random_seed: For reproducibility

    Returns:
        Sampled DataFrame (shuffled)
    """
    from sklearn.utils import resample

    classes = df[target_col].unique()
    n_per_class = max(1, n_samples // len(classes))
    parts = []

    for cls in classes:
        subset = df[df[target_col] == cls]
        sampled = resample(
            subset,
            n_samples=min(n_per_class, len(subset)),
            replace=False,
            random_state=random_seed
        )
        parts.append(sampled)

    result = (
        pd.concat(parts)
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )
    logger.info(
        f"Stratified sample: {len(result):,} rows | "
        f"{result[target_col].value_counts().to_dict()}"
    )
    return result


# need pandas for the function above
import pandas as pd  # noqa: E402
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b
