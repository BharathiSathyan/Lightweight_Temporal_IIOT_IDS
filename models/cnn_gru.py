"""
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

logger = logging.getLogger(__name__)


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
