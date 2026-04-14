"""
models/lstm_baseline.py
=======================
Bidirectional LSTM baseline model for comparison with CNN-GRU.

Architecture:
    Input(window_size, n_features)
        │
    BiLSTM(128)
        │
    Dropout
        │
    Dense(64) → Dropout
        │
    Dense(n_classes, softmax)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)


def build_lstm(
    n_timesteps: int,
    n_features: int,
    n_classes: int,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Bidirectional LSTM classifier.

    Args:
        n_timesteps:  Sequence length
        n_features:   Features per timestep
        n_classes:    Number of output classes
        lstm_units:   LSTM hidden size
        dense_units:  Dense layer size
        dropout_rate: Dropout probability
        learning_rate: Adam LR

    Returns:
        Compiled Keras model
    """
    inp = keras.Input(shape=(n_timesteps, n_features), name="input")

    # ── BiLSTM layers ────────────────────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True), name="bilstm1"
    )(inp)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=False), name="bilstm2"
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # ── Classification Head ──────────────────────────────────────────────────
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="BiLSTM_Baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary(print_fn=logger.info)
    return model
