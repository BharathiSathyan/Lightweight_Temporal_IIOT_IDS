"""
models/ablation_models.py
=========================
Ablation variants of the hybrid CNN-GRU model:
  - CNN only   (no GRU)
  - GRU only   (no CNN)

Used alongside the full CNN-GRU to measure each component's contribution.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)


def build_cnn_only(
    n_timesteps: int,
    n_features: int,
    n_classes: int,
    cnn_filters: int = 64,
    cnn_kernel: int = 3,
    dense_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    CNN-only model: two convolutional blocks → GlobalAvgPool → Dense.
    No recurrent component.
    """
    inp = keras.Input(shape=(n_timesteps, n_features), name="input")

    x = layers.Conv1D(cnn_filters, cnn_kernel, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(cnn_filters * 2, cnn_kernel, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling1D()(x)   # replaces GRU
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out, name="CNN_Only")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def build_gru_only(
    n_timesteps: int,
    n_features: int,
    n_classes: int,
    gru_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    GRU-only model: raw input → stacked GRU → Dense.
    No convolutional feature extraction.
    """
    inp = keras.Input(shape=(n_timesteps, n_features), name="input")

    x = layers.GRU(gru_units, return_sequences=True, name="gru1")(inp)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.GRU(gru_units // 2, return_sequences=False, name="gru2")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inp, outputs=out, name="GRU_Only")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
