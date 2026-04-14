"""
models/mlp_baseline.py
======================
Simple MLP (fully connected) baseline that operates on flattened sequences.
Useful as a lower-bound comparison.

Architecture:
    Input(window_size * n_features)
        │
    Dense(256) → BN → ReLU → Dropout
        │
    Dense(128) → BN → ReLU → Dropout
        │
    Dense(64)  → ReLU
        │
    Dense(n_classes, softmax)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging

logger = logging.getLogger(__name__)


def build_mlp(
    n_timesteps: int,
    n_features: int,
    n_classes: int,
    hidden_units: list = None,
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    MLP classifier over flattened sequence input.

    Args:
        n_timesteps:  Sequence length
        n_features:   Features per timestep
        n_classes:    Number of output classes
        hidden_units: List of hidden layer sizes
        dropout_rate: Dropout probability
        learning_rate: Adam LR

    Returns:
        Compiled Keras model
    """
    hidden_units = hidden_units or [256, 128, 64]

    inp = keras.Input(shape=(n_timesteps, n_features), name="input")
    x = layers.Flatten(name="flatten")(inp)   # (n_timesteps * n_features,)

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, name=f"dense_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="MLP_Baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary(print_fn=logger.info)
    return model
