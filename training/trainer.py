"""
training/trainer.py
===================
Handles model training with:
  - EarlyStopping on validation loss
  - ReduceLROnPlateau
  - Training time measurement
  - History saving
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging

logger = logging.getLogger(__name__)


def get_callbacks(
    model_save_path: str,
    early_stop_patience: int = 7,
    lr_reduce_patience: int = 3,
    lr_reduce_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list:
    """
    Build standard training callbacks.

    Returns list of:
        - ModelCheckpoint (saves best model)
        - EarlyStopping (stops on val_loss plateau)
        - ReduceLROnPlateau (halves LR on plateau)
        - TensorBoard (optional, disabled by default)
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=lr_reduce_factor,
            patience=lr_reduce_patience,
            min_lr=min_lr,
            verbose=1,
        ),
    ]
    return callbacks


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    save_dir: str,
    batch_size: int = 64,
    epochs: int = 30,
    early_stop_patience: int = 7,
    lr_reduce_patience: int = 3,
    lr_reduce_factor: float = 0.5,
    min_lr: float = 1e-6,
    class_weight: dict = None,
) -> tuple:
    """
    Train a Keras model and return history + wall-clock training time.

    Args:
        model:       Compiled Keras model
        X_train:     Training sequences (N, T, F)
        y_train:     One-hot labels (N, C)
        X_val:       Validation sequences
        y_val:       Validation one-hot labels
        model_name:  Used for naming saved files
        save_dir:    Directory to save best model
        batch_size:  Mini-batch size
        epochs:      Maximum training epochs
        class_weight: Dict mapping class index → weight (for imbalance)

    Returns:
        (history, training_time_seconds)
    """
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f"{model_name}.keras")

    callbacks = get_callbacks(
        model_save_path,
        early_stop_patience=early_stop_patience,
        lr_reduce_patience=lr_reduce_patience,
        lr_reduce_factor=lr_reduce_factor,
        min_lr=min_lr,
    )

    logger.info(f"Training [{model_name}] — {X_train.shape[0]:,} samples | "
                f"batch={batch_size} | max_epochs={epochs}")

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    elapsed = time.time() - start_time

    logger.info(f"  Training complete in {elapsed:.1f}s "
                f"({elapsed/60:.2f} min) | "
                f"Best val_loss at epoch {np.argmin(history.history['val_loss'])+1}")

    # Save history JSON for later plotting
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        # Convert numpy floats to Python floats
        serializable_history = {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        }
        json.dump(serializable_history, f, indent=2)
    logger.info(f"  History saved → {history_path}")

    return history, elapsed


def compute_class_weights(y_integer: np.ndarray, n_classes: int) -> dict:
    """
    Compute balanced class weights to handle label imbalance.

    Args:
        y_integer:  Integer class labels (N,)
        n_classes:  Total number of classes

    Returns:
        Dict {class_index: weight}
    """
    from sklearn.utils.class_weight import compute_class_weight
    unique_classes = np.unique(y_integer)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_integer,
    )
    weight_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}
    logger.info(f"Class weights: {weight_dict}")
    return weight_dict
