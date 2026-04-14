"""
utils/helpers.py
================
General utility functions:
  - Seed setting for reproducibility
  - Data split helpers
  - One-hot encoding utilities
  - Logging setup
"""

import os
import random
import numpy as np
import logging
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def set_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    logging.getLogger(__name__).info(f"Seeds set: {seed}")


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger with formatted output."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
) -> tuple:
    """
    Stratified 3-way split: train / val / test.

    Args:
        X:           Feature sequences (N, T, F) or tabular (N, F)
        y:           Integer labels (N,)
        test_size:   Fraction for test
        val_size:    Fraction of total for validation
        random_seed: For reproducibility

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)  — all integer labels
    """
    # Step 1: Split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    # Step 2: Split val from remaining
    adjusted_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=adjusted_val,
        stratify=y_trainval,
        random_state=random_seed
    )

    log = logging.getLogger(__name__)
    log.info(
        f"Data split → train: {len(X_train):,} | "
        f"val: {len(X_val):,} | test: {len(X_test):,}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def to_onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    return to_categorical(y, num_classes=n_classes).astype(np.float32)


def save_json(obj: dict, path: str) -> None:
    """Serialize dict to JSON with float conversion."""
    def convert(o):
        if isinstance(o, (np.integer,)):    return int(o)
        if isinstance(o, (np.floating,)):   return float(o)
        if isinstance(o, np.ndarray):       return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=convert)
    logging.getLogger(__name__).info(f"Saved JSON: {path}")


def load_json(path: str) -> dict:
    with open(path) as f:
<<<<<<< HEAD
        return json.load(f)
=======
        return json.load(f)
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b
