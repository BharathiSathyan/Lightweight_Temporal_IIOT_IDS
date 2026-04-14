"""
preprocessing/normalizer.py
===========================
Applies MinMaxScaler to numeric features.
Scaler is fitted on training data only to prevent data leakage.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def fit_scaler(X_train: np.ndarray, scaler_path: str) -> MinMaxScaler:
    """
    Fit MinMaxScaler on training data and save to disk.

    Args:
        X_train:     Training feature array (n_samples, n_features)
        scaler_path: Path to persist the scaler

    Returns:
        Fitted MinMaxScaler
    """
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler fitted and saved → {scaler_path}")
    return scaler


def apply_scaler(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Transform feature array using a fitted scaler.
    Clips values to [0, 1] to handle out-of-range test samples.
    """
    X_scaled = scaler.transform(X)
    X_scaled = np.clip(X_scaled, 0.0, 1.0)
    return X_scaled


def load_scaler(scaler_path: str) -> MinMaxScaler:
    """Load a previously fitted scaler."""
    scaler = joblib.load(scaler_path)
    logger.info(f"Scaler loaded from {scaler_path}")
<<<<<<< HEAD
    return scaler
=======
    return scaler
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b
