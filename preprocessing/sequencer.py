"""
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

logger = logging.getLogger(__name__)


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
