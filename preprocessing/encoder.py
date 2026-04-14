"""
preprocessing/encoder.py
========================
Encodes categorical features and the target label.
Uses LabelEncoder for the target, and one-hot encoding for other
high-cardinality categoricals if present.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def encode_target(df: pd.DataFrame, target_col: str, encoder_path: str) -> tuple:
    """
    Label-encode the multi-class target column.

    Args:
        df:           DataFrame containing target_col
        target_col:   Name of the column with attack class labels (string)
        encoder_path: Where to save the fitted LabelEncoder (.pkl)

    Returns:
        (df, label_encoder, class_names)
    """
    le = LabelEncoder()
    df = df.copy()
    df[target_col] = le.fit_transform(df[target_col].astype(str))

    class_names = list(le.classes_)
    logger.info(f"Encoded target '{target_col}' → {len(class_names)} classes: {class_names}")

    joblib.dump(le, encoder_path)
    logger.info(f"  LabelEncoder saved → {encoder_path}")

    return df, le, class_names


def encode_categorical_features(df: pd.DataFrame, exclude_cols: list = None) -> pd.DataFrame:
    """
    One-hot encode remaining categorical (object/string) columns.
    Drops the original columns and appends dummies.

    Args:
        df:           Input DataFrame (target already encoded)
        exclude_cols: Columns to skip (e.g., target column)

    Returns:
        DataFrame with categoricals encoded
    """
    exclude_cols = exclude_cols or []
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in exclude_cols
    ]

    if not cat_cols:
        logger.info("No categorical feature columns to encode.")
        return df

    logger.info(f"One-hot encoding columns: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    logger.info(f"  Shape after encoding: {df.shape}")
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list:
    """Return list of feature column names (everything except target)."""
    return [c for c in df.columns if c != target_col]
