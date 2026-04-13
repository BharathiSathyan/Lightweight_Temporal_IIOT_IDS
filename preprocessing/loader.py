"""
preprocessing/loader.py
=======================
Handles loading the NF-ToN-IoT-V2 parquet dataset, converting to CSV,
removing duplicates, and performing initial feature analysis.
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_parquet_to_csv(parquet_path: str, csv_path: str) -> pd.DataFrame:
    """
    Load parquet file and save as CSV.

    Args:
        parquet_path: Path to .parquet file
        csv_path:     Output .csv path

    Returns:
        DataFrame of the loaded data
    """
    logger.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    logger.info(f"  Shape: {df.shape} | Columns: {list(df.columns)}")
    df.to_csv(csv_path, index=False)
    logger.info(f"  Saved CSV → {csv_path}")
    return df


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load an existing CSV file."""
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"  Shape: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info(f"Duplicates removed: {before - after} ({before} → {after})")
    return df


def drop_irrelevant_features(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    """
    Drop columns that are not useful for classification
    (e.g., IP addresses, ports used as identifiers).

    Args:
        df:        Input DataFrame
        drop_cols: List of column names to drop (only existing ones are removed)

    Returns:
        Cleaned DataFrame
    """
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing, errors="ignore")
    logger.info(f"Dropped columns: {existing}")
    return df


def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Print and return basic dataset statistics.

    Returns dict with:
        n_rows, n_cols, class_dist, null_counts, dtypes
    """
    stats = {
        "n_rows":      len(df),
        "n_cols":      len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "dtypes":      df.dtypes.astype(str).to_dict(),
    }

    logger.info("─── Dataset Inspection ──────────────────────────")
    logger.info(f"  Rows: {stats['n_rows']:,} | Cols: {stats['n_cols']}")
    logger.info(f"  Null values:\n{pd.Series(stats['null_counts']).loc[lambda x: x > 0]}")

    return stats


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill or drop missing values.
    Numeric columns → filled with median.
    Categorical columns → filled with mode.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "unknown")

    logger.info("Missing values handled (numeric→median, categorical→mode)")
    return df