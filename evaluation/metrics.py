"""
evaluation/metrics.py
=====================
Compute and display all classification metrics:
  - Accuracy, Precision, Recall, F1-Score (macro + per-class)
  - Confusion matrix
  - Classification report
"""

import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test_onehot: np.ndarray,
    class_names: list,
    batch_size: int = 64,
) -> dict:
    """
    Full evaluation of a trained model on test data.

    Args:
        model:         Trained Keras model
        X_test:        Test sequences (N, T, F)
        y_test_onehot: One-hot encoded test labels (N, C)
        class_names:   List of class name strings
        batch_size:    Batch size for prediction

    Returns:
        Dict with all metrics + confusion matrix
    """
    # Predictions
    y_pred_proba = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    y_true       = np.argmax(y_test_onehot, axis=1)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class F1
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Full text report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

    results = {
        "accuracy":      round(acc, 4),
        "precision":     round(prec, 4),
        "recall":        round(rec, 4),
        "f1_macro":      round(f1, 4),
        "f1_per_class":  {cls: round(float(v), 4) for cls, v in zip(class_names, f1_per_class)},
        "confusion_matrix": cm.tolist(),
        "report":        report,
        "y_true":        y_true,
        "y_pred":        y_pred,
        "y_pred_proba":  y_pred_proba,
    }

    logger.info("─── Evaluation Results ─────────────────────────────────")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  F1 Macro:  {f1:.4f}")
    logger.info(f"\n{report}")

    return results


def compare_models(results_dict: dict) -> None:
    """
    Print a comparison table of multiple model results.

    Args:
        results_dict: {model_name: metrics_dict}
    """
    logger.info("─── Model Comparison ────────────────────────────────────")
    header = f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    logger.info(header)
    logger.info("─" * 65)
    for name, m in results_dict.items():
        row = (
            f"{name:<20} "
            f"{m['accuracy']:>10.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1_macro']:>10.4f}"
        )
<<<<<<< HEAD
        logger.info(row)
=======
        logger.info(row)
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b
