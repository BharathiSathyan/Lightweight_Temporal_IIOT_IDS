"""
explainability/shap_explainer.py
================================
SHAP-based feature importance and prediction explanations.

Uses KernelExplainer (model-agnostic) on flattened sequences,
or a wrapper to explain the underlying tabular features.

Outputs:
  - Global SHAP summary plot (bar + beeswarm)
  - Per-class force plots for selected samples
  - Feature importance ranking CSV
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import shap

logger = logging.getLogger(__name__)


def _build_predict_fn(model, n_timesteps: int, n_features: int):
    """
    Wrap a Keras sequence model to accept flattened 2D input for SHAP.
    SHAP expects a function: f(X_flat) → probabilities
    """
    def predict_fn(X_flat: np.ndarray) -> np.ndarray:
        # Reshape (N, T*F) → (N, T, F)
        X_seq = X_flat.reshape(-1, n_timesteps, n_features)
        return model.predict(X_seq, verbose=0)
    return predict_fn


def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    n_timesteps: int,
    n_features: int,
    n_background: int = 100,
) -> np.ndarray:
    """
    Compute SHAP values using KernelExplainer.

    Args:
        model:        Trained Keras model
        X_background: Background dataset for SHAP (N, T, F) — summarised internally
        X_explain:    Samples to explain (M, T, F)
        n_timesteps:  Sequence timesteps
        n_features:   Features per timestep
        n_background: Number of background samples (keep small for speed on CPU)

    Returns:
        SHAP values array: (n_classes, M, T*F)
    """
    logger.info("Computing SHAP values (KernelExplainer — this may take a few minutes)...")

    predict_fn = _build_predict_fn(model, n_timesteps, n_features)

    # Flatten background and explanation data
    bg_flat = X_background[:n_background].reshape(n_background, -1)
    ex_flat = X_explain.reshape(len(X_explain), -1)

    # Summarise background with k-means for speed
    bg_summary = shap.kmeans(bg_flat, k=min(50, n_background))

    explainer  = shap.KernelExplainer(predict_fn, bg_summary)
    shap_vals  = explainer.shap_values(ex_flat, nsamples=100, silent=True)

    logger.info(f"  SHAP values computed: {len(shap_vals)} classes")
    return shap_vals  # list of arrays, one per class


def plot_shap_summary(
    shap_values,
    X_explain: np.ndarray,
    feature_names: list,
    n_timesteps: int,
    n_features: int,
    save_dir: str,
    top_n: int = 15,
) -> None:
    """
    Generate global SHAP summary (bar) plot — averaged over classes and timesteps.

    Feature names are repeated for each timestep as: "feat_t0", "feat_t1", ...
    Then we collapse over timesteps to get per-feature importance.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Build expanded feature names (T * F names)
    expanded_names = [
        f"{fname}_t{t}" for t in range(n_timesteps) for fname in feature_names
    ]

    # Average SHAP values across classes
    # shap_values is list of (M, T*F) arrays
<<<<<<< HEAD
    # mean_shap = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)  # (M, T*F)

    # ── Fix: handle multi-class SHAP properly ─────────────────────
    # ── Fix multi-class SHAP output ───────────────────────────────
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=0)  # (C, M, T*F)
        shap_values = np.mean(np.abs(shap_values), axis=0)  # (M, T*F)

    # If SHAP returned (M, T*F, C)
    elif len(shap_values.shape) == 3:
        shap_values = np.mean(np.abs(shap_values), axis=2)  # (M, T*F)

    mean_shap = shap_values  # final shape (M, T*F)
=======
    mean_shap = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)  # (M, T*F)
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b
    X_flat = X_explain.reshape(len(X_explain), -1)

    # ── Collapse over timesteps: mean per original feature ───────────────────
    # Group by feature index (stride = n_features)
    mean_per_feature = np.zeros((mean_shap.shape[0], n_features))
    for t in range(n_timesteps):
        start = t * n_features
        end   = start + n_features
        mean_per_feature += mean_shap[:, start:end]
    mean_per_feature /= n_timesteps

    global_importance = mean_per_feature.mean(axis=0)   # (n_features,)
    top_idx = np.argsort(global_importance)[::-1][:top_n]

    # ── Bar Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in top_idx[::-1]],
        global_importance[top_idx[::-1]],
        color="#e05252"
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Feature Importances (SHAP)")
    plt.tight_layout()
    path = os.path.join(save_dir, "shap_global_importance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved: {path}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_shap": global_importance
    }).sort_values("mean_shap", ascending=False)
    csv_path = os.path.join(save_dir, "shap_feature_importance.csv")
    df_importance.to_csv(csv_path, index=False)
    logger.info(f"  Saved: {csv_path}")


def plot_shap_per_class(
    shap_values,
    X_explain: np.ndarray,
    y_true: np.ndarray,
    feature_names: list,
    class_names: list,
    n_timesteps: int,
    n_features: int,
    save_dir: str,
    target_classes: list = None,
    top_n: int = 10,
) -> None:
    """
    Bar plot of SHAP importance for 3+ specific attack classes.

    Args:
        target_classes: List of class indices to explain (default: first 3 attack classes)
    """
    os.makedirs(save_dir, exist_ok=True)

    if target_classes is None:
        # Skip class 0 (benign), take next 3 attack classes
        target_classes = list(range(1, min(4, len(class_names))))

    for cls_idx in target_classes:
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
<<<<<<< HEAD
        # sv_cls   = shap_values[cls_idx]   # (M, T*F)

        # ── Fix multi-class SHAP output ─────────────────────────────
        if isinstance(shap_values, list):
            # shap_values: list of (M, T*F)
            sv_cls = shap_values[cls_idx]

        elif len(shap_values.shape) == 3:
            # shap_values: (M, T*F, C)
            sv_cls = shap_values[:, :, cls_idx]  # select one class

        else:
            sv_cls = shap_values  # fallback
=======
        sv_cls   = shap_values[cls_idx]   # (M, T*F)
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b

        # Mean |SHAP| per original feature (collapsed over timesteps)
        mean_per_feature = np.zeros(n_features)
        for t in range(n_timesteps):
            start = t * n_features
            end   = start + n_features
            mean_per_feature += np.abs(sv_cls[:, start:end]).mean(axis=0)
        mean_per_feature /= n_timesteps

        top_idx = np.argsort(mean_per_feature)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            [feature_names[i] for i in top_idx[::-1]],
            mean_per_feature[top_idx[::-1]],
            color="#4a7ecb"
        )
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Feature Importance for Class: {cls_name}")
        plt.tight_layout()
        safe_name = cls_name.replace(" ", "_").replace("/", "_")
        path = os.path.join(save_dir, f"shap_{safe_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
<<<<<<< HEAD
        logger.info(f"  Saved class SHAP plot: {path}")
=======
        logger.info(f"  Saved class SHAP plot: {path}")
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b
