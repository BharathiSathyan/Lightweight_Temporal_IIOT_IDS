"""
demo_synthetic.py
=================
End-to-end demo using synthetically generated data.
Run this to verify your installation works before loading the real dataset.

Usage:
  python demo_synthetic.py

This will:
  1. Generate synthetic network flow data mimicking NF-ToN-IoT-V2
  2. Run full preprocessing (sequencing, normalization, splitting)
  3. Train CNN-GRU + baselines for 5 quick epochs
  4. Evaluate and generate all plots
  5. Run a mini SHAP explanation
  6. Run ablation study

No dataset download required — great for CI and setup testing.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

# Make project root importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers    import set_seeds, setup_logging, split_data, to_onehot, save_json
from utils.visualizer import (plot_training_curves, plot_confusion_matrix,
                               plot_model_comparison, plot_efficiency_comparison,
                               plot_class_f1)
from training.config  import PLOTS_DIR, MODELS_DIR, RANDOM_SEED
import tensorflow as tf

setup_logging()
logger = logging.getLogger(__name__)
set_seeds(RANDOM_SEED)


# ─── Synthetic Data Generation ────────────────────────────────────────────────

def make_synthetic_dataset(
    n_samples: int = 8_000,
    n_features: int = 30,
    n_classes: int = 7,
    random_seed: int = 42,
):
    """
    Generate a realistic-looking synthetic network flow dataset.

    Classes: Benign, DDoS, DoS, Backdoor, Injection, MITM, Ransomware
    """
    rng = np.random.default_rng(random_seed)
    class_names = ["Benign", "DDoS", "DoS", "Backdoor", "Injection", "MITM", "Ransomware"]

    X_parts, y_parts = [], []
    # Unequal class distribution (realistic imbalance)
    proportions = [0.40, 0.15, 0.15, 0.08, 0.08, 0.07, 0.07]

    for cls_idx, (cls_name, prop) in enumerate(zip(class_names, proportions)):
        n = max(1, int(n_samples * prop))
        # Each class has a slightly different mean/variance pattern
        mean = rng.uniform(0.1, 0.9, size=n_features) * (cls_idx * 0.1 + 0.5)
        std  = rng.uniform(0.05, 0.25, size=n_features)
        X_cls = rng.normal(mean, std, size=(n, n_features)).clip(0, 1).astype(np.float32)
        y_cls = np.full(n, cls_idx, dtype=np.int32)
        X_parts.append(X_cls)
        y_parts.append(y_cls)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle
    idx = rng.permutation(len(X))
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]

    logger.info(f"Synthetic dataset: {X.shape} | classes: {class_names}")
    return X, y, class_names, feature_names


# ─── Main Demo ────────────────────────────────────────────────────────────────

def run_demo():
    WINDOW_SIZE = 10     # short window for quick demo
    STEP_SIZE   = 5
    BATCH_SIZE  = 32
    EPOCHS      = 5      # intentionally short for demo speed
    N_FEATURES  = 30
    N_CLASSES   = 7

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║   IDS Demo — Synthetic Data (No Dataset Required)    ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    # 1. Generate data
    X_raw, y_raw, class_names, feature_names = make_synthetic_dataset(
        n_samples=8_000, n_features=N_FEATURES, n_classes=N_CLASSES
    )

    # 2. Split (tabular first)
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(
        X_raw, y_raw, test_size=0.15, val_size=0.15, random_seed=RANDOM_SEED
    )

    # 3. Sequence
    from preprocessing.sequencer import create_sequences
    X_seq_tr, y_seq_tr = create_sequences(X_tr, y_tr, WINDOW_SIZE, STEP_SIZE)
    X_seq_v,  y_seq_v  = create_sequences(X_v,  y_v,  WINDOW_SIZE, STEP_SIZE)
    X_seq_te, y_seq_te = create_sequences(X_te, y_te, WINDOW_SIZE, STEP_SIZE)

    y_oh_tr = to_onehot(y_seq_tr, N_CLASSES)
    y_oh_v  = to_onehot(y_seq_v,  N_CLASSES)
    y_oh_te = to_onehot(y_seq_te, N_CLASSES)

    logger.info(f"Sequences — Train: {X_seq_tr.shape} | Val: {X_seq_v.shape} | Test: {X_seq_te.shape}")

    # 4. Build + train models
    from models.cnn_gru       import build_cnn_gru
    from models.lstm_baseline import build_lstm
    from models.mlp_baseline  import build_mlp
    from training.trainer     import train_model, compute_class_weights

    class_weight = compute_class_weights(y_seq_tr, N_CLASSES)
    trained = {}

    for build_fn, name, kwargs in [
        (build_cnn_gru,  "CNN-GRU",
         dict(n_timesteps=WINDOW_SIZE, n_features=N_FEATURES, n_classes=N_CLASSES,
              cnn_filters=32, cnn_kernel=3, pool_size=2,
              gru_units=32, dense_units=64, dropout_rate=0.2,
              learning_rate=0.001, use_focal_loss=False)),
        (build_lstm, "BiLSTM",
         dict(n_timesteps=WINDOW_SIZE, n_features=N_FEATURES, n_classes=N_CLASSES,
              lstm_units=64, dense_units=32, dropout_rate=0.2, learning_rate=0.001)),
        (build_mlp, "MLP",
         dict(n_timesteps=WINDOW_SIZE, n_features=N_FEATURES, n_classes=N_CLASSES,
              hidden_units=[128, 64], dropout_rate=0.2, learning_rate=0.001)),
    ]:
        model = build_fn(**kwargs)
        hist, t = train_model(
            model, X_seq_tr, y_oh_tr, X_seq_v, y_oh_v,
            model_name=f"demo_{name.replace('-','_')}",
            save_dir=MODELS_DIR,
            batch_size=BATCH_SIZE, epochs=EPOCHS,
            early_stop_patience=3, lr_reduce_patience=2,
            class_weight=class_weight,
        )
        trained[name] = (model, hist, t)

    # 5. Evaluate
    from evaluation.metrics    import evaluate_model, compare_models
    from evaluation.efficiency import full_efficiency_report

    all_results = {}
    train_times = {}
    for name, (model, hist, t) in trained.items():
        res = evaluate_model(model, X_seq_te, y_oh_te, class_names, batch_size=BATCH_SIZE)
        all_results[name] = res
        train_times[name] = t
        plot_training_curves(hist, model_name=f"demo_{name}", save_dir=PLOTS_DIR)
        plot_confusion_matrix(
            np.array(res["confusion_matrix"]), class_names,
            model_name=f"demo_{name}", save_dir=PLOTS_DIR
        )

    compare_models(all_results)
    plot_model_comparison(all_results, save_dir=PLOTS_DIR)
    plot_class_f1(all_results, class_names=class_names, save_dir=PLOTS_DIR)

    models_dict = {n: m for n, (m, _, __) in trained.items()}
    eff = full_efficiency_report(models_dict, X_seq_te, train_times)
    plot_efficiency_comparison(eff, save_dir=PLOTS_DIR)

    # 6. Mini SHAP (fast)
    try:
        from explainability.shap_explainer import (
            compute_shap_values, plot_shap_summary, plot_shap_per_class
        )
        cnn_gru_model = trained["CNN-GRU"][0]
        shap_vals = compute_shap_values(
            cnn_gru_model,
            X_background=X_seq_tr[:30],
            X_explain=X_seq_te[:20],
            n_timesteps=WINDOW_SIZE,
            n_features=N_FEATURES,
            n_background=20,
        )
        plot_shap_summary(
            shap_vals, X_seq_te[:20], feature_names,
            n_timesteps=WINDOW_SIZE, n_features=N_FEATURES,
            save_dir=PLOTS_DIR, top_n=10,
        )
        plot_shap_per_class(
            shap_vals, X_seq_te[:20], y_true=None,
            feature_names=feature_names, class_names=class_names,
            n_timesteps=WINDOW_SIZE, n_features=N_FEATURES,
            save_dir=PLOTS_DIR, target_classes=[1, 2, 3], top_n=8,
        )
    except Exception as e:
        logger.warning(f"SHAP skipped: {e}")

    # 7. Ablation
    from models.ablation_models import build_cnn_only, build_gru_only
    abl_trained = {}
    for build_fn, name, kwargs in [
        (build_cnn_only, "CNN-Only",
         dict(n_timesteps=WINDOW_SIZE, n_features=N_FEATURES, n_classes=N_CLASSES,
              cnn_filters=32, cnn_kernel=3, dense_units=64, dropout_rate=0.2, learning_rate=0.001)),
        (build_gru_only, "GRU-Only",
         dict(n_timesteps=WINDOW_SIZE, n_features=N_FEATURES, n_classes=N_CLASSES,
              gru_units=64, dense_units=32, dropout_rate=0.2, learning_rate=0.001)),
    ]:
        m = build_fn(**kwargs)
        h, t = train_model(
            m, X_seq_tr, y_oh_tr, X_seq_v, y_oh_v,
            model_name=f"demo_{name.replace('-','_')}",
            save_dir=MODELS_DIR,
            batch_size=BATCH_SIZE, epochs=EPOCHS,
            early_stop_patience=3, lr_reduce_patience=2,
            class_weight=class_weight,
        )
        abl_trained[name] = (m, h, t)

    abl_trained["CNN-GRU"] = trained["CNN-GRU"]
    abl_results = {
        name: evaluate_model(mdl, X_seq_te, y_oh_te, class_names, BATCH_SIZE)
        for name, (mdl, _, __) in abl_trained.items()
    }
    compare_models(abl_results)
    plot_model_comparison(abl_results, save_dir=PLOTS_DIR)

    logger.info("═" * 60)
    logger.info("DEMO COMPLETE")
    logger.info(f"  Plots saved : {PLOTS_DIR}")
    logger.info(f"  Models saved: {MODELS_DIR}")
    logger.info("═" * 60)

    return all_results, abl_results


if __name__ == "__main__":
    run_demo()
