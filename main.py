"""
main.py
=======
Full IDS pipeline runner.

Modes (--mode):
  all         Run entire pipeline end-to-end (default)
  preprocess  Data loading, cleaning, encoding, normalizing, sequencing
  train       Train CNN-GRU + baselines
  evaluate    Evaluate all models and generate plots
  explain     Run SHAP explainability
  ablation    Train CNN-only and GRU-only, compare with CNN-GRU

Usage:
  python main.py                  # full pipeline
  python main.py --mode train
  python main.py --mode explain
"""

import argparse
import logging
import os
import json
import numpy as np
import joblib
import tensorflow as tf

# ── Project imports ───────────────────────────────────────────────────────────
from training.config import (
    PARQUET_FILE, CSV_FILE, PROCESSED_FILE, SCALER_FILE, ENCODER_FILE,
    MODELS_DIR, PLOTS_DIR, DATA_DIR,
    DROP_FEATURES, ATTACK_COL, LABEL_COL,
    SAMPLE_SIZE, RANDOM_SEED,
    WINDOW_SIZE, STEP_SIZE,
    TEST_SIZE, VALIDATION_SIZE,
    CNN_FILTERS, CNN_KERNEL_SIZE, CNN_POOL_SIZE,
    GRU_UNITS, DENSE_UNITS, DROPOUT_RATE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    EARLY_STOP_PAT, LR_REDUCE_FACTOR, LR_REDUCE_PAT, MIN_LR,
    USE_FOCAL_LOSS, FOCAL_GAMMA,
    TOP_SHAP_FEATURES, N_SHAP_SAMPLES,
)
from preprocessing.loader    import (load_parquet_to_csv, load_csv,
                                      remove_duplicates, drop_irrelevant_features,
                                      inspect_dataset, handle_missing_values)
from preprocessing.encoder   import encode_target, encode_categorical_features, get_feature_columns
from preprocessing.normalizer import fit_scaler, apply_scaler, load_scaler
from preprocessing.sequencer import create_sequences, stratified_sample

from models.cnn_gru          import build_cnn_gru
from models.lstm_baseline    import build_lstm
from models.mlp_baseline     import build_mlp
from models.ablation_models  import build_cnn_only, build_gru_only

from training.trainer        import train_model, compute_class_weights

from evaluation.metrics      import evaluate_model, compare_models
from evaluation.efficiency   import full_efficiency_report

from utils.helpers           import set_seeds, setup_logging, split_data, to_onehot, save_json
from utils.visualizer        import (plot_training_curves, plot_confusion_matrix,
                                      plot_model_comparison, plot_efficiency_comparison,
                                      plot_class_f1)
from models.cnn_gru import focal_loss

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def run_preprocessing() -> tuple:
    """
    Full preprocessing pipeline.

    Returns:
        (X_seq_train, X_seq_val, X_seq_test,
         y_int_train, y_int_val, y_int_test,
         y_oh_train,  y_oh_val,  y_oh_test,
         feature_names, class_names, n_classes)
    """
    logger.info("═" * 60)
    logger.info("STEP 1: PREPROCESSING")
    logger.info("═" * 60)

    # ── 1a. Load / Convert ────────────────────────────────────────────────────
    if os.path.exists(PROCESSED_FILE):
        logger.info("Found processed file — loading directly.")
        df = load_csv(PROCESSED_FILE)
        le = joblib.load(ENCODER_FILE)
        class_names = list(le.classes_)
    else:
        # Load parquet or CSV
        if os.path.exists(PARQUET_FILE):
            df = load_parquet_to_csv(PARQUET_FILE, CSV_FILE)
        elif os.path.exists(CSV_FILE):
            df = load_csv(CSV_FILE)
        else:
            raise FileNotFoundError(
                f"Dataset not found.\n"
                f"  Expected parquet: {PARQUET_FILE}\n"
                f"  Or CSV:           {CSV_FILE}\n"
                f"  Place NF-ToN-IoT-V2.parquet (or .csv) in the data/ directory."
            )

        inspect_dataset(df)

        # ── 1b. Clean ─────────────────────────────────────────────────────────
        df = remove_duplicates(df)
        df = handle_missing_values(df)

        # ── 1c. Stratified sampling ───────────────────────────────────────────
        # Require the multi-class attack column for stratification
        if ATTACK_COL not in df.columns and LABEL_COL in df.columns:
            logger.warning(
                f"Column '{ATTACK_COL}' not found — using '{LABEL_COL}' as target."
            )
            df[ATTACK_COL] = df[LABEL_COL].astype(str)

        df = stratified_sample(df, target_col=ATTACK_COL,
                               n_samples=SAMPLE_SIZE, random_seed=RANDOM_SEED)

        # ── 1d. Encode target ─────────────────────────────────────────────────
        df, le, class_names = encode_target(df, target_col=ATTACK_COL,
                                            encoder_path=ENCODER_FILE)

        # Drop irrelevant features (keep encoded target)
        keep_target = ATTACK_COL
        drop_cols_actual = [c for c in DROP_FEATURES if c != keep_target]
        df = drop_irrelevant_features(df, drop_cols=drop_cols_actual)

        # ── 1e. Encode categoricals ───────────────────────────────────────────
        df = encode_categorical_features(df, exclude_cols=[ATTACK_COL])

        # Save processed dataset
        df.to_csv(PROCESSED_FILE, index=False)
        logger.info(f"Processed dataset saved → {PROCESSED_FILE}")


    # ── 1f. Feature / label split ─────────────────────────────────────────────
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(df.median(numeric_only=True))
    feature_names = get_feature_columns(df, target_col=ATTACK_COL)
    df[feature_names] = df[feature_names].clip(-1e10, 1e10)
    X_raw = df[feature_names].values.astype(np.float64)
    y_raw = df[ATTACK_COL].values.astype(np.int32)
    n_classes = len(class_names)
  

    logger.info(f"Features: {len(feature_names)} | Classes: {n_classes} | Samples: {len(X_raw):,}")

    # ── 1g. Train/val/test split (tabular, before sequencing) ────────────────
    X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(
        X_raw, y_raw,
        test_size=TEST_SIZE,
        val_size=VALIDATION_SIZE,
        random_seed=RANDOM_SEED,
    )
    
    # ── 1h. Normalize — fit on train only ─────────────────────────────────────
    if os.path.exists(SCALER_FILE):
        scaler = load_scaler(SCALER_FILE)
    else:
        scaler = fit_scaler(X_tr, SCALER_FILE)

    X_tr = apply_scaler(X_tr, scaler)
    X_v  = apply_scaler(X_v,  scaler)
    X_te = apply_scaler(X_te, scaler)

    # ── 1i. Sliding window sequencing ─────────────────────────────────────────
    logger.info("Creating sliding window sequences …")
    X_seq_tr, y_seq_tr = create_sequences(X_tr, y_tr, WINDOW_SIZE, STEP_SIZE)
    X_seq_v,  y_seq_v  = create_sequences(X_v,  y_v,  WINDOW_SIZE, STEP_SIZE)
    X_seq_te, y_seq_te = create_sequences(X_te, y_te, WINDOW_SIZE, STEP_SIZE)

    # ── 1j. One-hot labels for training ──────────────────────────────────────
    y_oh_tr = to_onehot(y_seq_tr, n_classes)
    y_oh_v  = to_onehot(y_seq_v,  n_classes)
    y_oh_te = to_onehot(y_seq_te, n_classes)

    logger.info(
        f"Final shapes → "
        f"Train: {X_seq_tr.shape} | Val: {X_seq_v.shape} | Test: {X_seq_te.shape}"
    )

    return (X_seq_tr, X_seq_v, X_seq_te,
            y_seq_tr, y_seq_v, y_seq_te,
            y_oh_tr,  y_oh_v,  y_oh_te,
            feature_names, class_names, n_classes)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def run_training(
    X_tr, X_v, y_oh_tr, y_oh_v,
    y_int_tr, n_classes, n_features,
) -> dict:
    """
    Train CNN-GRU + LSTM + MLP.

    Returns:
        {model_name: (keras_model, history, training_time_s)}
    """
    logger.info("═" * 60)
    logger.info("STEP 2: TRAINING")
    logger.info("═" * 60)

    n_timesteps = X_tr.shape[1]

    # Compute class weights for imbalance handling
    class_weight = compute_class_weights(y_int_tr, n_classes)

    trained = {}

    # ── CNN-GRU (main model) ──────────────────────────────────────────────────
    model_cnn_gru = build_cnn_gru(
        n_timesteps=n_timesteps, n_features=n_features,
        n_classes=n_classes,
        cnn_filters=CNN_FILTERS, cnn_kernel=CNN_KERNEL_SIZE,
        pool_size=CNN_POOL_SIZE,
        gru_units=GRU_UNITS, dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
        use_focal_loss=USE_FOCAL_LOSS, focal_gamma=FOCAL_GAMMA,
    )
    hist_cnn_gru, t_cnn_gru = train_model(
        model_cnn_gru, X_tr, y_oh_tr, X_v, y_oh_v,
        model_name="CNN_GRU_IDS", save_dir=MODELS_DIR,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PAT,
        lr_reduce_patience=LR_REDUCE_PAT,
        lr_reduce_factor=LR_REDUCE_FACTOR, min_lr=MIN_LR,
        class_weight=class_weight,
    )
    trained["CNN-GRU"] = (model_cnn_gru, hist_cnn_gru, t_cnn_gru)

    # ── BiLSTM baseline ───────────────────────────────────────────────────────
    model_lstm = build_lstm(
        n_timesteps=n_timesteps, n_features=n_features,
        n_classes=n_classes,
        lstm_units=128, dense_units=64,
        dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE,
    )
    hist_lstm, t_lstm = train_model(
        model_lstm, X_tr, y_oh_tr, X_v, y_oh_v,
        model_name="BiLSTM_Baseline", save_dir=MODELS_DIR,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PAT,
        lr_reduce_patience=LR_REDUCE_PAT,
        lr_reduce_factor=LR_REDUCE_FACTOR, min_lr=MIN_LR,
        class_weight=class_weight,
    )
    trained["BiLSTM"] = (model_lstm, hist_lstm, t_lstm)

    # ── MLP baseline ──────────────────────────────────────────────────────────
    model_mlp = build_mlp(
        n_timesteps=n_timesteps, n_features=n_features,
        n_classes=n_classes,
        hidden_units=[256, 128, 64],
        dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE,
    )
    hist_mlp, t_mlp = train_model(
        model_mlp, X_tr, y_oh_tr, X_v, y_oh_v,
        model_name="MLP_Baseline", save_dir=MODELS_DIR,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PAT,
        lr_reduce_patience=LR_REDUCE_PAT,
        lr_reduce_factor=LR_REDUCE_FACTOR, min_lr=MIN_LR,
        class_weight=class_weight,
    )
    trained["MLP"] = (model_mlp, hist_mlp, t_mlp)

    return trained


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    trained_models: dict,
    X_te, y_oh_te, class_names,
) -> dict:
    """
    Evaluate all models, generate plots.

    Returns:
        {model_name: metrics_dict}
    """
    logger.info("═" * 60)
    logger.info("STEP 3: EVALUATION")
    logger.info("═" * 60)

    all_results  = {}
    train_times  = {}

    for name, (model, history, train_time) in trained_models.items():
        logger.info(f"\n── Evaluating: {name} ─────────────────────────────")

        results = evaluate_model(model, X_te, y_oh_te, class_names, batch_size=BATCH_SIZE)
        all_results[name] = results
        train_times[name] = train_time

        # Training curves
        history_path = os.path.join(MODELS_DIR, f"{name.replace('-', '_')}_history.json")

        if os.path.exists(history_path):
            with open(history_path) as f:
                history_data = json.load(f)
            plot_training_curves(history_data, model_name=name, save_dir=PLOTS_DIR)

        # Confusion matrix
        plot_confusion_matrix(
            np.array(results["confusion_matrix"]),
            class_names=class_names,
            model_name=name,
            save_dir=PLOTS_DIR,
            normalize=True,
        )

    # Comparison plots
    compare_models(all_results)
    plot_model_comparison(all_results, save_dir=PLOTS_DIR)
    plot_class_f1(all_results, class_names=class_names, save_dir=PLOTS_DIR)

    # Efficiency report
    models_dict = {name: mdl for name, (mdl, _, __) in trained_models.items()}
    eff_report  = full_efficiency_report(models_dict, X_te, train_times)
    plot_efficiency_comparison(eff_report, save_dir=PLOTS_DIR)

    # Persist results
    save_json(
        {k: {kk: vv for kk, vv in v.items() if kk not in ("y_true","y_pred","y_pred_proba","confusion_matrix")}
         for k, v in all_results.items()},
        os.path.join(MODELS_DIR, "evaluation_results.json")
    )
    save_json(eff_report, os.path.join(MODELS_DIR, "efficiency_results.json"))

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: EXPLAINABILITY
# ═════════════════════════════════════════════════════════════════════════════

def run_explainability(
    model,
    X_tr, X_te,
    feature_names: list,
    class_names: list,
) -> None:
    """Run SHAP feature importance analysis."""
    logger.info("═" * 60)
    logger.info("STEP 4: EXPLAINABILITY (SHAP)")
    logger.info("═" * 60)

    from explainability.shap_explainer import (
        compute_shap_values, plot_shap_summary, plot_shap_per_class
    )

    n_timesteps = X_tr.shape[1]
    n_features  = X_tr.shape[2]

    # Use a small subset for speed on CPU
    bg_samples  = X_tr[:N_SHAP_SAMPLES]
    exp_samples = X_te[:100]

    shap_values = compute_shap_values(
        model, bg_samples, exp_samples,
        n_timesteps=n_timesteps, n_features=n_features,
        n_background=min(50, N_SHAP_SAMPLES),
    )

    plot_shap_summary(
        shap_values, exp_samples, feature_names,
        n_timesteps=n_timesteps, n_features=n_features,
        save_dir=PLOTS_DIR, top_n=TOP_SHAP_FEATURES,
    )

    # Per-class plots for first 3 attack classes (skip benign = 0 if present)
    n_cls = len(class_names)
    target_cls = list(range(1, min(4, n_cls))) if n_cls > 1 else [0]
    plot_shap_per_class(
        shap_values, exp_samples,
        y_true=None,
        feature_names=feature_names,
        class_names=class_names,
        n_timesteps=n_timesteps, n_features=n_features,
        save_dir=PLOTS_DIR,
        target_classes=target_cls,
        top_n=10,
    )

    logger.info("SHAP analysis complete.")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: ABLATION STUDY
# ═════════════════════════════════════════════════════════════════════════════

def run_ablation(
    X_tr, X_v, X_te,
    y_oh_tr, y_oh_v, y_oh_te,
    y_int_tr, class_names,
) -> None:
    """Train CNN-only and GRU-only, compare with pre-trained CNN-GRU."""
    logger.info("═" * 60)
    logger.info("STEP 5: ABLATION STUDY")
    logger.info("═" * 60)

    n_timesteps = X_tr.shape[1]
    n_features  = X_tr.shape[2]
    n_classes   = len(class_names)
    class_weight = compute_class_weights(y_int_tr, n_classes)

    ablation_trained = {}

    # CNN only
    m_cnn = build_cnn_only(n_timesteps, n_features, n_classes,
                            cnn_filters=CNN_FILTERS, cnn_kernel=CNN_KERNEL_SIZE,
                            dense_units=DENSE_UNITS, dropout_rate=DROPOUT_RATE,
                            learning_rate=LEARNING_RATE)
    h_cnn, t_cnn = train_model(
        m_cnn, X_tr, y_oh_tr, X_v, y_oh_v,
        model_name="CNN_Only", save_dir=MODELS_DIR,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PAT,
        lr_reduce_patience=LR_REDUCE_PAT,
        lr_reduce_factor=LR_REDUCE_FACTOR, min_lr=MIN_LR,
        class_weight=class_weight,
    )
    ablation_trained["CNN-Only"] = (m_cnn, h_cnn, t_cnn)

    # GRU only
    m_gru = build_gru_only(n_timesteps, n_features, n_classes,
                            gru_units=GRU_UNITS * 2, dense_units=DENSE_UNITS,
                            dropout_rate=DROPOUT_RATE, learning_rate=LEARNING_RATE)
    h_gru, t_gru = train_model(
        m_gru, X_tr, y_oh_tr, X_v, y_oh_v,
        model_name="GRU_Only", save_dir=MODELS_DIR,
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        early_stop_patience=EARLY_STOP_PAT,
        lr_reduce_patience=LR_REDUCE_PAT,
        lr_reduce_factor=LR_REDUCE_FACTOR, min_lr=MIN_LR,
        class_weight=class_weight,
    )
    ablation_trained["GRU-Only"] = (m_gru, h_gru, t_gru)

    # Load pre-trained CNN-GRU if exists
    cnn_gru_path = os.path.join(MODELS_DIR, "CNN_GRU_IDS.keras")
    if os.path.exists(cnn_gru_path):
        # m_hybrid = tf.keras.models.load_model(cnn_gru_path)
        try:
            m_hybrid = tf.keras.models.load_model(
                cnn_gru_path,
                custom_objects={"focal_loss": focal_loss(gamma=2.0)}
            )
        except:
            m_hybrid = tf.keras.models.load_model(cnn_gru_path, compile=False)
        ablation_trained["CNN-GRU (Full)"] = (m_hybrid, None, None)

    # Evaluate all ablation models
    ablation_results = {}
    for name, (model, history, train_time) in ablation_trained.items():
        results = evaluate_model(model, X_te, y_oh_te, class_names, batch_size=BATCH_SIZE)
        ablation_results[name] = results
        if history:
            plot_training_curves(history, model_name=name.replace(" ", "_"), save_dir=PLOTS_DIR)

    compare_models(ablation_results)
    plot_model_comparison(ablation_results, save_dir=PLOTS_DIR)
    plot_class_f1(ablation_results, class_names=class_names, save_dir=PLOTS_DIR)
    save_json(
        {k: {kk: vv for kk, vv in v.items() if kk not in ("y_true","y_pred","y_pred_proba","confusion_matrix")}
         for k, v in ablation_results.items()},
        os.path.join(MODELS_DIR, "ablation_results.json")
    )

    logger.info("Ablation study complete. Results saved.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IDS Pipeline for NF-ToN-IoT-V2")
    parser.add_argument(
        "--mode",
        choices=["all", "preprocess", "train", "evaluate", "explain", "ablation"],
        default="all",
        help="Pipeline stage to run (default: all)"
    )
    args = parser.parse_args()

    setup_logging()
    set_seeds(RANDOM_SEED)

    # Limit TF threads for clean CPU operation
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║  Industrial IoT Intrusion Detection System (IDS)     ║")
    logger.info("║  NF-ToN-IoT-V2 | CNN-GRU Hybrid | CPU-Optimized     ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info(f"Mode: {args.mode}")

    mode = args.mode

    # ── Preprocess ─────────────────────────────────────────────────────────
    if mode in ("all", "preprocess", "train", "evaluate", "explain", "ablation"):
        (X_tr, X_v, X_te,
         y_tr, y_v, y_te,
         y_oh_tr, y_oh_v, y_oh_te,
         feature_names, class_names, n_classes) = run_preprocessing()

    if mode == "preprocess":
        logger.info("Preprocessing complete. Exiting.")
        return

    # ── Train ──────────────────────────────────────────────────────────────
    trained_models = None
    if mode in ("all", "train"):
        trained_models = run_training(
            X_tr, X_v, y_oh_tr, y_oh_v,
            y_tr, n_classes, X_tr.shape[2],
        )

    if mode == "train":
        logger.info("Training complete. Exiting.")
        return

    # Load pre-trained models if not freshly trained
    if trained_models is None:
        trained_models = {}
        for mname, fname in [
            ("CNN-GRU", "CNN_GRU_IDS.keras"),
            ("BiLSTM",  "BiLSTM_Baseline.keras"),
            ("MLP",     "MLP_Baseline.keras"),
        ]:
            p = os.path.join(MODELS_DIR, fname)
            if os.path.exists(p):                
                if mname == "CNN-GRU":
                    trained_models[mname] = (
                        tf.keras.models.load_model(
                            p,
                            custom_objects={
                                "focal_loss": focal_loss(gamma=2.0)  # IMPORTANT
                            }
                        ),
                        None,
                        None
                    )
                else:
                    trained_models[mname] = (
                        tf.keras.models.load_model(p),
                        None,
                        None
                    )
                logger.info(f"Loaded model: {mname} from {p}")
            else:
                logger.warning(f"Model not found: {p} — run --mode train first.")

    # ── Evaluate ───────────────────────────────────────────────────────────
    if mode in ("all", "evaluate"):
        run_evaluation(trained_models, X_te, y_oh_te, class_names)

    if mode == "evaluate":
        return

    # ── Explain ────────────────────────────────────────────────────────────
    if mode in ("all", "explain"):
        cnn_gru_model = trained_models.get("CNN-GRU", (None,))[0]
        if cnn_gru_model is not None:
            run_explainability(
                cnn_gru_model, X_tr, X_te,
                feature_names, class_names,
            )
        else:
            logger.warning("CNN-GRU model not available for SHAP.")

    if mode == "explain":
        return

    # ── Ablation ───────────────────────────────────────────────────────────
    if mode in ("all", "ablation"):
        run_ablation(
            X_tr, X_v, X_te,
            y_oh_tr, y_oh_v, y_oh_te,
            y_tr, class_names,
        )

    logger.info("═" * 60)
    logger.info("ALL DONE. Outputs saved to:")
    logger.info(f"  Models : {MODELS_DIR}")
    logger.info(f"  Plots  : {PLOTS_DIR}")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
