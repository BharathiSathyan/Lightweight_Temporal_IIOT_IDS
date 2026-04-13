"""
main.py — Industrial IoT Intrusion Detection System (IDS)
Dataset: NF-ToN-IoT-V2
Pipeline: Load → Encode → Normalize → Sequence → Train → Evaluate → Explain → Save
"""

import os
import random
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ── Project-local imports ──────────────────────────────────────────────────────
from preprocessing.loader     import DataLoader         as IdsLoader
from preprocessing.encoder    import CategoricalEncoder
from preprocessing.normalizer import FeatureNormalizer
from preprocessing.sequencer  import SlidingWindowSequencer

from models.cnn_gru       import CnnGruModel
from models.lstm_baseline import LstmBaseline
from models.mlp_baseline  import MlpBaseline

from training.trainer import Trainer
from training.config  import TrainingConfig

from evaluation.metrics    import evaluate_model
from evaluation.efficiency import measure_efficiency

from explainability.shap_explainer import ShapExplainer
from explainability.gradcam        import generate_gradcam, GradCamNotApplicableWarning

from utils.visualizer import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
    plot_feature_importance,
    plot_efficiency_comparison,
)
from utils.helpers import save_json, ensure_dir

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join("data", "NF-ToN-IoT-V2.parquet")   # adjust extension if CSV
OUTPUT_DIR   = os.path.join("outputs", "plots")
METRICS_DIR  = os.path.join("outputs", "metrics")

ensure_dir(OUTPUT_DIR)
ensure_dir(METRICS_DIR)

# ── Config ─────────────────────────────────────────────────────────────────────
CFG = TrainingConfig(
    epochs        = 30,
    batch_size    = 256,
    learning_rate = 1e-3,
    window_size   = 10,       # sliding-window sequence length
    stride        = 1,
    val_split     = 0.15,
    test_split    = 0.15,
    device        = "cpu",    # CPU-only execution
    seed          = SEED,
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load raw dataset
# ══════════════════════════════════════════════════════════════════════════════
def load_data(cfg: TrainingConfig):
    print("\n[1/9] Loading dataset …")
    loader = IdsLoader(data_path=DATA_PATH, seed=cfg.seed)
    df = loader.load()
    print(f"      Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Encode + Normalise + Sequence
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df, cfg: TrainingConfig):
    print("\n[2/9] Encoding categorical features …")
    encoder = CategoricalEncoder()
    df = encoder.fit_transform(df)

    print("      Normalising numeric features …")
    normalizer = FeatureNormalizer()
    df = normalizer.fit_transform(df)

    print(f"      Building sliding-window sequences "
          f"(window={cfg.window_size}, stride={cfg.stride}) …")
    sequencer = SlidingWindowSequencer(
        window_size = cfg.window_size,
        stride      = cfg.stride,
        label_col   = "label",          # adjust to actual target column name
    )
    X, y = sequencer.transform(df)
    print(f"      Sequences: X={X.shape}, y={y.shape}")
    return X, y, encoder, normalizer

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train / Val / Test split (stratified)
# ══════════════════════════════════════════════════════════════════════════════
def split_data(X, y, cfg: TrainingConfig):
    from sklearn.model_selection import train_test_split

    print("\n[3/9] Splitting data (stratified) …")
    test_ratio = cfg.test_split
    val_ratio  = cfg.val_split / (1.0 - test_ratio)   # relative to remaining

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=cfg.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio,
        stratify=y_train_val, random_state=cfg.seed
    )
    print(f"      Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build models
# ══════════════════════════════════════════════════════════════════════════════
def build_models(input_shape, num_classes: int, cfg: TrainingConfig):
    """
    input_shape : (window_size, n_features)
    """
    print("\n[4/9] Initialising models …")
    seq_len, n_feat = input_shape

    models = {
        "CNN-GRU":  CnnGruModel(seq_len=seq_len, n_features=n_feat,
                                num_classes=num_classes),
        "LSTM":     LstmBaseline(seq_len=seq_len, n_features=n_feat,
                                 num_classes=num_classes),
        "MLP":      MlpBaseline(n_features=seq_len * n_feat,
                                num_classes=num_classes),
    }
    for name, m in models.items():
        total = sum(p.numel() for p in m.parameters())
        print(f"      {name:10s}  params={total:,}")
    return models

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train
# ══════════════════════════════════════════════════════════════════════════════
def train_all(models: dict, train_data, val_data, cfg: TrainingConfig):
    print("\n[5/9] Training models …")
    histories = {}
    trained   = {}

    X_train, y_train = train_data
    X_val,   y_val   = val_data

    for name, model in models.items():
        print(f"\n  ── {name} ──")
        trainer = Trainer(model=model, config=cfg)
        history = trainer.fit(
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
        )
        histories[name] = history
        trained[name]   = trainer.model   # retrieve trained weights

        # Plot and save per-model training curve
        fig_path = os.path.join(OUTPUT_DIR, f"training_history_{name}.png")
        plot_training_history(history, title=f"{name} Training History",
                              save_path=fig_path)

    return trained, histories

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_all(trained: dict, test_data, cfg: TrainingConfig):
    print("\n[6/9] Evaluating models …")
    X_test, y_test = test_data
    all_metrics = {}

    for name, model in trained.items():
        print(f"  ── {name} ──")
        metrics = evaluate_model(
            model     = model,
            X_test    = X_test,
            y_test    = y_test,
            device    = cfg.device,
            model_name= name,
        )
        all_metrics[name] = metrics
        print(f"     Accuracy={metrics['accuracy']:.4f}  "
              f"F1={metrics['f1_macro']:.4f}  "
              f"AUC={metrics.get('auc_macro', 'N/A')}")

        # Confusion matrix
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png")
        plot_confusion_matrix(metrics["confusion_matrix"],
                              title=f"{name} — Confusion Matrix",
                              save_path=cm_path)

    # Combined ROC curve
    roc_path = os.path.join(OUTPUT_DIR, "roc_curves_all_models.png")
    plot_roc_curves(all_metrics, save_path=roc_path)

    # Persist metrics to JSON
    save_json(all_metrics, os.path.join(METRICS_DIR, "evaluation_metrics.json"))
    return all_metrics

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Efficiency benchmarking
# ══════════════════════════════════════════════════════════════════════════════
def benchmark_efficiency(trained: dict, test_data, cfg: TrainingConfig):
    print("\n[7/9] Measuring inference efficiency …")
    X_test, _ = test_data
    eff_results = {}

    for name, model in trained.items():
        eff = measure_efficiency(
            model      = model,
            X_sample   = X_test[:512],    # small representative batch
            device     = cfg.device,
            model_name = name,
        )
        eff_results[name] = eff
        print(f"  {name:10s}  "
              f"latency={eff['avg_latency_ms']:.2f} ms  "
              f"throughput={eff['throughput_samples_per_sec']:.0f} sps  "
              f"params={eff['num_parameters']:,}")

    eff_path = os.path.join(OUTPUT_DIR, "efficiency_comparison.png")
    plot_efficiency_comparison(eff_results, save_path=eff_path)
    save_json(eff_results, os.path.join(METRICS_DIR, "efficiency_metrics.json"))
    return eff_results

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SHAP explainability (primary model: CNN-GRU)
# ══════════════════════════════════════════════════════════════════════════════
def run_shap(trained: dict, test_data, feature_names: list, cfg: TrainingConfig):
    print("\n[8/9] Running SHAP explainability on CNN-GRU …")
    X_test, y_test = test_data
    model = trained["CNN-GRU"]

    explainer = ShapExplainer(model=model, device=cfg.device)
    shap_values = explainer.explain(
        X_background = X_test[:200],   # background reference samples
        X_explain    = X_test[:100],   # samples to explain
    )

    shap_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.png")
    plot_feature_importance(
        shap_values  = shap_values,
        feature_names= feature_names,
        title        = "SHAP Feature Importance — CNN-GRU",
        save_path    = shap_path,
    )
    print(f"      SHAP plot saved → {shap_path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Grad-CAM (CNN-GRU; falls back gracefully for tabular data)
# ══════════════════════════════════════════════════════════════════════════════
def run_gradcam(trained: dict, test_data, cfg: TrainingConfig):
    print("\n[9/9] Attempting Grad-CAM on CNN-GRU …")
    X_test, y_test = test_data
    model = trained["CNN-GRU"]

    # Take a single test sample: shape (1, window_size, n_features)
    sample_tensor = torch.tensor(X_test[:1], dtype=torch.float32)
    target_class  = int(y_test[0])

    try:
        heatmap = generate_gradcam(
            model        = model,
            input_tensor = sample_tensor,
            target_class = target_class,
            device       = cfg.device,
        )
        if heatmap is not None:
            gc_path = os.path.join(OUTPUT_DIR, "gradcam_heatmap_sample0.png")
            # For tabular/sequence data the heatmap is a 1-D importance vector;
            # visualizer handles the bar-chart rendering.
            from utils.visualizer import plot_gradcam_heatmap
            plot_gradcam_heatmap(heatmap, save_path=gc_path,
                                 title="Grad-CAM — CNN-GRU (sample 0)")
            print(f"      Grad-CAM plot saved → {gc_path}")
        else:
            print("      Grad-CAM returned None — skipping plot.")

    except GradCamNotApplicableWarning as w:
        print(f"      [WARNING] {w}")
    except Exception as exc:
        print(f"      [ERROR] Grad-CAM failed: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  Industrial IoT IDS — NF-ToN-IoT-V2 — Full Pipeline")
    print("=" * 65)

    set_seed(SEED)

    # 1. Load
    df = load_data(CFG)

    # 2. Preprocess
    X, y, encoder, normalizer = preprocess(df, CFG)
    feature_names = [f"feat_{i}" for i in range(X.shape[2])]   # fallback names

    # 3. Split
    train_data, val_data, test_data = split_data(X, y, CFG)

    # 4. Build models
    num_classes  = int(y.max()) + 1
    input_shape  = (X.shape[1], X.shape[2])          # (window_size, n_features)
    models       = build_models(input_shape, num_classes, CFG)

    # 5. Train
    trained, histories = train_all(models, train_data, val_data, CFG)

    # 6. Evaluate
    metrics = evaluate_all(trained, test_data, CFG)

    # 7. Efficiency
    eff = benchmark_efficiency(trained, test_data, CFG)

    # 8. SHAP
    run_shap(trained, test_data, feature_names, CFG)

    # 9. Grad-CAM
    run_gradcam(trained, test_data, CFG)

    print("\n" + "=" * 65)
    print("  Pipeline complete.  Outputs saved to:", OUTPUT_DIR)
    print("=" * 65)


if __name__ == "__main__":
    main()