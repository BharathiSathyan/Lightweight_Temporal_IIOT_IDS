"""
utils/visualizer.py
===================
All plotting functions:
  - Training curves (loss + accuracy)
  - Confusion matrix heatmap
  - Ablation bar chart
  - Efficiency comparison chart
  - Feature importance bar chart
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4f",
    "axes.labelcolor":  "#d0d4e8",
    "xtick.color":      "#d0d4e8",
    "ytick.color":      "#d0d4e8",
    "text.color":       "#d0d4e8",
    "grid.color":       "#2a2d3f",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

ACCENT_COLORS = ["#5b8dee", "#e05252", "#52c48a", "#e0a852", "#9b59b6", "#1abc9c"]


def plot_training_curves(
    history_or_path,
    model_name: str,
    save_dir: str,
) -> None:
    """
    Plot loss and accuracy curves (train vs. val) for a trained model.

    Args:
        history_or_path: Keras History object OR path to JSON history file
        model_name:      Used in title and filename
        save_dir:        Output directory
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(history_or_path, str):
        with open(history_or_path) as f:
            h = json.load(f)
    else:
        h = history_or_path.history

    epochs = range(1, len(h["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training Curves — {model_name}", fontsize=14, color="#d0d4e8")

    # Loss
    ax1.plot(epochs, h["loss"],     color=ACCENT_COLORS[0], lw=2, label="Train Loss")
    ax1.plot(epochs, h["val_loss"], color=ACCENT_COLORS[1], lw=2, ls="--", label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss")
    ax1.legend(); ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, h["accuracy"],     color=ACCENT_COLORS[0], lw=2, label="Train Acc")
    ax2.plot(epochs, h["val_accuracy"], color=ACCENT_COLORS[1], lw=2, ls="--", label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved training curves: {path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    model_name: str,
    save_dir: str,
    normalize: bool = True,
) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        cm:           Confusion matrix (n_classes, n_classes)
        class_names:  Class labels
        model_name:   Used in title and filename
        save_dir:     Output directory
        normalize:    If True, show row-normalized (recall) percentages
    """
    os.makedirs(save_dir, exist_ok=True)
    cm_arr = np.array(cm)

    if normalize:
        row_sums = cm_arr.sum(axis=1, keepdims=True).astype(float)
        cm_plot = np.where(row_sums == 0, 0, cm_arr / row_sums)
        fmt = ".2f"
        title_suffix = "(Normalized)"
    else:
        cm_plot = cm_arr
        fmt = "d"
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="#2a2d3f",
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name} {title_suffix}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved confusion matrix: {path}")


def plot_model_comparison(
    results_dict: dict,
    save_dir: str,
) -> None:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1 across models.

    Args:
        results_dict: {model_name: {"accuracy": ..., "precision": ..., ...}}
        save_dir:     Output directory
    """
    os.makedirs(save_dir, exist_ok=True)

    metrics = ["accuracy", "precision", "recall", "f1_macro"]
    model_names = list(results_dict.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (name, res) in enumerate(results_dict.items()):
        vals = [res.get(m, 0) for m in metrics]
        bars = ax.bar(
            x + i * width - (len(model_names) - 1) * width / 2,
            vals,
            width,
            label=name,
            color=ACCENT_COLORS[i % len(ACCENT_COLORS)],
            alpha=0.9,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#d0d4e8"
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.grid(True, axis="y")
    plt.tight_layout()

    path = os.path.join(save_dir, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved model comparison: {path}")


def plot_efficiency_comparison(
    efficiency_dict: dict,
    save_dir: str,
) -> None:
    """
    Side-by-side bar charts for Size, Training Time, Inference Latency.
    """
    os.makedirs(save_dir, exist_ok=True)
    names = list(efficiency_dict.keys())
    colors = ACCENT_COLORS[:len(names)]

    def safe_val(d, *keys, default=0):
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d if d != "N/A" else 0

    sizes      = [safe_val(efficiency_dict[n], "size_mb") for n in names]
    train_min  = [safe_val(efficiency_dict[n], "train_time_min") for n in names]
    infer_ms   = [safe_val(efficiency_dict[n], "inference_ms", "per_sample_mean_ms") for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Computational Efficiency Comparison", fontsize=13)

    for ax, vals, title, unit in zip(
        axes,
        [sizes, train_min, infer_ms],
        ["Model Size", "Training Time", "Inference Latency"],
        ["MB", "Minutes", "ms/sample"]
    ):
        bars = ax.bar(names, vals, color=colors, alpha=0.9)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{val:.2f}",
                ha="center", fontsize=8.5, color="#d0d4e8"
            )
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(True, axis="y")

    plt.tight_layout()
    path = os.path.join(save_dir, "efficiency_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved efficiency comparison: {path}")


def plot_class_f1(
    results_dict: dict,
    class_names: list,
    save_dir: str,
) -> None:
    """
    Grouped bar chart of per-class F1 for all compared models.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_classes = len(class_names)
    x = np.arange(n_classes)
    model_names = list(results_dict.keys())
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(max(12, n_classes * 1.2), 6))

    for i, (name, res) in enumerate(results_dict.items()):
        f1_pc = res.get("f1_per_class", {})
        vals = [f1_pc.get(cls, 0) for cls in class_names]
        ax.bar(
            x + i * width - (len(model_names) - 1) * width / 2,
            vals,
            width,
            label=name,
            color=ACCENT_COLORS[i % len(ACCENT_COLORS)],
            alpha=0.9
        )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-Class F1-Score Comparison")
    ax.legend()
    ax.grid(True, axis="y")
    plt.tight_layout()

    path = os.path.join(save_dir, "per_class_f1.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved per-class F1: {path}")