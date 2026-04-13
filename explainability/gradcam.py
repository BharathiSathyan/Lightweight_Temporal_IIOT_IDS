"""
explainability/gradcam.py
─────────────────────────
Grad-CAM implementation for the CNN-GRU IDS model.

Design note
──────────────
The NF-ToN-IoT-V2 dataset is *tabular* (network-flow features), not images.
Grad-CAM was originally designed for 2-D spatial CNNs (image classification).
Here we adapt it for a 1-D temporal CNN:

  • The "spatial" dimension becomes the *time-step* dimension of the sequence.
  • The resulting heatmap is a 1-D vector of length `window_size`, indicating
    which time-steps were most important for the predicted class.
  • `overlay_heatmap()` stacks the heatmap over the feature matrix so that
    the caller can render a 2-D importance grid (time × feature).

If the model's CNN layers cannot be located automatically, a
`GradCamNotApplicableWarning` is raised instead of crashing the pipeline.

Framework: PyTorch (CPU-only).
"""

import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on CPU servers
import matplotlib.pyplot as plt


# ── Custom warning ─────────────────────────────────────────────────────────────
class GradCamNotApplicableWarning(UserWarning):
    """Raised when Grad-CAM cannot be applied to the given model / input."""


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Walk the module tree and return the *last* Conv1d layer found.
    Returns None if no Conv1d exists.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            last_conv = module
    return last_conv


class _GradCamHooks:
    """
    Attach forward/backward hooks to a target layer and capture
    activations + gradients during a single forward-backward pass.
    """

    def __init__(self, layer: nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None

        self._fwd_hook = layer.register_forward_hook(self._save_activation)
        self._bwd_hook = layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        # output shape for Conv1d: (batch, channels, length)
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        # grad_out[0] shape: (batch, channels, length)
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_gradcam(
    model:        nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    device:       str = "cpu",
) -> Optional[np.ndarray]:
    """
    Compute a Grad-CAM heatmap for a 1-D temporal CNN.

    Parameters
    ----------
    model        : trained PyTorch model (must contain at least one Conv1d).
    input_tensor : shape (1, window_size, n_features) — a single sample.
    target_class : integer class index to explain.
    device       : 'cpu' (default) or 'cuda'.

    Returns
    -------
    heatmap : np.ndarray of shape (window_size,) — per-time-step importance,
              normalised to [0, 1].
              Returns None (with a warning) if Grad-CAM is not applicable.

    Notes
    -----
    • The CNN in the CNN-GRU model operates over the *feature* dimension
      (channels) and the *time* dimension (sequence length).  The resulting
      class-activation map therefore highlights which time-steps contribute
      most to the predicted class.
    • This is **not** the same as image-level Grad-CAM (no spatial grid).
      For true spatial heatmaps an image-based input is required — a warning
      is emitted if the caller passes a 4-D tensor (batch × C × H × W).
    """
    # ── Guard: image-shaped input ──────────────────────────────────────────
    if input_tensor.ndim == 4:
        warnings.warn(
            "Grad-CAM received a 4-D tensor (batch × C × H × W). "
            "This project uses tabular/sequence data (3-D tensors). "
            "Grad-CAM is only directly applicable for image-based CNN input. "
            "For tabular data, SHAP explainability is recommended instead.",
            GradCamNotApplicableWarning,
            stacklevel=2,
        )
        return None

    # ── Guard: no Conv1d found ─────────────────────────────────────────────
    target_layer = _find_last_conv_layer(model)
    if target_layer is None:
        warnings.warn(
            "No Conv1d layer found in the model. "
            "Grad-CAM requires at least one convolutional layer. "
            "Consider using SHAP for this architecture.",
            GradCamNotApplicableWarning,
            stacklevel=2,
        )
        return None

    model = model.to(device).eval()

    # input_tensor : (1, window_size, n_features)
    # Conv1d expects  (batch, channels, length)  →  permute to (1, n_features, window_size)
    x = input_tensor.to(device)
    if x.ndim == 3:
        x = x.permute(0, 2, 1)   # → (1, n_features, window_size)
    x = x.requires_grad_(True)

    hooks = _GradCamHooks(target_layer)

    # ── Forward pass ───────────────────────────────────────────────────────
    # The model may expect (batch, window_size, n_features); feed the
    # *original* layout and rely on the model's internal permute/reshape.
    # We re-create a clean tensor in the original layout for the model call.
    x_model = input_tensor.clone().to(device).requires_grad_(True)

    logits = model(x_model)          # shape: (1, num_classes)

    # ── Backward pass for target class ────────────────────────────────────
    model.zero_grad()
    score = logits[0, target_class]
    score.backward()

    hooks.remove()

    activations = hooks.activations   # (1, C_conv, L)
    gradients   = hooks.gradients     # (1, C_conv, L)

    if activations is None or gradients is None:
        warnings.warn(
            "Grad-CAM hooks did not capture activations / gradients. "
            "The forward pass may not have passed through the Conv1d layer.",
            GradCamNotApplicableWarning,
            stacklevel=2,
        )
        return None

    # ── Class-activation map ───────────────────────────────────────────────
    # Global Average Pool over the length dimension → weights per channel
    weights = gradients.mean(dim=2, keepdim=True)    # (1, C_conv, 1)

    # Weighted sum of feature maps → cam shape (1, L)
    cam = (weights * activations).sum(dim=1)         # (1, L)
    cam = torch.relu(cam)                            # keep positive activations
    cam = cam.squeeze(0).cpu().numpy()               # (L,) == (window_size,)

    # Normalise to [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam   # shape: (window_size,)


def overlay_heatmap(
    original_input: np.ndarray,
    heatmap:        np.ndarray,
) -> np.ndarray:
    """
    Overlay a 1-D Grad-CAM heatmap onto a 2-D tabular sequence.

    Parameters
    ----------
    original_input : np.ndarray, shape (window_size, n_features).
                     The raw (normalised) feature matrix for the sample.
    heatmap        : np.ndarray, shape (window_size,).
                     Output of `generate_gradcam`.

    Returns
    -------
    overlay : np.ndarray, shape (window_size, n_features).
              Each row (time-step) of the feature matrix is scaled by the
              corresponding Grad-CAM importance weight.  The result can be
              rendered as a 2-D heatmap (time × feature).

    Notes
    -----
    For true *image* Grad-CAM overlays (RGB blending with a colourmap), the
    caller should use an image-domain input and a dedicated image library.
    This function handles the tabular / sequence equivalent.
    """
    if original_input.ndim != 2:
        raise ValueError(
            f"original_input must be 2-D (window_size × n_features), "
            f"got shape {original_input.shape}."
        )
    if heatmap.ndim != 1 or heatmap.shape[0] != original_input.shape[0]:
        raise ValueError(
            f"heatmap length ({heatmap.shape[0]}) must match "
            f"original_input window_size ({original_input.shape[0]})."
        )

    # Broadcast heatmap (window_size,) → (window_size, n_features)
    overlay = original_input * heatmap[:, np.newaxis]
    return overlay


def save_gradcam_visualization(
    heatmap:     np.ndarray,
    save_path:   str,
    title:       str = "Grad-CAM — Time-step Importance",
    feature_names: Optional[list] = None,
    original_input: Optional[np.ndarray] = None,
) -> None:
    """
    Save a Grad-CAM visualisation to disk.

    Produces two sub-plots when `original_input` is provided:
      1. Bar chart of per-time-step importance (heatmap).
      2. 2-D heatmap of the overlay (time × feature).

    Parameters
    ----------
    heatmap        : 1-D array of shape (window_size,).
    save_path      : full path including filename, e.g. 'outputs/plots/gradcam.png'.
    title          : figure title.
    feature_names  : optional list of feature names for the y-axis of the overlay.
    original_input : optional 2-D array (window_size, n_features) for overlay plot.
    """
    has_overlay = original_input is not None and original_input.ndim == 2

    n_cols = 2 if has_overlay else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    # ── Plot 1: bar chart ─────────────────────────────────────────────────
    ax = axes[0]
    time_steps = np.arange(len(heatmap))
    bars = ax.bar(time_steps, heatmap, color=plt.cm.RdYlGn(heatmap))  # noqa
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Grad-CAM Importance")
    ax.set_title("Per-time-step Importance")
    ax.set_ylim(0, 1.05)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Importance")

    # ── Plot 2: 2-D overlay ───────────────────────────────────────────────
    if has_overlay:
        overlay = overlay_heatmap(original_input, heatmap)
        ax2 = axes[1]
        im = ax2.imshow(
            overlay.T,           # shape: (n_features, window_size)
            aspect="auto",
            cmap="hot",
            origin="lower",
            interpolation="nearest",
        )
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Feature Index")
        ax2.set_title("Grad-CAM Overlay (time × feature)")
        if feature_names and len(feature_names) == overlay.shape[1]:
            ax2.set_yticks(range(len(feature_names)))
            ax2.set_yticklabels(feature_names, fontsize=6)
        fig.colorbar(im, ax=ax2, label="Activation × Importance")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradcam] Visualisation saved → {save_path}")