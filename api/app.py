"""
api/app.py
==========
Lightweight Flask REST API for IDS inference.

Endpoints:
  GET  /health         → status check
  POST /predict        → classify a network flow (or sequence)
  POST /predict_batch  → classify multiple flows

Usage:
  python api/app.py

Request body (POST /predict):
  {
    "features": [0.1, 0.5, 0.3, ...]   // flat array of ONE flow's features
    // OR
    "sequence": [[...], [...], ...]     // pre-built (T, F) sequence
  }

Response:
  {
    "prediction": "DDoS",
    "class_index": 3,
    "confidence": 0.94,
    "class_probabilities": {"Benign": 0.02, "DDoS": 0.94, ...}
  }
"""

import os
import sys
import json
import time
import logging
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
import joblib
import tensorflow as tf

from training.config import (
    MODELS_DIR, DATA_DIR, WINDOW_SIZE, SCALER_FILE, ENCODER_FILE
)
<<<<<<< HEAD
from models.cnn_gru import focal_loss
=======
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
MODEL       = None
SCALER      = None
LE          = None
CLASS_NAMES = None


def load_artifacts():
    """Load model, scaler, and label encoder at startup."""
    global MODEL, SCALER, LE, CLASS_NAMES

    model_path  = os.path.join(MODELS_DIR, "CNN_GRU_IDS.keras")
    scaler_path = SCALER_FILE
    encoder_path = ENCODER_FILE

    if os.path.exists(model_path):
<<<<<<< HEAD
        # MODEL = tf.keras.models.load_model(model_path)
        MODEL = tf.keras.models.load_model(model_path, compile=False)
=======
        MODEL = tf.keras.models.load_model(model_path)
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b
        logger.info(f"Model loaded: {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}. Run main.py first.")

    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
        logger.info("Scaler loaded.")

    if os.path.exists(encoder_path):
        LE = joblib.load(encoder_path)
        CLASS_NAMES = list(LE.classes_)
        logger.info(f"Classes: {CLASS_NAMES}")


def preprocess_single_flow(features: list) -> np.ndarray:
    """
    Convert a single flow feature list to a (1, T, F) sequence.
    Pads/repeats to fill WINDOW_SIZE if only one flow is given.
    Applies scaler normalization.
    """
    features = np.array(features, dtype=np.float32).reshape(1, -1)  # (1, F)

    if SCALER is not None:
        features = SCALER.transform(features)
        features = np.clip(features, 0, 1)

    n_features = features.shape[1]
    # Repeat single flow WINDOW_SIZE times to create a sequence
    sequence = np.tile(features, (WINDOW_SIZE, 1))   # (T, F)
    return sequence[np.newaxis, :, :]                 # (1, T, F)


def preprocess_sequence(sequence: list) -> np.ndarray:
    """
    Convert a (T, F) list-of-lists to (1, T, F) numpy array.
    Applies scaler normalization.
    """
    seq = np.array(sequence, dtype=np.float32)       # (T, F)
    if SCALER is not None:
        T, F = seq.shape
        seq_flat = seq.reshape(-1, F)
        seq_flat = SCALER.transform(seq_flat)
        seq_flat = np.clip(seq_flat, 0, 1)
        seq = seq_flat.reshape(T, F)
    return seq[np.newaxis, :, :]                      # (1, T, F)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": MODEL is not None,
        "classes":      CLASS_NAMES,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Run main.py first."}), 503

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Empty request body"}), 400

    t0 = time.perf_counter()

    try:
        if "sequence" in data:
            X = preprocess_sequence(data["sequence"])
        elif "features" in data:
            X = preprocess_single_flow(data["features"])
        else:
            return jsonify({"error": "Provide 'features' or 'sequence' key"}), 400

        proba = MODEL.predict(X, verbose=0)[0]          # (n_classes,)
        cls_idx = int(np.argmax(proba))
        cls_name = CLASS_NAMES[cls_idx] if CLASS_NAMES else str(cls_idx)
        confidence = float(proba[cls_idx])

        class_probs = {}
        if CLASS_NAMES:
            class_probs = {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, proba)}

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return jsonify({
            "prediction":          cls_name,
            "class_index":         cls_idx,
            "confidence":          round(confidence, 4),
            "class_probabilities": class_probs,
            "inference_ms":        elapsed_ms,
        })

    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if MODEL is None:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(force=True)
    sequences = data.get("sequences", [])
    if not sequences:
        return jsonify({"error": "Provide 'sequences' list"}), 400

    X_list = [preprocess_sequence(s)[0] for s in sequences]
    X_batch = np.stack(X_list, axis=0)                  # (N, T, F)

    probas   = MODEL.predict(X_batch, verbose=0)        # (N, n_classes)
    cls_idxs = np.argmax(probas, axis=1)

    results = []
    for i, (cls_idx, proba) in enumerate(zip(cls_idxs, probas)):
        cls_name = CLASS_NAMES[int(cls_idx)] if CLASS_NAMES else str(int(cls_idx))
        results.append({
            "prediction":  cls_name,
            "class_index": int(cls_idx),
            "confidence":  round(float(proba[cls_idx]), 4),
        })

    return jsonify({"predictions": results, "count": len(results)})


if __name__ == "__main__":
    load_artifacts()
<<<<<<< HEAD
    app.run(host="0.0.0.0", port=5000, debug=False)
=======
    app.run(host="0.0.0.0", port=5000, debug=False)
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b
