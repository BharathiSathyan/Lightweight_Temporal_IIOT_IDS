# 🔐 Industrial IoT Intrusion Detection System (IDS)
## NF-ToN-IoT-V2 | CNN-GRU Hybrid | CPU-Optimized

---

## 📌 Overview

A production-ready, lightweight IDS for Industrial IoT networks using the NF-ToN-IoT-V2 dataset.
The system uses a **hybrid CNN-GRU architecture** for temporal flow classification with full explainability support.

### Key Features
- 🧠 **Hybrid CNN-GRU model** — captures spatial + temporal patterns
- ⚡ **CPU-optimized** — no GPU required
- 🔍 **SHAP explainability** — interpretable predictions
- 📊 **Full evaluation suite** — confusion matrix, F1, ablation study
- 🚀 **Flask REST API** — real-time inference endpoint
- 📁 **Modular codebase** — research + deployment ready

---

## 📁 Project Structure

```
ids_project/
├── data/                    # Raw & processed datasets
├── preprocessing/
│   ├── loader.py            # Parquet → CSV + cleaning
│   ├── encoder.py           # Feature encoding
│   ├── normalizer.py        # MinMaxScaler
│   └── sequencer.py         # Sliding window sequences
├── models/
│   ├── cnn_gru.py           # Main hybrid model
│   ├── lstm_baseline.py     # LSTM comparison
│   └── mlp_baseline.py      # MLP comparison
├── training/
│   ├── trainer.py           # Training loop + callbacks
│   └── config.py            # Hyperparameters
├── evaluation/
│   ├── metrics.py           # Accuracy, F1, confusion matrix
│   └── efficiency.py        # Time, size, memory analysis
├── explainability/
│   ├── shap_explainer.py    # SHAP feature importance
│   └── gradcam.py           # Grad-CAM (optional CNN-image path)
├── utils/
│   ├── visualizer.py        # All plot generation
│   └── helpers.py           # Utility functions
├── api/
│   └── app.py               # Flask REST API
├── outputs/
│   └── plots/               # Generated visualizations
├── main.py                  # Full pipeline runner
└── requirements.txt
```

---

## 🚀 Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Dataset
```
Place NF-ToN-IoT-V2.parquet in the data/ directory
```

### 3. Run Full Pipeline
```bash
python main.py
```

### 4. Run Individual Modules
```bash
# Preprocess only
python main.py --mode preprocess

# Train only (after preprocessing)
python main.py --mode train

# Evaluate only
python main.py --mode evaluate

# Explain predictions
python main.py --mode explain

# Ablation study
python main.py --mode ablation
```

### 5. Start Flask API
```bash
python api/app.py
# POST http://localhost:5000/predict
# Body: {"features": [...]}
```

---

## 📊 Dataset: NF-ToN-IoT-V2

- **Source**: UNSW Canberra — Network Flow ToN-IoT Version 2
- **Format**: Parquet → converted to CSV
- **Classes**: Benign + multiple attack types
- **Features**: Network flow statistics (bytes, packets, duration, etc.)

---

## 🏗️ Model Architecture

```
Input (tabular sequences)
    │
    ▼
[1D CNN] — Local pattern extraction
    │
    ▼
[GRU Layer] — Temporal dependency modeling
    │
    ▼
[Dense + Dropout] — Classification head
    │
    ▼
Softmax Output (multi-class)
```

---

## 📈 Evaluation

- Accuracy, Precision, Recall, F1-Score (macro + per-class)
- Confusion matrix heatmap
- Training/validation loss & accuracy curves
- Computational efficiency benchmarks
- Ablation: CNN-only vs GRU-only vs CNN+GRU

---

## 🔍 Explainability

- **SHAP TreeExplainer** / **KernelExplainer** for feature importance
- Summary plots and force plots per attack class
- Top-N feature rankings

---

## 🌐 API Usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.5, 0.3, ...]}'
```

Response:
```json
{
  "prediction": "DDoS",
  "confidence": 0.94,
  "class_probabilities": {...}
}
```