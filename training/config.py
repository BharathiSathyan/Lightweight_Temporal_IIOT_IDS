"""
training/config.py
==================
Central configuration for all hyperparameters and paths.
Edit this file to tune the system without touching model code.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR       = os.path.join(OUTPUTS_DIR, "plots")
MODELS_DIR      = os.path.join(OUTPUTS_DIR, "saved_models")

PARQUET_FILE    = os.path.join(DATA_DIR, "NF-ToN-IoT-V2.parquet")
CSV_FILE        = os.path.join(DATA_DIR, "NF-ToN-IoT-V2.csv")
PROCESSED_FILE  = os.path.join(DATA_DIR, "processed.csv")
SCALER_FILE     = os.path.join(DATA_DIR, "scaler.pkl")
ENCODER_FILE    = os.path.join(DATA_DIR, "label_encoder.pkl")

# ─── Preprocessing ────────────────────────────────────────────────────────────
SAMPLE_SIZE         = 35_000        # Stratified samples (~30K–40K)
RANDOM_SEED         = 42
TEST_SIZE           = 0.15
VALIDATION_SIZE     = 0.15          # of total (after test split)

# Features to drop (identifiers, IPs, timestamps — adjust per dataset)
DROP_FEATURES = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
    "Attack", "Label",                  # kept separately
]

# Target column names (adjust to actual column names in dataset)
LABEL_COL   = "Label"       # binary: 0/1
ATTACK_COL  = "Attack"      # multi-class attack type string

# ─── Sequence (Sliding Window) ────────────────────────────────────────────────
WINDOW_SIZE  = 20           # flows per sequence (timesteps)
STEP_SIZE    = 10           # stride for sliding window

# ─── Model ───────────────────────────────────────────────────────────────────
CNN_FILTERS      = 64       # 1D CNN filters
CNN_KERNEL_SIZE  = 3        # CNN kernel
CNN_POOL_SIZE    = 2        # MaxPool size
GRU_UNITS        = 64       # GRU hidden units
DENSE_UNITS      = 128      # Dense layer size
DROPOUT_RATE     = 0.3      # Dropout regularization

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE        = 64
EPOCHS            = 30
LEARNING_RATE     = 0.001
EARLY_STOP_PAT    = 7       # patience for EarlyStopping
LR_REDUCE_FACTOR  = 0.5
LR_REDUCE_PAT     = 3
MIN_LR            = 1e-6
USE_FOCAL_LOSS    = True    # True = FocalLoss, False = CrossEntropy
FOCAL_GAMMA       = 2.0

# ─── Evaluation ──────────────────────────────────────────────────────────────
TOP_SHAP_FEATURES = 15      # number of features to show in SHAP plots
N_SHAP_SAMPLES    = 200     # samples for SHAP background

# ─── Ensure directories exist ─────────────────────────────────────────────────
for _dir in [DATA_DIR, OUTPUTS_DIR, PLOTS_DIR, MODELS_DIR]:
<<<<<<< HEAD
    os.makedirs(_dir, exist_ok=True)
=======
    os.makedirs(_dir, exist_ok=True)
>>>>>>> de5c81167c17183540ab354e797bd66b1ffbf19b
