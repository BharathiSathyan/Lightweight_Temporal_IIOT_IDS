# Lightweight Temporal Hybrid Intrusion Detection System for IIoT

## Overview

This project presents a lightweight, scalable, and explainable Intrusion Detection System (IDS) for Industrial Internet of Things (IIoT) environments. The system is designed to efficiently detect cyber-attacks such as DDoS, ransomware, injection, and scanning attacks using NetFlow-based traffic data.

Unlike existing approaches that rely on computationally expensive deep transfer learning, image transformations, and ensemble models, this implementation focuses on a hybrid CNN-GRU architecture operating directly on tabular temporal data. The model is optimized for CPU-based environments and real-time deployment.

The work is inspired by the IEEE paper on Adaptive NetFlow-based IIoT IDS, while introducing significant improvements in temporal modeling, computational efficiency, and explainability.

---

## Problem Statement

IIoT networks generate high-dimensional, sequential network traffic data that is difficult to model using traditional IDS techniques. Existing solutions face several challenges:

* Inability to capture temporal dependencies in traffic
* High computational overhead due to deep transfer learning and ensembles
* Poor handling of class imbalance
* Lack of explainability in predictions

This project addresses these limitations by designing a lightweight IDS that is both efficient and interpretable.

---

## Key Contributions

* Lightweight CNN-GRU hybrid architecture for spatial and temporal learning
* Direct processing of tabular NetFlow data without image transformation
* Sliding window-based temporal sequence generation
* CPU-efficient training pipeline without heavy ensembles or genetic optimization
* Integrated SHAP-based explainability for feature importance analysis
* Comparative evaluation with baseline models (LSTM and MLP)
* Performance + efficiency evaluation (accuracy, latency, throughput)

---

## System Architecture

### 1. Data Preprocessing and Temporal Sequencing

Raw NetFlow data is cleaned, encoded, and normalized. A sliding window mechanism converts tabular data into sequential format, enabling temporal learning.

### 2. Hybrid CNN-GRU Model

* CNN layers extract local feature interactions
* GRU layers capture temporal dependencies
* Combined architecture balances performance and efficiency

### 3. Baseline Models

* LSTM for sequence modeling comparison
* MLP for non-temporal baseline

### 4. Lightweight Training Strategy

* No pretrained CNNs
* No genetic algorithm optimization
* Minimal hyperparameter tuning
* Optimized for CPU execution

### 5. Explainability Integration

SHAP (SHapley Additive exPlanations) is used to:

* Identify important features
* Improve model transparency
* Provide interpretability for predictions

### 6. Evaluation Metrics

* Accuracy, Precision, Recall, F1-score
* Confusion Matrix (class-wise analysis)
* Latency and throughput for deployment feasibility

---

## Dataset

* Dataset: NF-ToN-IoT-V2
* Type: Industrial IoT NetFlow dataset
* Size: ~34,000 samples (subset used for efficiency)
* Contains multiple attack categories and normal traffic

---

## Project Structure

```
Lightweight_Temporal_IIOT_IDS/
│
├── main.py                # Entry point for training and evaluation
├── data/                  # Dataset (or loading scripts)
├── models/                # CNN-GRU, LSTM, MLP implementations
├── preprocessing/         # Data cleaning and sequence generation
├── evaluation/            # Metrics and performance analysis
├── explainability/        # SHAP analysis
├── utils/                 # Helper functions
└── README.md
```

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/BharathiSathyan/Lightweight_Temporal_IIOT_IDS.git
cd Lightweight_Temporal_IIOT_IDS
```

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## Usage

Run the main pipeline:

```
python main.py
```

This will:

* Load and preprocess the dataset
* Generate temporal sequences
* Train CNN-GRU and baseline models
* Evaluate performance
* Generate explainability outputs

---

## Results (Expected Outcomes)

* High classification performance across attack types
* Improved detection of temporal attack patterns
* Reduced computational cost compared to transfer learning models
* Faster training and inference suitable for real-time use
* Clear feature importance insights via SHAP

---

## Comparison with Base Paper

This implementation is inspired by the Adaptive NetFlow-based IIoT IDS (IEEE), but introduces key improvements:

| Aspect              | Base Paper                      | This Project                |
| ------------------- | ------------------------------- | --------------------------- |
| Data Representation | Tabular to Image                | Direct Tabular              |
| Model               | Transfer Learning CNN Ensembles | Lightweight CNN-GRU         |
| Optimization        | Genetic Algorithm               | Simplified Training         |
| Temporal Modeling   | Not Explicit                    | GRU-based Temporal Learning |
| Computational Cost  | High                            | Low                         |
| Explainability      | Limited                         | SHAP Integrated             |
| Deployment          | GPU-oriented                    | CPU-friendly                |

---

## Future Scope (Aligned with Base Paper Enhancements)

This project covers and extends several limitations of the base paper while opening new directions:

* Real-time streaming IDS using live NetFlow data
* Online learning and adaptive model updates
* Edge deployment in IIoT devices
* Integration with SIEM systems
* Advanced imbalance handling (focal loss, synthetic data)
* Transformer-based temporal models for further improvement
* Federated learning for distributed IIoT environments
* Automated hyperparameter tuning with lightweight methods

---

## Conclusion

This project demonstrates that efficient and explainable IDS systems can be built without relying on heavy deep learning pipelines. By leveraging a CNN-GRU hybrid model and temporal sequence modeling, the system achieves strong detection performance while remaining lightweight and deployable in real-world IIoT environments.

The integration of explainability further enhances trust and usability, making this approach a practical alternative to computationally expensive IDS frameworks.

---

## References

* Adaptive NetFlow-based IIoT IDS (IEEE)
* NF-ToN-IoT-V2 Dataset
* SHAP: Lundberg and Lee (2017)

