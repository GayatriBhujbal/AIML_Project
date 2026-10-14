# Explainable AI-Based Network Threat Detection and Classification
## Components 1 & 2: Dataset Preprocessing, Feature Selection, and Binary Threat Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-22%20passed-brightgreen.svg)](tests/)
[![Dataset](https://img.shields.io/badge/dataset-CIC--IDS2017-orange.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Model](https://img.shields.io/badge/Random%20Forest-Accuracy%2099.98%25-success.svg)](artifacts/models/)

---

## 1. Project Overview
This repository contains the production-grade implementation of **Component 1 (Data Preprocessing, EDA, Feature Selection)** and **Component 2 (Binary Threat Detection)** for the college mini-project: **Explainable AI-Based Network Threat Detection and Classification**.

### Key Deliverables:
1. **Component 1: Data Engineering & Validation:** Full schema validation across all 8 raw CSVs (2,830,743 records), anomaly cleaning (infs, NaNs, sensor rollovers), stratified 70/15/15 splitting, and systematic feature selection (reducing 78 features to 25 optimal features).
2. **Component 2: Binary Threat Detection:**
   - **Baseline:** Decision Tree classifier for empirical comparison.
   - **Primary Model:** Random Forest classifier (100 trees, balanced class weights, calibrated decision threshold).
   - **Fair Evaluation:** Both models evaluated on the identical held-out Test Set (29,173 flows).
   - **Interfacing:** Unified `ThreatDetectionInferenceEngine` delivering turnkey explainer inputs for Member 4 (SHAP & LIME) and automatic hierarchical traffic routing for Member 3 (XGBoost).

---

## 2. Directory Structure

```
AIML_project/
├── .gitignore                      # Prevents committing large CSVs/archives
├── README.md                       # Main project documentation
├── requirements.txt                # Pinned production dependencies
├── INTEGRATION_GUIDE.md            # Integration contract for Members 2, 3, and 4
│
├── config/
│   ├── __init__.py
│   └── dataset_config.py           # Paths, constants, thresholds, env overrides
│
├── src/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── schema.py               # Canonical 79-column schema, label mappings
│   │   └── loader.py               # Robust CSV loader (chunked, memory-safe)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py              # Infs, NaNs, negatives, and deduplication
│   │   ├── splitter.py             # Stratified, leakage-safe train/val/test split
│   │   └── pipeline.py             # PreprocessingPipeline (fitted on train only)
│   ├── feature_selection/
│   │   ├── __init__.py
│   │   └── selector.py             # Variance, collinearity, and RF importance
│   ├── eda/
│   │   ├── __init__.py
│   │   └── visualizer.py           # Publication-quality plotting routines
│   └── models/
│       ├── __init__.py
│       ├── binary_detector.py      # DecisionTreeBaseline & RandomForestBinaryDetector
│       ├── evaluator.py            # Confusion matrix, ROC, PR curves, latency evaluator
│       └── predictor.py            # Turnkey engine for Member 4 (XAI) and Member 3 (XGBoost)
│
├── scripts/
│   ├── run_eda.py                  # CLI generating full audit report & 5 figures
│   ├── run_feature_selection.py    # CLI running feature selection & validation benchmarking
│   ├── prepare_dataset.py          # CLI fitting and serializing preprocessor.joblib
│   ├── train_binary_models.py      # CLI training DT baseline & RF detector on train set
│   └── evaluate_binary_models.py   # CLI evaluating saved models on test CSVs
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Interactive exploration notebook
│
├── artifacts/
│   ├── schemas/
│   │   ├── feature_schema.json     # Documented feature definitions & dtypes
│   │   ├── label_mapping.json      # Binary, multiclass, and category mappings
│   │   └── selected_features.json  # Final selected 25 features list
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── attack_categories.png
│   │   ├── missing_and_inf_analysis.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_distributions_by_class.png
│   │   ├── feature_importance.png
│   │   ├── binary_roc_curve.png
│   │   ├── binary_precision_recall_curve.png
│   │   └── binary_confusion_matrices.png
│   ├── models/
│   │   ├── preprocessor.joblib           # Pre-fitted preprocessing pipeline artifact
│   │   ├── binary_decision_tree.joblib   # Trained Decision Tree baseline model
│   │   ├── binary_random_forest.joblib   # Trained Random Forest detector model
│   │   └── inference_samples.json       # Raw sample flows for integration testing
│   └── reports/
│       ├── dataset_quality_report.md
│       ├── feature_selection_report.json
│       ├── binary_model_evaluation_report.md
│       └── binary_model_evaluation.json
│
└── tests/
    ├── __init__.py
    ├── test_schema.py              # Schema and label encoding tests
    ├── test_cleaner.py             # Cleaning, inf, negative, and dedup tests
    ├── test_splitter.py            # Stratification and zero-leakage tests
    ├── test_pipeline.py            # Fit on train only, transform, and joblib tests
    ├── test_binary_models.py       # Decision Tree & Random Forest unit tests
    └── test_predictor_interface.py # Integration tests for Member 4 and Member 3 contracts
```
└── tests/
    ├── __init__.py
    ├── test_schema.py              # Schema and label encoding tests
    ├── test_cleaner.py             # Cleaning, inf, negative, and dedup tests
    ├── test_splitter.py            # Stratification and zero-leakage tests
    └── test_pipeline.py            # Fit on train only, transform, and joblib tests
```

---

## 3. Dataset Acquisition & Setup Instructions

The dataset used is the **CIC-IDS2017** dataset from the Canadian Institute for Cybersecurity (UNB):

1. **Obtaining the Dataset:**
   - Download `MachineLearningCSV.zip` from the official [CIC-IDS2017 Portal](https://www.unb.ca/cic/datasets/ids-2017.html).
2. **Placement:**
   - Extract the 8 CSV files into the `MachineLearningCVE/` folder in the project root:
     - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
     - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
     - `Friday-WorkingHours-Morning.pcap_ISCX.csv`
     - `Monday-WorkingHours.pcap_ISCX.csv`
     - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
     - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
     - `Tuesday-WorkingHours.pcap_ISCX.csv`
     - `Wednesday-workingHours.pcap_ISCX.csv`
3. **Environment Configuration (Optional):**
   - Override default directory using environment variable:
     ```bash
     export CIC_DATA_DIR="/path/to/MachineLearningCVE"
     ```

---

## 4. Environment Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m pytest -v
```
*(All 14 unit tests pass verifying schema validation, anomaly cleaning, leak-free splitting, and pipeline serialization).*

### 3. Generate Exploratory Data Analysis & Quality Report
```bash
python scripts/run_eda.py
```
Outputs figures into `artifacts/figures/` and comprehensive report to `artifacts/reports/dataset_quality_report.md`.

### 4. Execute Feature Selection Pipeline
```bash
python scripts/run_feature_selection.py
```
Performs collinearity pruning and Random Forest importance ranking on training data, validates F1 improvement, and saves `artifacts/schemas/selected_features.json`.

### 5. Fit & Export Production Preprocessing Pipeline
```bash
python scripts/prepare_dataset.py --sample-per-file 30000
```
Fits `PreprocessingPipeline` strictly on training data and serializes `artifacts/models/preprocessor.joblib`.

---

## 5. Empirical Dataset Audit Summary

From our inspection across all 8 CSV files:
- **Total Records:** 2,830,743 flows.
- **Raw Features:** 79 columns (78 flow statistics + 1 label).
- **Exact Duplicates:** 308,381 flows (10.9%) identified and pruned.
- **Identical Features with Differing Labels:** 698 flows identified with label collisions.
- **Missing Values:** Only `Flow Bytes/s` has missing values (1,358 NaNs across 2.83M rows).
- **Infinite Values:** Only `Flow Bytes/s` (1,509 Infs) and `Flow Packets/s` (2,867 Infs) contain infinities, caused by `Flow Duration = 0` division.
- **Zero-Variance Features:** Exactly 8 features are constant 0 across all 2.83M rows (`Bwd PSH Flags`, `Bwd URG Flags`, and 6 bulk transmission rates).
- **Duplicate Columns:** `Fwd Header Length.1` is 100% identical to `Fwd Header Length` (0 differences in 2.83M rows).

### Ground-Truth Class Distribution:
| Rank | Class | Flow Count | Percentage |
| :--- | :--- | :--- | :--- |
| 1 | **BENIGN** | 2,273,097 | 80.3004% |
| 2 | **DoS Hulk** | 231,073 | 8.1630% |
| 3 | **PortScan** | 158,930 | 5.6144% |
| 4 | **DDoS** | 128,027 | 4.5227% |
| 5 | **DoS GoldenEye** | 10,293 | 0.3636% |
| 6 | **FTP-Patator** | 7,938 | 0.2804% |
| 7 | **SSH-Patator** | 5,897 | 0.2083% |
| 8 | **DoS slowloris** | 5,796 | 0.2048% |
| 9 | **DoS Slowhttptest** | 5,499 | 0.1943% |
| 10 | **Bot** | 1,966 | 0.0695% |
| 11 | **Web Attack - Brute Force** | 1,507 | 0.0532% |
| 12 | **Web Attack - XSS** | 652 | 0.0230% |
| 13 | **Infiltration** | 36 | 0.0013% |
| 14 | **Web Attack - Sql Injection** | 21 | 0.0007% |
| 15 | **Heartbleed** | 11 | 0.0004% |

---

## 6. Feature Selection Justification & Validation Benchmark

We benchmarked model performance on held-out validation data before and after feature selection:

| Metric | Full Feature Set (69 Features) | Selected Feature Subset (25 Features) | Improvement / Difference |
| :--- | :--- | :--- | :--- |
| **Validation Accuracy** | 0.99949 | **0.99959** | +0.00010 |
| **Validation F1 Score (Macro)** | 0.99620 | **0.99696** | **+0.00076** |
| **Validation Precision** | 0.99926 | **0.99931** | +0.00005 |
| **Validation Recall** | 0.99318 | **0.99464** | **+0.00146** |
| **Training Time** | 1.631s | **1.335s** | **1.22x Speedup** |
| **Dimensionality** | 69 features | **25 features** | **63.8% Reduction** |

Pruning noisy, zero-variance, and collinear features yields superior generalization and faster inference without information loss.

---

## 7. Binary Threat Detection: Decision Tree Baseline vs. Random Forest Detector

Both models were trained using identical training data (with balanced class weights) and evaluated on the untouched held-out **Test Set (29,173 flows)**:

| Performance Metric | Decision Tree (Baseline) | Random Forest (Selected Model) | Empirical Difference |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 99.907% | **99.976%** | **+0.069%** |
| **Attack Precision** | 97.716% (24 False Positives) | **100.000% (0 False Positives)** | **+2.284% (Zero False Alarms)** |
| **Attack Recall** | **99.709%** | 99.320% | -0.389% |
| **Attack F1-Score** | 0.98703 | **0.99659** | **+0.00956** |
| **Macro F1-Score** | 0.99327 | **0.99823** | **+0.00496** |
| **ROC-AUC** | 0.99849 | **1.00000** | **+0.00151** |
| **PR-AUC** | 0.99425 | **0.99996** | **+0.00571** |
| **Inference Latency** | **0.0009 ms / flow** | 0.0053 ms / flow | +0.0044 ms |

### Key Justifications for Random Forest Selection:
1. **False Positive Elimination:** Decision Tree produced 24 false alarms on benign traffic. In production SOC environments, false alarms overwhelm security teams. Random Forest achieved **0 False Positives (100% precision)**.
2. **Probability Smoothness:** Random Forest provides continuous, well-calibrated class probabilities ($AUC = 1.000$), enabling precise threshold calibration for Member 4's SHAP explainers.
3. **Sub-millisecond Latency:** At **5.3 microseconds per flow**, Random Forest comfortably meets line-rate processing requirements for 10Gbps+ networks.

---

## 8. Integration Contract for Members 3 and 4
For complete instructions, code snippets, and usage contracts, see [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md).
