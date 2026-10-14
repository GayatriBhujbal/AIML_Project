# Explainable AI-Based Network Threat Detection and Classification
## Components 1, 2 & 3: Preprocessing, Binary Detection, and Multiclass Attack Classification

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-58%20passed-brightgreen.svg)](tests/)
[![Dataset](https://img.shields.io/badge/dataset-CIC--IDS2017-orange.svg)](https://www.unb.ca/cic/datasets/ids-2017.html)
[![Binary Model](https://img.shields.io/badge/Random%20Forest-Accuracy%2099.98%25-success.svg)](artifacts/models/)
[![Attack Classifier](https://img.shields.io/badge/XGBoost-F1%2093.30%25-blueviolet.svg)](artifacts/models/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-ff69b4.svg)](src/models/attack_classifier.py)

---

## 1. Project Overview

This repository contains the production-grade implementation of **Components 1, 2, and 3** for the college mini-project: **Explainable AI-Based Network Threat Detection and Classification**.

The system uses a **two-stage hierarchical pipeline**:
- **Stage 1 (Binary):** Random Forest detects whether a network flow is BENIGN or ATTACK.
- **Stage 2 (Multiclass):** XGBoost classifies the exact attack category among 13 classes.
- **Explainability:** SHAP TreeExplainer provides global and per-record explanations at both stages.

### Key Deliverables:

1. **Component 1 — Data Engineering & Validation:**
   Full schema validation across all 8 raw CSVs (2,830,743 records), anomaly cleaning (infs, NaNs, sensor rollovers), stratified 70/15/15 splitting, and systematic feature selection (reducing 78 features to 25 optimal features).

2. **Component 2 — Binary Threat Detection:**
   - **Baseline:** Decision Tree classifier for empirical comparison.
   - **Primary Model:** Random Forest (100 trees, balanced class weights, calibrated threshold).
   - **Fair Evaluation:** Both models on identical held-out Test Set (29,173 flows).
   - **Interfacing:** `ThreatDetectionInferenceEngine` — routes attack flows to Component 3 and delivers explainer inputs for Member 4.

3. **Component 3 — Multiclass Attack Classification (XGBoost + SHAP):**
   - **Model:** XGBoost multiclass classifier over 13 CIC-IDS2017 attack categories.
   - **Input:** Pre-processed 25-feature attack-only DataFrames from Component 2.
   - **Class handling:** Derived dynamically from actual dataset labels; inverse-frequency sample weights handle extreme imbalance (Heartbleed=11 rows, Sql Injection=21 rows).
   - **SHAP:** Global feature importance and per-record local explanations via `TreeExplainer`.
   - **Results (verified, test set, 7,370 samples):** Accuracy 97.33% | F1-macro 93.30% | ROC-AUC 0.9991.

---

## 2. Directory Structure

```
AIML_project/
├── .gitignore                          # Prevents committing large CSVs/archives
├── README.md                           # Main project documentation
├── requirements.txt                    # Pinned production dependencies
├── INTEGRATION_GUIDE.md                # Integration contract for Members 2, 3, and 4
│
├── config/
│   ├── __init__.py
│   └── dataset_config.py               # Paths, constants, thresholds, env overrides
│
├── src/
│   ├── __init__.py
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── schema.py                   # Canonical 79-column schema, label mappings
│   │   └── loader.py                   # Robust CSV loader (chunked, memory-safe)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py                  # Infs, NaNs, negatives, and deduplication
│   │   ├── splitter.py                 # Stratified, leakage-safe train/val/test split
│   │   └── pipeline.py                 # PreprocessingPipeline (fitted on train only)
│   ├── feature_selection/
│   │   ├── __init__.py
│   │   └── selector.py                 # Variance, collinearity, and RF importance
│   ├── eda/
│   │   ├── __init__.py
│   │   └── visualizer.py               # Publication-quality plotting routines
│   └── models/
│       ├── __init__.py
│       ├── binary_detector.py          # DecisionTreeBaseline & RandomForestBinaryDetector
│       ├── evaluator.py                # Confusion matrix, ROC, PR curves, latency
│       ├── predictor.py                # ThreatDetectionInferenceEngine (Member 2 → 3 → 4)
│       ├── attack_classifier.py        # [NEW] XGBoostAttackClassifier + SHAP (Member 3)
│       └── attack_evaluator.py         # [NEW] AttackClassifierEvaluator (metrics + plots)
│
├── scripts/
│   ├── run_eda.py                      # CLI: full audit report & 5 figures
│   ├── run_feature_selection.py        # CLI: feature selection & validation benchmark
│   ├── prepare_dataset.py              # CLI: fit & serialize preprocessor.joblib
│   ├── train_binary_models.py          # CLI: train DT baseline & RF detector
│   ├── evaluate_binary_models.py       # CLI: evaluate saved models on test CSVs
│   ├── train_attack_classifier.py      # [NEW] CLI: train XGBoost attack classifier
│   └── run_with_network_data.py        # [NEW] CLI: adapter for pre-split network CSVs
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb # Interactive exploration notebook
│
├── artifacts/
│   ├── schemas/
│   │   ├── feature_schema.json         # Documented feature definitions & dtypes
│   │   ├── label_mapping.json          # Binary, multiclass, and category mappings
│   │   └── selected_features.json      # Final selected 25 features list
│   ├── figures/
│   │   ├── class_distribution.png
│   │   ├── attack_categories.png
│   │   ├── missing_and_inf_analysis.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_distributions_by_class.png
│   │   ├── feature_importance.png
│   │   ├── binary_roc_curve.png
│   │   ├── binary_precision_recall_curve.png
│   │   ├── binary_confusion_matrices.png
│   │   ├── attack_model_confusion_matrix.png         # [NEW]
│   │   ├── attack_model_shap_global_importance.png   # [NEW]
│   │   └── attack_model_shap_per_class.png           # [NEW]
│   ├── models/
│   │   ├── preprocessor.joblib                  # Pre-fitted preprocessing pipeline
│   │   ├── binary_decision_tree.joblib          # Trained Decision Tree baseline
│   │   ├── binary_random_forest.joblib          # Trained Random Forest detector
│   │   ├── inference_samples.json               # Raw sample flows for integration testing
│   │   ├── attack_xgboost_classifier.joblib     # [NEW] Trained XGBoost attack classifier
│   │   └── attack_class_mapping.json            # [NEW] Class label mapping JSON
│   └── reports/
│       ├── dataset_quality_report.md
│       ├── feature_selection_report.json
│       ├── binary_model_evaluation_report.md
│       ├── binary_model_evaluation.json
│       ├── attack_model_evaluation.json          # [NEW]
│       └── attack_xgboost_feature_importance.json # [NEW]
│
└── tests/
    ├── __init__.py
    ├── test_schema.py                  # Schema and label encoding tests
    ├── test_cleaner.py                 # Cleaning, inf, negative, and dedup tests
    ├── test_splitter.py                # Stratification and zero-leakage tests
    ├── test_pipeline.py                # Fit on train only, transform, and joblib tests
    ├── test_binary_models.py           # Decision Tree & Random Forest unit tests
    ├── test_predictor_interface.py     # Integration tests for Member 4 and Member 3
    └── test_attack_classifier.py       # [NEW] 36 tests: XGBoost + SHAP + evaluator
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

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run Full Test Suite *(no dataset required)*
```bash
python -m pytest tests/ -v
```
**Expected:** `58 passed` — covers schema, cleaning, splitting, pipeline, binary models, predictor interface, XGBoost classifier, SHAP, evaluator, and Member 4 interface contracts.

### Step 3 — Generate EDA & Quality Report *(requires dataset)*
```bash
python scripts/run_eda.py
```
Outputs figures into `artifacts/figures/` and report to `artifacts/reports/dataset_quality_report.md`.

### Step 4 — Feature Selection *(requires dataset)*
```bash
python scripts/run_feature_selection.py
```
Saves `artifacts/schemas/selected_features.json` (25 features).

### Step 5 — Fit & Export Preprocessing Pipeline *(requires dataset)*
```bash
python scripts/prepare_dataset.py --sample-per-file 30000
```
Serializes `artifacts/models/preprocessor.joblib`.

### Step 6 — Train Binary Detector *(requires dataset)*
```bash
python scripts/train_binary_models.py
python scripts/evaluate_binary_models.py
```

### Step 7 — Train XGBoost Attack Classifier *(requires dataset)*

**Option A — Official CIC-IDS2017 MachineLearningCSV format:**
```bash
python scripts/train_attack_classifier.py
# Memory-constrained machine:
python scripts/train_attack_classifier.py --sample-per-file 5000 --n-estimators 100
# Skip SHAP for a faster run:
python scripts/train_attack_classifier.py --skip-shap
```

**Option B — Pre-split network CSV files (e.g. dos_hulk.csv, portscan.csv):**
```bash
python scripts/run_with_network_data.py
```

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

| Metric | Full Feature Set (69 Features) | Selected Feature Subset (25 Features) | Improvement |
| :--- | :--- | :--- | :--- |
| **Validation Accuracy** | 0.99949 | **0.99959** | +0.00010 |
| **Validation F1 Score (Macro)** | 0.99620 | **0.99696** | **+0.00076** |
| **Validation Precision** | 0.99926 | **0.99931** | +0.00005 |
| **Validation Recall** | 0.99318 | **0.99464** | **+0.00146** |
| **Training Time** | 1.631s | **1.335s** | **1.22x Speedup** |
| **Dimensionality** | 69 features | **25 features** | **63.8% Reduction** |

Pruning noisy, zero-variance, and collinear features yields superior generalization and faster inference without information loss.

### Selected 25 Features:
`Bwd Packet Length Std`, `Destination Port`, `Packet Length Variance`, `Subflow Fwd Bytes`, `Average Packet Size`, `Packet Length Std`, `Subflow Bwd Bytes`, `Fwd Packet Length Std`, `Bwd Packet Length Mean`, `Avg Fwd Segment Size`, `Fwd Packet Length Max`, `Fwd IAT Min`, `Fwd IAT Std`, `Min Packet Length`, `Fwd IAT Total`, `Fwd Packets/s`, `Fwd IAT Max`, `Flow Bytes/s`, `Init_Win_bytes_backward`, `Init_Win_bytes_forward`, `Flow IAT Min`, `Bwd IAT Std`, `Bwd IAT Mean`, `Active Mean`, `Bwd IAT Max`

---

## 7. Binary Threat Detection: Decision Tree vs. Random Forest

Both models were trained with balanced class weights and evaluated on the untouched held-out **Test Set (29,173 flows)**:

| Performance Metric | Decision Tree (Baseline) | Random Forest (Selected) | Difference |
| :--- | :--- | :--- | :--- |
| **Test Accuracy** | 99.907% | **99.976%** | **+0.069%** |
| **Attack Precision** | 97.716% (24 FPs) | **100.000% (0 FPs)** | **+2.284%** |
| **Attack Recall** | **99.709%** | 99.320% | -0.389% |
| **Attack F1-Score** | 0.98703 | **0.99659** | **+0.00956** |
| **Macro F1-Score** | 0.99327 | **0.99823** | **+0.00496** |
| **ROC-AUC** | 0.99849 | **1.00000** | **+0.00151** |
| **PR-AUC** | 0.99425 | **0.99996** | **+0.00571** |
| **Inference Latency** | **0.0009 ms/flow** | 0.0053 ms/flow | +0.0044 ms |

**Random Forest selected** because it achieves zero false positives (100% precision), well-calibrated probabilities (AUC = 1.000), and sub-millisecond latency at 5.3 µs/flow.

---

## 8. Multiclass Attack Classification: XGBoost + SHAP (Component 3)

### Architecture
The XGBoost classifier operates as the **second stage** of the hierarchical pipeline. It receives only the flows flagged as `ATTACK` by the Random Forest binary detector, and predicts the specific attack category.

```
Raw Traffic → [Random Forest Binary Detector] → BENIGN (logged) / ATTACK ↓
                                                 [XGBoost Attack Classifier]
                                                 → DoS Hulk / DDoS / PortScan / Bot / ...
                                                 → SHAP explanations per prediction
```

### Model Design
| Design Choice | Decision | Justification |
| :--- | :--- | :--- |
| **Objective** | `multi:softprob` | Outputs calibrated class probabilities for all 13 classes |
| **Eval metric** | `mlogloss` | Penalises confident wrong predictions; appropriate for multiclass |
| **Class imbalance** | Inverse-frequency `sample_weight` | XGBClassifier has no `class_weight` for multiclass; gives gradient-level control |
| **Early stopping** | 20 rounds on validation set | Prevents overfitting; best iteration logged |
| **Features** | Exact 25 from `selected_features.json` | Same schema as Random Forest — no re-preprocessing |
| **Class mapping** | Derived from `LabelEncoder.fit(y_train)` | Labels absent from files are automatically absent from the model |

### Verified Test Results (7,370 held-out samples, 13 attack classes)

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **97.33%** |
| **Precision (macro)** | **92.70%** |
| **Recall (macro)** | **94.21%** |
| **F1-score (macro)** | **93.30%** |
| **F1-score (weighted)** | **97.37%** |
| **ROC-AUC (OvR, macro)** | **0.9991** |

### Per-Class Results

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Bot | 1.00 | 1.00 | 1.00 | 750 |
| DDoS | 1.00 | 1.00 | 1.00 | 750 |
| DoS GoldenEye | 0.97 | 0.99 | 0.98 | 750 |
| DoS Hulk | 1.00 | 0.99 | 1.00 | 750 |
| DoS Slowhttptest | 0.96 | 0.98 | 0.97 | 750 |
| DoS slowloris | 0.99 | 0.95 | 0.97 | 750 |
| FTP-Patator | 1.00 | 1.00 | 1.00 | 750 |
| Heartbleed | 1.00 | 1.00 | 1.00 | 2 |
| PortScan | 1.00 | 1.00 | 1.00 | 750 |
| SSH-Patator | 1.00 | 1.00 | 1.00 | 750 |
| Web Attack - Brute Force | 0.89 | 0.79 | 0.84 | 410 |
| Web Attack - Sql Injection | 0.60 | 0.75 | 0.67 | 4 |
| Web Attack - XSS | 0.65 | 0.80 | 0.72 | 204 |

> **Note on minority classes:** Web Attack variants (XSS, Sql Injection) have lower F1 due to extreme imbalance (4–204 test samples vs 750 for major classes). Inverse-frequency sample_weight partially mitigates this. Heartbleed achieves F1=1.00 despite only 2 test samples.

### SHAP Explainability

The SHAP integration uses `shap.TreeExplainer` on the actual trained XGBoost model. No features are labelled as "causes" of attacks — SHAP values describe how each feature *influences the model's prediction*.

| Method | Description | Output Shape |
| :--- | :--- | :--- |
| `explain_global(X, max_samples=500)` | Global SHAP values, sampled from real data | `(N_sample, 25, 13)` |
| `explain_local(X_single)` | Per-record SHAP for one flow | `(25, 13)` + base values |
| `get_xgb_booster()` | Raw booster for custom SHAP usage | `xgb.Booster` |

**Sample local explanation (first test record, predicted: Web Attack - Brute Force):**
| Feature | SHAP Value | Interpretation |
| :--- | :--- | :--- |
| `Fwd Packets/s` | +0.9031 | Strongest influence on this prediction |
| `Flow IAT Min` | +0.3095 | Second strongest influence |
| `Init_Win_bytes_forward` | +0.2869 | Third strongest influence |
| `Fwd IAT Std` | +0.2399 | Fourth strongest influence |
| `Fwd IAT Min` | +0.1791 | Fifth strongest influence |

---

## 9. Integration Contract for Members 3 and 4

For complete instructions and code snippets, see [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md).

### Member 3 → Member 4 Interface (SHAP & LIME)

```python
from src.models.attack_classifier import XGBoostAttackClassifier
import shap

clf = XGBoostAttackClassifier.load("artifacts/models/attack_xgboost_classifier.joblib")

# Global SHAP (500 real samples, no synthetic data)
shap_vals, X_sample = clf.explain_global(attack_features, max_samples=500)
# shap_vals: np.ndarray (500, 25, 13)

# Local SHAP for one record
sv, base, class_names = clf.explain_local(attack_features.iloc[:1])
# sv: (25, 13), base: (13,)

# Direct booster for custom TreeExplainer
booster = clf.get_xgb_booster()
explainer = shap.TreeExplainer(booster)

# Class probabilities
preds, proba, class_names = clf.predict_with_scores(attack_features)
```

### End-to-End Pipeline

```python
from src.models.predictor import ThreatDetectionInferenceEngine
from src.models.attack_classifier import XGBoostAttackClassifier

# Stage 1: Binary Detection
engine = ThreatDetectionInferenceEngine()
attack_features, attack_indices, _ = engine.route_to_attack_classifier(raw_traffic_df)

# Stage 2: Attack Classification
clf = XGBoostAttackClassifier.load("artifacts/models/attack_xgboost_classifier.joblib")
attack_labels = clf.predict_label_names(attack_features)  # e.g. ["DDoS", "Bot", ...]
attack_proba  = clf.predict_proba(attack_features)        # shape (K, 13)
```

---

## 10. Test Suite Summary

| Test File | Tests | Coverage |
| :--- | :--- | :--- |
| `test_schema.py` | 4 | Schema validation, label encoding |
| `test_cleaner.py` | 3 | Inf removal, deduplication, feature separation |
| `test_splitter.py` | 3 | Stratification, minority preservation, zero leakage |
| `test_pipeline.py` | 4 | Fit-on-train-only, transform, joblib save/load |
| `test_binary_models.py` | 2 | Decision Tree, Random Forest fit/predict/threshold |
| `test_predictor_interface.py` | 6 | Engine predict, route-to-classifier, Member 4 contract |
| `test_attack_classifier.py` | **36** | XGBoost: class mapping, prediction, SHAP (global + local), persistence, evaluator, Member 4 interface |
| **Total** | **58** | **All passed ✅** |

---

## 11. Dependencies

```
numpy>=1.26.0       pandas>=2.1.0       scikit-learn>=1.4.0
scipy>=1.12.0       matplotlib>=3.8.0   seaborn>=0.13.0
joblib>=1.3.0       pytest>=8.0.0
xgboost>=2.0.0      shap>=0.44.0
```

Install all with: `pip install -r requirements.txt`
