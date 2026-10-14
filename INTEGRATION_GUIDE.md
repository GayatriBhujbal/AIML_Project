# Integration Guide for Members 2, 3, and 4
## Explainable AI-Based Network Threat Detection and Classification

**Component 1 (Dataset Handling, Preprocessing, and EDA)** provides a standardized, leak-free feature schema and reusable preprocessing interface. Follow this contract to ensure seamless integration.

---

## 1. Quick Integration Checklist

| Teammate | Role | Primary Artifacts & Methods to Use |
| :--- | :--- | :--- |
| **Member 2** | Random Forest Binary Detector | `artifacts/models/preprocessor.joblib`, `mode='binary'` |
| **Member 3** | XGBoost Multiclass Classifier | `artifacts/models/preprocessor.joblib`, `mode='multiclass'` / `mode='category'` |
| **Member 4** | XAI (SHAP & LIME) | `artifacts/schemas/selected_features.json`, `pipeline.feature_names_out_` |

---

## 2. How to Load and Preprocess Data in Your Module

Do not re-implement data loading, whitespace cleaning, or imputation. Import the shared pipeline:

```python
import joblib
import pandas as pd
from src.dataset.loader import DatasetLoader
from src.preprocessing.pipeline import PreprocessingPipeline

# 1. Load raw dataset or test file
loader = DatasetLoader()
raw_df = loader.load_file("MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# 2. Separate features (X) and raw labels (y)
X_raw = raw_df.drop(columns=["Label"])
y_raw = raw_df["Label"]

# 3. Load the pre-fitted preprocessing pipeline
pipeline = PreprocessingPipeline.load("artifacts/models/preprocessor.joblib")

# 4. Transform features into the identical 25-feature representation
X_processed = pipeline.transform(X_raw)
# Shape: (N, 25), scaled with RobustScaler, imputed with training medians
```

---

## 3. Member 2: Binary Threat Detector (Random Forest & Baseline Decision Tree)

### Deployed Artifacts:
- **Primary Detector:** `artifacts/models/binary_random_forest.joblib` (Random Forest, 100 trees, balanced class weights, calibrated threshold: 0.60)
- **Baseline Model:** `artifacts/models/binary_decision_tree.joblib` (Decision Tree, depth 12)
- **Evaluation Report:** `artifacts/reports/binary_model_evaluation_report.md`
- **Evaluation Curves:**
  - `artifacts/figures/binary_roc_curve.png`
  - `artifacts/figures/binary_precision_recall_curve.png`
  - `artifacts/figures/binary_confusion_matrices.png`

### Test Performance on Held-Out Data (29,173 flows):
- **Random Forest:** Accuracy: **99.976%**, Attack Precision: **100.00%**, Attack Recall: **99.32%**, F1: **0.9966**, ROC-AUC: **1.0000**, Latency: **0.0053 ms / sample**.
- **Decision Tree Baseline:** Accuracy: 99.907%, Attack Precision: 97.72%, Attack Recall: 99.71%, F1: 0.9870, ROC-AUC: 0.9985, Latency: 0.0009 ms / sample.

---

## 4. Member 3: Multiclass Attack Classifier (XGBoost) Integration

### Hierarchical Architecture Hand-off:
In the end-to-end detection pipeline, Member 2 filters benign traffic, routing only suspected attacks to Member 3 for fine-grained attack classification (DDoS, PortScan, Web Attack, Patator, Bot, Infiltration, etc.):

```python
import pandas as pd
from src.models.predictor import ThreatDetectionInferenceEngine

# 1. Initialize binary inference engine
engine = ThreatDetectionInferenceEngine()

# 2. Ingest raw traffic batch
raw_traffic_df = pd.read_csv("path/to/incoming_traffic.csv")

# 3. Route suspected attack traffic directly to XGBoost
# attack_features: DataFrame of shape (K, 25) with ONLY rows flagged as ATTACK
# attack_indices: Original DataFrame indices corresponding to the attack flows
attack_features, attack_indices, _ = engine.route_to_attack_classifier(raw_traffic_df)

# 4. Member 3 classifies the attack types:
# xgb_predictions = member3_xgboost_model.predict(attack_features)
```

---

## 5. Member 4: Explainable AI (SHAP & LIME) Integration

### Direct Explainer Inputs:
Member 4 does not need to manually configure feature matrices or extract tree weights. `ThreatDetectionInferenceEngine` provides a turnkey explainer interface:

```python
import shap
from src.models.predictor import ThreatDetectionInferenceEngine

# 1. Initialize inference engine
engine = ThreatDetectionInferenceEngine()

# 2. Get preprocessed DataFrame and underlying scikit-learn RandomForestClassifier
X_trans, rf_estimator = engine.get_explainer_inputs(sample_raw_df)

# 3. Initialize SHAP TreeExplainer
explainer = shap.TreeExplainer(rf_estimator, data=shap.sample(X_trans, 100))
shap_values = explainer(X_trans)

# 4. Generate local waterfall explanation for any flow:
shap.plots.waterfall(shap_values[0])
```

---

## 6. Selected 25 Features Reference

The exact 25 features selected from empirical Random Forest importance and collinearity filtering:
1. `Bwd Packet Length Std`
2. `Destination Port`
3. `Packet Length Variance`
4. `Subflow Fwd Bytes`
5. `Average Packet Size`
6. `Packet Length Std`
7. `Subflow Bwd Bytes`
8. `Fwd Packet Length Std`
9. `Bwd Packet Length Mean`
10. `Avg Fwd Segment Size`
11. `Fwd Packet Length Max`
12. `Fwd IAT Min`
13. `Fwd IAT Std`
14. `Min Packet Length`
15. `Fwd IAT Total`
16. `Fwd Packets/s`
17. `Fwd IAT Max`
18. `Flow Bytes/s`
19. `Init_Win_bytes_backward`
20. `Init_Win_bytes_forward`
21. `Flow IAT Min`
22. `Bwd IAT Std`
23. `Bwd IAT Mean`
24. `Active Mean`
25. `Bwd IAT Max`

---

## 7. Need Help or Schema Changes?
If your model requires an unscaled representation or a different subset of features:
- Use `PreprocessingPipeline(scaler_type='none')`.
- All schema files are in `artifacts/schemas/`.
- Preprocessing bug reports and questions: refer to unit tests in `tests/test_pipeline.py`.
