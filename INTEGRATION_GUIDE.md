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

## 3. Member 2: Binary Threat Detector (Random Forest)

### Target Encoding:
```python
# Encodes: BENIGN -> 0, ANY ATTACK -> 1
y_binary = PreprocessingPipeline.encode_labels(y_raw, mode="binary")

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
rf.fit(X_processed, y_binary)
```

### Decoding Predictions for Inference:
```python
y_pred = rf.predict(X_processed[:5])
pred_labels = PreprocessingPipeline.decode_labels(y_pred, mode="binary")
# Output: ['BENIGN', 'ATTACK', ...]
```

---

## 4. Member 3: Multiclass Attack Classifier (XGBoost)

### Target Encoding:
```python
# 15-class granular attack encoding (0 to 14)
y_multiclass = PreprocessingPipeline.encode_labels(y_raw, mode="multiclass")

# OR 7-class attack family encoding (Benign, DoS/DDoS, PortScan, BruteForce, WebAttack, Botnet, Infiltration/Exploit)
y_category = PreprocessingPipeline.encode_labels(y_raw, mode="category")
```

### Handling Class Imbalance:
Because minority classes (e.g., Heartbleed: 11 samples, SQL Injection: 21 samples) are heavily imbalanced, compute balanced sample weights:
```python
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight("balanced", y_multiclass)

# In XGBoost:
# model.fit(X_processed, y_multiclass, sample_weight=sample_weights)
```

---

## 5. Member 4: Explainable AI (SHAP & LIME)

### Feature Names & Ordering:
SHAP and LIME require exact feature names corresponding to each column in `X_processed`:

```python
import json
import shap

with open("artifacts/schemas/selected_features.json", "r") as f:
    feature_names = json.load(f)["selected_features"]

# TreeExplainer for Member 2's Random Forest or Member 3's XGBoost:
explainer = shap.TreeExplainer(rf, data=shap.sample(X_processed, 100))
shap_values = explainer(X_processed.iloc[:10])

# Feature names match exactly:
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
