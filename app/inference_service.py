"""
Reusable inference service for the Explainable AI Network Threat Detection project.

Responsibilities
----------------
* Load and cache all saved artifacts (preprocessor, RF, XGBoost, schemas, reports).
* Validate artifact compatibility at startup.
* Validate CSV input schemas against the loaded preprocessor.
* Run Stage-1 Random Forest binary detection.
* Route ATTACK records to Stage-2 XGBoost attack-category classifier.
* Generate SHAP explanations via the existing model interfaces.
* Load evaluation reports from disk.
* Report missing artifacts and errors without fabricating results.

This module contains NO Streamlit code. Import it from Streamlit pages.
"""

import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.dataset_config import MODELS_DIR, SCHEMAS_DIR, REPORTS_DIR
from src.dataset.schema import TARGET_COLUMN, clean_column_name

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Artifact status dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ArtifactStatus:
    preprocessor_ok: bool = False
    rf_ok: bool = False
    dt_ok: bool = False
    xgb_ok: bool = False
    selected_features_ok: bool = False
    feature_schema_ok: bool = False
    class_mapping_ok: bool = False
    binary_report_ok: bool = False
    attack_report_ok: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def can_predict_binary(self) -> bool:
        return self.preprocessor_ok and self.rf_ok and self.selected_features_ok

    @property
    def can_predict_attack(self) -> bool:
        return self.can_predict_binary and self.xgb_ok

    @property
    def can_compare_baseline(self) -> bool:
        return self.preprocessor_ok and self.dt_ok and self.selected_features_ok


# ──────────────────────────────────────────────────────────────────────────────
# Prediction result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Holds per-record prediction results."""
    row_index: Any
    binary_label: str          # 'BENIGN' or 'ATTACK'
    binary_code: int           # 0 or 1
    attack_proba: float        # P(ATTACK) from RF
    attack_category: Optional[str] = None   # XGBoost label if ATTACK
    category_proba: Optional[float] = None  # max class probability from XGBoost
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Inference Service
# ──────────────────────────────────────────────────────────────────────────────

class InferenceService:
    """
    Central inference service.  Instantiate once per Streamlit session using
    @st.cache_resource so artifacts are loaded only once.
    """

    def __init__(self):
        self.status = ArtifactStatus()
        self.preprocessor = None
        self.rf_detector = None
        self.dt_baseline = None
        self.xgb_classifier = None
        self.selected_features: List[str] = []
        self.feature_schema: Dict = {}
        self.class_mapping: Dict = {}
        self.binary_report: Dict = {}
        self.attack_report: Dict = {}
        self._load_all_artifacts()

    # ── Artifact loading ───────────────────────────────────────────────────────

    def _load_all_artifacts(self) -> None:
        """Load every artifact and populate ArtifactStatus."""
        self._load_preprocessor()
        self._load_rf()
        self._load_dt()
        self._load_xgb()
        self._load_selected_features()
        self._load_feature_schema()
        self._load_class_mapping()
        self._load_reports()

    def _load_preprocessor(self) -> None:
        from src.preprocessing.pipeline import PreprocessingPipeline
        path = MODELS_DIR / "preprocessor.joblib"
        try:
            self.preprocessor = PreprocessingPipeline.load(path)
            self.status.preprocessor_ok = True
            logger.info(f"Preprocessor loaded: {len(self.preprocessor.feature_names_out_)} output features.")
        except FileNotFoundError:
            self.status.errors.append(f"Preprocessor not found: {path}")
        except RuntimeError as e:
            self.status.errors.append(f"Preprocessor schema mismatch: {e}")
        except Exception as e:
            self.status.errors.append(f"Preprocessor load error: {e}")

    def _load_rf(self) -> None:
        from src.models.binary_detector import RandomForestBinaryDetector
        path = MODELS_DIR / "binary_random_forest.joblib"
        try:
            self.rf_detector = RandomForestBinaryDetector.load(path)
            self.status.rf_ok = True
            logger.info("RandomForest binary detector loaded.")
        except FileNotFoundError:
            self.status.errors.append(f"RF model not found: {path}")
        except Exception as e:
            self.status.errors.append(f"RF model load error: {e}")

    def _load_dt(self) -> None:
        from src.models.binary_detector import DecisionTreeBaseline
        path = MODELS_DIR / "binary_decision_tree.joblib"
        try:
            self.dt_baseline = DecisionTreeBaseline.load(path)
            self.status.dt_ok = True
            logger.info("Decision Tree baseline loaded.")
        except FileNotFoundError:
            self.status.warnings.append(f"Decision Tree baseline not found: {path}")
        except Exception as e:
            self.status.warnings.append(f"DT load error: {e}")

    def _load_xgb(self) -> None:
        from src.models.attack_classifier import XGBoostAttackClassifier
        path = MODELS_DIR / "attack_xgboost_classifier.joblib"
        try:
            self.xgb_classifier = XGBoostAttackClassifier.load(path)
            self.status.xgb_ok = True
            logger.info(f"XGBoost classifier loaded: {self.xgb_classifier.n_classes_} classes.")
        except FileNotFoundError:
            self.status.warnings.append(f"XGBoost model not found: {path}")
        except Exception as e:
            self.status.warnings.append(f"XGBoost load error: {e}")

    def _load_selected_features(self) -> None:
        path = SCHEMAS_DIR / "selected_features.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.selected_features = data["selected_features"]
            self.status.selected_features_ok = True
        except FileNotFoundError:
            # Fall back to preprocessor output if available
            if self.preprocessor is not None:
                self.selected_features = self.preprocessor.feature_names_out_
                self.status.selected_features_ok = True
                self.status.warnings.append("selected_features.json not found; using preprocessor output features.")
            else:
                self.status.errors.append(f"selected_features.json not found: {path}")
        except Exception as e:
            self.status.errors.append(f"selected_features.json load error: {e}")

    def _load_feature_schema(self) -> None:
        path = SCHEMAS_DIR / "feature_schema.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.feature_schema = json.load(f)
            self.status.feature_schema_ok = True
        except FileNotFoundError:
            self.status.warnings.append(f"feature_schema.json not found: {path}")
        except Exception as e:
            self.status.warnings.append(f"feature_schema.json load error: {e}")

    def _load_class_mapping(self) -> None:
        path = MODELS_DIR / "attack_class_mapping.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.class_mapping = json.load(f)
            self.status.class_mapping_ok = True
        except FileNotFoundError:
            self.status.warnings.append(f"attack_class_mapping.json not found: {path}")
        except Exception as e:
            self.status.warnings.append(f"Class mapping load error: {e}")

    def _load_reports(self) -> None:
        # Binary report
        bp = REPORTS_DIR / "binary_model_evaluation.json"
        try:
            with open(bp, "r", encoding="utf-8") as f:
                self.binary_report = json.load(f)
            self.status.binary_report_ok = True
        except Exception as e:
            self.status.warnings.append(f"Binary evaluation report not found/readable: {e}")

        # Attack (XGBoost) report
        ap = REPORTS_DIR / "attack_model_evaluation.json"
        try:
            with open(ap, "r", encoding="utf-8") as f:
                self.attack_report = json.load(f)
            self.status.attack_report_ok = True
        except Exception as e:
            self.status.warnings.append(f"Attack evaluation report not found/readable: {e}")

    # ── CSV validation ─────────────────────────────────────────────────────────

    def validate_uploaded_csv(
        self, df: pd.DataFrame
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate that df has the raw CIC-IDS2017 columns required by the preprocessor.

        Returns:
            (is_valid, errors, warnings)
        """
        errors: List[str] = []
        warnings: List[str] = []

        if df.empty:
            errors.append("Uploaded file contains no data rows.")
            return False, errors, warnings

        # Clean column names the same way the pipeline does
        df_cols = [clean_column_name(c) for c in df.columns]

        if self.preprocessor is None:
            errors.append("Preprocessor not loaded — cannot validate columns.")
            return False, errors, warnings

        required = self.preprocessor.feature_names_in_
        missing = [c for c in required if c not in df_cols]
        extra = [c for c in df_cols if c not in required and c != TARGET_COLUMN]

        if missing:
            errors.append(
                f"{len(missing)} required feature column(s) are missing from the uploaded file: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}. "
                "Please upload a raw CIC-IDS2017-format CSV."
            )

        if extra:
            warnings.append(
                f"{len(extra)} extra column(s) found (will be ignored): "
                f"{extra[:10]}{'...' if len(extra) > 10 else ''}."
            )

        if TARGET_COLUMN in df_cols:
            warnings.append(
                f"Column '{TARGET_COLUMN}' detected. It will be excluded from model inputs."
            )

        # Check for infinite values
        numeric_df = df.select_dtypes(include=[np.number])
        inf_cols = [c for c in numeric_df.columns if np.isinf(numeric_df[c]).any()]
        if inf_cols:
            warnings.append(
                f"Infinite values found in {len(inf_cols)} column(s): {inf_cols[:5]}. "
                "These will be replaced with column medians during preprocessing."
            )

        return len(errors) == 0, errors, warnings

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted preprocessing pipeline to a raw input DataFrame.
        Returns a (N, 25) scaled DataFrame.
        Raises ValueError if required columns are missing.
        """
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not loaded.")
        # Clean column names
        df = df.copy()
        df.columns = [clean_column_name(c) for c in df.columns]
        # Strip label if present
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])
        # Run pipeline
        X_trans = self.preprocessor.transform(df)
        # Enforce selected feature ordering
        return X_trans[self.selected_features]

    # ── Binary prediction ──────────────────────────────────────────────────────

    def predict_binary(
        self, X_trans: pd.DataFrame, threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Random Forest binary detector on preprocessed data.

        Returns:
            (binary_predictions, attack_probabilities)
            binary_predictions: int array {0: BENIGN, 1: ATTACK}
            attack_probabilities: float array P(ATTACK) per record
        """
        if self.rf_detector is None:
            raise RuntimeError("Random Forest model not loaded.")
        thresh = threshold if threshold is not None else self.rf_detector.decision_threshold
        proba = self.rf_detector.predict_proba(X_trans)
        attack_proba = proba[:, 1]
        preds = (attack_proba >= thresh).astype(int)
        return preds, attack_proba

    def predict_binary_dt(
        self, X_trans: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Decision Tree baseline on preprocessed data.
        Returns (predictions, attack_probabilities).
        """
        if self.dt_baseline is None:
            raise RuntimeError("Decision Tree baseline not loaded.")
        proba = self.dt_baseline.predict_proba(X_trans)
        attack_proba = proba[:, 1]
        preds = (attack_proba >= 0.5).astype(int)
        return preds, attack_proba

    # ── Attack classification ──────────────────────────────────────────────────

    def predict_attack_categories(
        self, X_trans: pd.DataFrame
    ) -> Tuple[List[str], np.ndarray]:
        """
        Run XGBoost attack classifier on preprocessed ATTACK records.

        Args:
            X_trans: Preprocessed DataFrame containing only ATTACK records.

        Returns:
            (category_labels, category_probabilities)
            category_labels: List of attack category strings per record.
            category_probabilities: max class probability per record.
        """
        if self.xgb_classifier is None:
            raise RuntimeError("XGBoost model not loaded.")
        if X_trans.empty:
            return [], np.array([])
        preds, proba, class_names = self.xgb_classifier.predict_with_scores(X_trans)
        labels = [class_names[p] for p in preds]
        max_proba = proba.max(axis=1)
        return labels, max_proba

    # ── Full two-stage pipeline ────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        raw_df: pd.DataFrame,
        id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run the complete two-stage detection pipeline on raw CSV data.

        Args:
            raw_df: Raw CIC-IDS2017-format DataFrame.
            id_col: Optional column name to preserve as row identifier.

        Returns:
            results_df: DataFrame with columns:
                row_id, binary_label, attack_proba, attack_category, category_proba, status
        """
        if not self.status.can_predict_binary:
            raise RuntimeError(
                "Cannot run pipeline: preprocessor or RF model not available. "
                f"Errors: {self.status.errors}"
            )

        # Preserve identifiers
        ids = raw_df[id_col].tolist() if (id_col and id_col in raw_df.columns) else list(range(len(raw_df)))

        # Preprocess
        X_trans = self.preprocess(raw_df)

        # Stage 1: binary detection
        binary_preds, attack_proba = self.predict_binary(X_trans)

        # Stage 2: attack classification for ATTACK rows
        attack_mask = binary_preds == 1
        attack_cats = [None] * len(raw_df)
        attack_cat_proba = [None] * len(raw_df)

        if attack_mask.any() and self.status.can_predict_attack:
            X_attack = X_trans[attack_mask]
            try:
                cat_labels, cat_proba = self.predict_attack_categories(X_attack)
                attack_indices = np.where(attack_mask)[0]
                for i, (lbl, prob) in zip(attack_indices, zip(cat_labels, cat_proba)):
                    attack_cats[i] = lbl
                    attack_cat_proba[i] = float(prob)
            except Exception as e:
                logger.error(f"XGBoost classification error: {e}")
                # Mark attack rows with error; binary result still valid
                for i in np.where(attack_mask)[0]:
                    attack_cats[i] = "XGBoost Error"

        results = pd.DataFrame({
            "row_id": ids,
            "binary_label": ["ATTACK" if p == 1 else "BENIGN" for p in binary_preds],
            "attack_proba": [round(float(p), 4) for p in attack_proba],
            "attack_category": attack_cats,
            "category_proba": attack_cat_proba,
            "status": ["OK"] * len(raw_df),
        })
        return results

    # ── SHAP explanations ──────────────────────────────────────────────────────

    def explain_rf_local(
        self, X_single_trans: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Generate SHAP TreeExplainer explanation for a single preprocessed record
        using the Random Forest model.

        Returns:
            Dict with keys: shap_values, expected_value, feature_names, feature_values
            or None on failure.
        """
        try:
            import shap
            if self.rf_detector is None:
                return None

            rf_model = getattr(self.rf_detector, "model", self.rf_detector)
            classes = list(getattr(rf_model, "classes_", [0, 1]))

            # Determine class index corresponding to ATTACK (class 1)
            if 1 in classes:
                attack_idx = classes.index(1)
            elif "ATTACK" in [str(c).upper() for c in classes]:
                attack_idx = [str(c).upper() for c in classes].index("ATTACK")
            else:
                attack_idx = 1 if len(classes) > 1 else 0

            explainer = shap.TreeExplainer(rf_model)
            sv = explainer.shap_values(X_single_trans)
            n_features = X_single_trans.shape[1]

            # Handle different SHAP output formats across SHAP / sklearn versions
            if isinstance(sv, list):
                # List of class-specific arrays, e.g. [class0_vals, class1_vals]
                if len(sv) > attack_idx:
                    class_sv = np.asarray(sv[attack_idx])
                elif len(sv) > 0:
                    class_sv = np.asarray(sv[-1])
                else:
                    raise ValueError("Empty SHAP values list returned by TreeExplainer.")

                if class_sv.ndim == 2:
                    sv_attack = class_sv[0]
                else:
                    sv_attack = class_sv.reshape(-1)

            elif isinstance(sv, np.ndarray):
                if sv.ndim == 3:
                    # Shape (n_samples, n_features, n_classes) or variations
                    if sv.shape[1] == n_features and sv.shape[2] > 1:
                        idx = attack_idx if sv.shape[2] > attack_idx else -1
                        sv_attack = sv[0, :, idx]
                    elif sv.shape[2] == n_features and sv.shape[0] > 1:
                        idx = attack_idx if sv.shape[0] > attack_idx else -1
                        sv_attack = sv[idx, 0, :]
                    elif sv.shape[2] == n_features and sv.shape[1] > 1:
                        idx = attack_idx if sv.shape[1] > attack_idx else -1
                        sv_attack = sv[0, idx, :]
                    else:
                        idx = attack_idx if sv.shape[2] > attack_idx else 0
                        sv_attack = sv[0, :, idx]
                elif sv.ndim == 2:
                    # Shape (n_features, n_classes), (n_samples, n_features), or (n_classes, n_features)
                    if sv.shape[0] == n_features and sv.shape[1] > 1:
                        idx = attack_idx if sv.shape[1] > attack_idx else -1
                        sv_attack = sv[:, idx]
                    elif sv.shape[1] == n_features and sv.shape[0] > 1:
                        idx = attack_idx if sv.shape[0] > attack_idx else -1
                        sv_attack = sv[idx, :]
                    elif sv.shape[0] == 1 and sv.shape[1] == n_features:
                        sv_attack = sv[0, :]
                    elif sv.shape[1] == n_features:
                        sv_attack = sv[0, :]
                    else:
                        sv_attack = sv.reshape(-1)[:n_features]
                elif sv.ndim == 1:
                    sv_attack = sv
                else:
                    sv_attack = sv.reshape(-1)[:n_features]
            else:
                sv_attack = np.asarray(sv).reshape(-1)[:n_features]

            sv_attack = np.asarray(sv_attack, dtype=float).reshape(-1)
            if len(sv_attack) > n_features:
                sv_attack = sv_attack[:n_features]

            # Select expected_value (base value) corresponding to ATTACK class
            base_vals = explainer.expected_value
            if isinstance(base_vals, (list, np.ndarray, tuple)):
                base_arr = np.asarray(base_vals).reshape(-1)
                if len(base_arr) > attack_idx:
                    base = float(base_arr[attack_idx])
                elif len(base_arr) > 0:
                    base = float(base_arr[-1])
                else:
                    base = 0.0
            else:
                base = float(base_vals)

            return {
                "shap_values": sv_attack,
                "expected_value": float(base),
                "feature_names": list(X_single_trans.columns),
                "feature_values": [float(v) for v in X_single_trans.iloc[0].tolist()],
                "model": "Random Forest (Binary Detection)",
                "class_explained": "P(ATTACK)",
            }
        except Exception as e:
            logger.error(f"RF SHAP error: {e}\n{traceback.format_exc()}")
            return None

    def explain_xgb_local(
        self, X_single_trans: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """
        Generate SHAP explanation for a single record using the XGBoost classifier.

        Returns:
            Dict with keys: shap_values (per class), expected_value, feature_names,
            feature_values, class_names, predicted_class_idx, predicted_class_name
            or None on failure.
        """
        try:
            if self.xgb_classifier is None:
                return None
            sv, base_vals, class_names = self.xgb_classifier.explain_local(X_single_trans)
            pred_code = int(self.xgb_classifier.predict(X_single_trans)[0])
            pred_name = self.xgb_classifier.label_encoder_.inverse_transform([pred_code])[0]
            return {
                "shap_values": sv,
                "expected_value": base_vals,
                "feature_names": list(X_single_trans.columns),
                "feature_values": X_single_trans.iloc[0].tolist(),
                "class_names": class_names,
                "predicted_class_idx": pred_code,
                "predicted_class_name": pred_name,
                "model": "XGBoost (Attack Classification)",
            }
        except Exception as e:
            logger.error(f"XGBoost SHAP error: {e}\n{traceback.format_exc()}")
            return None

    # ── Utility methods ────────────────────────────────────────────────────────

    def get_canonical_columns(self) -> List[str]:
        """Return list of required raw input columns from the preprocessor."""
        if self.preprocessor:
            return self.preprocessor.feature_names_in_
        if self.feature_schema:
            return self.feature_schema.get("canonical_columns", [])
        return []

    def get_artifact_summary(self) -> Dict[str, Any]:
        """Return a human-readable summary of loaded artifacts."""
        return {
            "Preprocessor": "Loaded" if self.status.preprocessor_ok else "MISSING",
            "Random Forest (Binary)": "Loaded" if self.status.rf_ok else "MISSING",
            "Decision Tree (Baseline)": "Loaded" if self.status.dt_ok else "Missing (optional)",
            "XGBoost (Attack Classification)": "Loaded" if self.status.xgb_ok else "Missing",
            "Selected Features": f"{len(self.selected_features)} features" if self.status.selected_features_ok else "MISSING",
            "Feature Schema": "Loaded" if self.status.feature_schema_ok else "Missing",
            "Class Mapping": f"{self.class_mapping.get('n_classes', '?')} attack classes" if self.status.class_mapping_ok else "Missing",
            "Binary Eval Report": "Loaded" if self.status.binary_report_ok else "Missing",
            "Attack Eval Report": "Loaded" if self.status.attack_report_ok else "Missing",
        }
