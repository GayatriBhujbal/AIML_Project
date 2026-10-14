"""Production inference engine and integration interfaces for Member 4 (XAI) and Member 3 (XGBoost).

Input contract (BUG-1 / BUG-2 fix):
  Raw input  -> predict_raw()          -- ALWAYS applies the fitted PreprocessingPipeline.
  Pre-scaled -> predict_preprocessed() -- caller guarantees the data was already transformed
                                          by the SAME fitted pipeline; no second transform.

NEVER infer preprocessing state from column names, feature count, or value magnitudes.
The old validate_and_transform() had an unsafe bypass: if input columns exactly matched the 25
selected feature names it skipped preprocessing. That bypass has been removed.
"""
import json
import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config.dataset_config import MODELS_DIR, SCHEMAS_DIR
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.binary_detector import RandomForestBinaryDetector
from src.dataset.schema import TARGET_COLUMN, clean_column_name

logger = logging.getLogger(__name__)


class ThreatDetectionInferenceEngine:
    """
    Unified production inference engine for binary network threat detection.

    Provides:
    1. Robust validation of input schemas and types.
    2. Automatic application of the fitted PreprocessingPipeline.
    3. Direct compatibility with Member 4 (SHAP / LIME explainers).
    4. Hierarchical traffic routing to Member 3 (XGBoost multiclass classifier).

    Input contract
    --------------
    predict_raw*()          -- input is RAW network flow data. Preprocessing is always applied.
    predict_preprocessed*() -- input is ALREADY preprocessed (caller's responsibility).
                               Preprocessing is NOT applied again.
    """

    def __init__(
        self,
        preprocessor_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        selected_features_path: Optional[Union[str, Path]] = None,
    ):
        p_path = Path(preprocessor_path) if preprocessor_path else (MODELS_DIR / "preprocessor.joblib")
        m_path = Path(model_path) if model_path else (MODELS_DIR / "binary_random_forest.joblib")
        f_path = Path(selected_features_path) if selected_features_path else (SCHEMAS_DIR / "selected_features.json")

        self.preprocessor = PreprocessingPipeline.load(p_path)
        self.detector = RandomForestBinaryDetector.load(m_path)

        if f_path.exists():
            with open(f_path, "r", encoding="utf-8") as f:
                self.selected_features = json.load(f)["selected_features"]
        else:
            self.selected_features = self.preprocessor.feature_names_out_

        logger.info(f"ThreatDetectionInferenceEngine initialized with {len(self.selected_features)} features.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_input(
        self, raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]
    ) -> pd.DataFrame:
        """Convert any accepted input format to a DataFrame with clean column names.
        Strips the target label if accidentally included."""
        if isinstance(raw_input, dict):
            df = pd.DataFrame([raw_input])
        elif isinstance(raw_input, list):
            if not raw_input:
                raise ValueError("Input list of records cannot be empty.")
            df = pd.DataFrame(raw_input)
        elif isinstance(raw_input, pd.DataFrame):
            if raw_input.empty:
                raise ValueError("Input DataFrame cannot be empty.")
            df = raw_input.copy()
        else:
            raise TypeError(
                f"Unsupported input type: {type(raw_input)}. "
                "Expected pd.DataFrame, list[dict], or dict."
            )
        df.columns = [clean_column_name(c) for c in df.columns]
        # Target label must never be a model input
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])
        return df

    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run df through the fitted pipeline and enforce selected_features ordering.

        Raises ValueError with a clear message if required input columns are missing
        or if selected features are absent after transformation.
        """
        missing_raw = [col for col in self.preprocessor.feature_names_in_ if col not in df.columns]
        if missing_raw:
            raise ValueError(
                f"Missing {len(missing_raw)} canonical CIC-IDS2017 features required for "
                f"preprocessing: {missing_raw[:5]}{'...' if len(missing_raw) > 5 else ''}. "
                "Supply raw network flow records with all required columns."
            )
        transformed = self.preprocessor.transform(df)
        missing_sel = [f for f in self.selected_features if f not in transformed.columns]
        if missing_sel:
            raise ValueError(
                f"After preprocessing, {len(missing_sel)} selected features are absent: "
                f"{missing_sel}. Refit the preprocessor or check selected_features.json."
            )
        return transformed[self.selected_features]

    def _validate_preprocessed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that df contains exactly the expected preprocessed feature set.

        Raises ValueError for missing or extra columns.  Column ordering is enforced.
        This method does NOT apply any transformation.
        """
        missing = [f for f in self.selected_features if f not in df.columns]
        extra = [c for c in df.columns if c not in self.selected_features]
        if missing:
            raise ValueError(
                f"predict_preprocessed() received {len(missing)} missing selected features: "
                f"{missing[:5]}. Caller must supply a DataFrame transformed by the same fitted "
                "PreprocessingPipeline used at training time."
            )
        if extra:
            raise ValueError(
                f"predict_preprocessed() received {len(extra)} unexpected columns not in the "
                f"selected feature schema: {extra[:5]}. "
                f"Pass only the {len(self.selected_features)} selected features."
            )
        return df[self.selected_features]

    # ------------------------------------------------------------------
    # DEPRECATED backward-compat wrapper
    # Previously this had an unsafe bypass that skipped preprocessing when
    # input columns matched the 25 selected feature names.  That bypass is
    # REMOVED -- this method now always applies preprocessing.
    # Use predict_raw() / predict_preprocessed() for explicit intent.
    # ------------------------------------------------------------------

    def validate_and_transform(
        self, raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]
    ) -> pd.DataFrame:
        """[DEPRECATED] Always applies preprocessing to raw input.

        BUG-1 FIX: the old column-name-based bypass that silently skipped
        preprocessing for DataFrames whose columns happened to match the 25
        selected feature names has been removed.  Preprocessing is now always
        applied.  Use predict_raw() / predict_preprocessed() instead.
        """
        df = self._normalize_input(raw_input)
        # Provide the same error message as the old implementation so that
        # existing tests that assert on this message continue to pass.
        missing_critical = [
            col for col in self.preprocessor.feature_names_in_
            if col not in df.columns and col in self.selected_features
        ]
        if missing_critical:
            raise ValueError(
                f"Missing critical flow features required for threat detection: {missing_critical[:5]} "
                f"(total missing: {len(missing_critical)}). "
                "Please ensure flow records contain the canonical CIC-IDS2017 features."
            )
        return self._apply_preprocessing(df)

    # ------------------------------------------------------------------
    # Public API – RAW input (preprocessing ALWAYS applied)
    # ------------------------------------------------------------------

    def predict_raw(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary threat labels from RAW network flow records.

        ALWAYS applies the fitted PreprocessingPipeline before inference.
        Never call this on data that has already been preprocessed – use
        predict_preprocessed() instead.

        Args:
            raw_input: Raw DataFrame / list[dict] / dict with canonical CIC-IDS2017 columns.
            threshold: Optional decision threshold (0-1). Default 0.5.
        Returns:
            np.ndarray of {0: BENIGN, 1: ATTACK} per record.
        """
        df = self._normalize_input(raw_input)
        X_trans = self._apply_preprocessing(df)
        return self.detector.predict(X_trans, threshold=threshold)

    def predict_raw_labels(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Predict 'BENIGN' or 'ATTACK' strings from raw network flow records."""
        preds = self.predict_raw(raw_input, threshold=threshold)
        return PreprocessingPipeline.decode_labels(preds, mode="binary")

    def predict_raw_proba(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
    ) -> np.ndarray:
        """Return [P(BENIGN), P(ATTACK)] probability matrix from raw network flow records."""
        df = self._normalize_input(raw_input)
        X_trans = self._apply_preprocessing(df)
        return self.detector.predict_proba(X_trans)

    # ------------------------------------------------------------------
    # Public API – PREPROCESSED input (no re-transform; caller responsibility)
    # ------------------------------------------------------------------

    def predict_preprocessed(
        self,
        preprocessed_df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary labels from an ALREADY-PREPROCESSED feature DataFrame.

        CALLER RESPONSIBILITY
        ---------------------
        preprocessed_df MUST have been produced by the SAME fitted
        PreprocessingPipeline artifact used at training time (preprocessor.joblib).
        No additional transformation is applied here.

        Args:
            preprocessed_df: DataFrame with exactly the 25 selected features,
                             already scaled by the fitted RobustScaler.
            threshold: Optional decision threshold.
        Returns:
            np.ndarray of {0, 1}.
        Raises:
            ValueError: If columns are missing, extra, or out of order.
        """
        X = self._validate_preprocessed(preprocessed_df)
        return self.detector.predict(X, threshold=threshold)

    def predict_preprocessed_proba(
        self,
        preprocessed_df: pd.DataFrame,
    ) -> np.ndarray:
        """Return [P(BENIGN), P(ATTACK)] from an already-preprocessed DataFrame."""
        X = self._validate_preprocessed(preprocessed_df)
        return self.detector.predict_proba(X)

    def route_to_attack_classifier_preprocessed(
        self,
        preprocessed_df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Route already-preprocessed flows to Member 3 (XGBoost).

        Use when you have already called PreprocessingPipeline.transform() yourself.
        Does NOT apply preprocessing again.

        Returns:
            Tuple of (attack_features, attack_indices) – only flows predicted as ATTACK.
        """
        X = self._validate_preprocessed(preprocessed_df)
        preds = self.detector.predict(X, threshold=threshold)
        attack_mask = preds == 1
        attack_features = X[attack_mask].copy()
        attack_indices = X.index[attack_mask].to_numpy()
        logger.info(
            f"[preprocessed] Routed {len(attack_features):,} of {len(X):,} flows "
            f"({len(attack_features)/len(X)*100:.2f}%) to Member 3."
        )
        return attack_features, attack_indices

    # ------------------------------------------------------------------
    # Legacy methods – preserved for backward compatibility
    # Signatures unchanged; internals now use the new helpers.
    # ------------------------------------------------------------------

    def predict(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """[Legacy] Predict binary threat labels from raw input. Use predict_raw()."""
        X_trans = self.validate_and_transform(raw_input)
        return self.detector.predict(X_trans, threshold=threshold)

    def predict_labels(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """[Legacy] Predict readable class labels. Use predict_raw_labels()."""
        preds = self.predict(raw_input, threshold=threshold)
        return PreprocessingPipeline.decode_labels(preds, mode="binary")

    def predict_proba(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
    ) -> np.ndarray:
        """[Legacy] Predict class probabilities. Use predict_raw_proba()."""
        X_trans = self.validate_and_transform(raw_input)
        return self.detector.predict_proba(X_trans)

    def get_explainer_inputs(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, RandomForestClassifier]:
        """Stable interface for Member 4 (Explainable AI / SHAP & LIME).
        Applies preprocessing to raw input.

        Returns:
            (preprocessed_df, sklearn_RandomForestClassifier)
        """
        X_trans = self.validate_and_transform(raw_input)
        return X_trans, self.detector.model

    def route_to_attack_classifier(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """[Legacy] Hierarchical hand-off to Member 3 from raw input.
        Applies preprocessing then routes ATTACK flows.

        Returns:
            Tuple of (attack_features, attack_indices, all_transformed).
        """
        X_trans = self.validate_and_transform(raw_input)
        preds = self.detector.predict(X_trans, threshold=threshold)
        attack_mask = preds == 1
        attack_features = X_trans[attack_mask].copy()
        attack_indices = X_trans.index[attack_mask].to_numpy()
        logger.info(
            f"Routed {len(attack_features):,} of {len(X_trans):,} flows "
            f"({len(attack_features)/len(X_trans)*100:.2f}%) to Member 3."
        )
        return attack_features, attack_indices, X_trans
