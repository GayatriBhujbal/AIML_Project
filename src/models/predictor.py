"""Production inference engine and integration interfaces for Member 4 (XAI) and Member 3 (XGBoost)."""
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
    """

    def __init__(
        self,
        preprocessor_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        selected_features_path: Optional[Union[str, Path]] = None
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

    def validate_and_transform(self, raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
        """
        Validate input format, verify required features, and apply preprocessing.
        
        Args:
            raw_input: Raw network flow DataFrame, list of dicts, or single flow dict.
            
        Returns:
            Preprocessed DataFrame with shape (N, 25) aligned to self.selected_features.
        """
        # 1. Format normalization
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
            raise TypeError(f"Unsupported input type: {type(raw_input)}. Expected DataFrame, list of dicts, or dict.")

        # Clean column names
        df.columns = [clean_column_name(c) for c in df.columns]

        # Remove target label if accidentally passed in inference data
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])

        # 2. Check if input is ALREADY preprocessed
        if set(self.selected_features).issubset(set(df.columns)) and len(df.columns) == len(self.selected_features):
            # Input already has the 25 preprocessed features; ensure correct column ordering
            return df[self.selected_features]

        # 3. Verify minimum required features for raw flow input
        missing_in_pipeline = [
            col for col in self.preprocessor.feature_names_in_
            if col not in df.columns and col in self.selected_features
        ]
        if missing_in_pipeline:
            raise ValueError(
                f"Missing critical flow features required for threat detection: {missing_in_pipeline[:5]} "
                f"(total missing: {len(missing_in_pipeline)}). "
                f"Please ensure flow records contain the canonical CIC-IDS2017 features."
            )

        # 4. Transform through fitted PreprocessingPipeline
        transformed_df = self.preprocessor.transform(df)

        # Ensure exact column ordering
        return transformed_df[self.selected_features]

    def predict(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict binary threat labels (0: BENIGN, 1: ATTACK).
        """
        X_trans = self.validate_and_transform(raw_input)
        return self.detector.predict(X_trans, threshold=threshold)

    def predict_labels(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Predict readable class labels ('BENIGN' or 'ATTACK').
        """
        preds = self.predict(raw_input, threshold=threshold)
        return PreprocessingPipeline.decode_labels(preds, mode="binary")

    def predict_proba(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]
    ) -> np.ndarray:
        """
        Predict class probabilities: [P(BENIGN), P(ATTACK)].
        """
        X_trans = self.validate_and_transform(raw_input)
        return self.detector.predict_proba(X_trans)

    def get_explainer_inputs(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, RandomForestClassifier]:
        """
        Stable interface for Member 4 (Explainable AI / SHAP & LIME).
        
        Returns:
            Tuple of:
            - preprocessed_df: DataFrame with exact feature names and scaled values
            - sklearn_estimator: Underlying trained RandomForestClassifier instance
        """
        X_trans = self.validate_and_transform(raw_input)
        return X_trans, self.detector.model

    def route_to_attack_classifier(
        self,
        raw_input: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Stable hierarchical hand-off interface for Member 3 (XGBoost Multiclass Classifier).
        
        Filters the incoming traffic stream:
        - Benign flows are safely logged/passed.
        - Attack flows (prediction == 1) are routed to Member 3.
        
        Returns:
            Tuple of:
            - attack_features: Preprocessed DataFrame containing ONLY flows predicted as ATTACK
            - attack_indices: Array of original indices corresponding to the attack flows
            - all_transformed: Full preprocessed DataFrame for all input flows
        """
        X_trans = self.validate_and_transform(raw_input)
        preds = self.detector.predict(X_trans, threshold=threshold)

        attack_mask = (preds == 1)
        attack_features = X_trans[attack_mask].copy()
        attack_indices = X_trans.index[attack_mask].to_numpy()

        logger.info(f"Routed {len(attack_features):,} of {len(X_trans):,} flows ({(len(attack_features)/len(X_trans))*100:.2f}%) to Member 3.")
        return attack_features, attack_indices, X_trans
