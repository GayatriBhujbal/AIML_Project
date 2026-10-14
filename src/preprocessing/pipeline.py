"""Production-grade reusable preprocessing pipeline for CIC-IDS2017."""
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler

from config.dataset_config import MODELS_DIR
from src.dataset.schema import (
    LABEL_MAPPING,
    INVERSE_LABEL_MAPPING,
    BINARY_MAPPING,
    ATTACK_CATEGORY_MAPPING,
    TARGET_COLUMN,
    clean_column_name,
    normalize_label
)
from src.preprocessing.cleaner import DataCleaner

logger = logging.getLogger(__name__)


class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Leakage-free preprocessing pipeline that fits imputers, scalers, and column
    alignments strictly on training data and reuses them for validation, testing,
    and production inference.
    """

    def __init__(
        self,
        scaler_type: str = "robust",
        selected_features: Optional[List[str]] = None,
        drop_duplicate_cols: bool = True,
        drop_constant_cols: bool = True,
        clip_negative_durations: bool = True
    ):
        self.scaler_type = scaler_type
        self.selected_features = selected_features
        self.drop_duplicate_cols = drop_duplicate_cols
        self.drop_constant_cols = drop_constant_cols
        self.clip_negative_durations = clip_negative_durations

        # Cleaner helper
        self.cleaner = DataCleaner(
            drop_duplicate_cols=drop_duplicate_cols,
            drop_constant_cols=drop_constant_cols,
            clip_negative_durations=clip_negative_durations
        )

        # Fitted attributes
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.impute_medians_: Dict[str, float] = {}
        self.zero_variance_cols_: List[str] = []
        self.scaler_: Optional[Union[RobustScaler, StandardScaler]] = None
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "PreprocessingPipeline":
        """
        Fit imputer medians, zero-variance filters, and scaler strictly on training features.
        Target label is never used or passed to feature transformations.
        """
        logger.info("Fitting PreprocessingPipeline on training features...")
        df = X.copy()
        df.columns = [clean_column_name(c) for c in df.columns]

        # Ensure target is not present in X
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])

        # Clean duplicates/constants/negatives using cleaner
        cleaned_df, _ = self.cleaner.clean_dataframe(df, is_training=False)

        # Handle infinite values: replace with NaN for imputer fitting
        cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)

        # Select only numeric features
        numeric_df = cleaned_df.select_dtypes(include=[np.number])
        self.feature_names_in_ = list(numeric_df.columns)

        # 1. Compute training median imputation values
        self.impute_medians_ = numeric_df.median().to_dict()
        imputed_df = numeric_df.fillna(self.impute_medians_)

        # 2. Identify zero-variance features on training data
        variances = imputed_df.var()
        self.zero_variance_cols_ = variances[variances == 0].index.tolist()
        if self.zero_variance_cols_:
            logger.info(f"Identified {len(self.zero_variance_cols_)} zero-variance features on train: {self.zero_variance_cols_}")
            active_df = imputed_df.drop(columns=self.zero_variance_cols_)
        else:
            active_df = imputed_df

        # 3. Filter to explicitly selected features if specified
        if self.selected_features is not None:
            # BUG-5 FIX: raise an explicit error if a required selected feature is absent
            # after preprocessing (e.g. it was zero-variance in the training sample).
            # Previously this silently produced a smaller-than-expected feature set.
            missing_selected = [f for f in self.selected_features if f not in active_df.columns]
            if missing_selected:
                raise ValueError(
                    f"BUG-5: {len(missing_selected)} selected feature(s) are absent after "
                    f"preprocessing: {missing_selected}. "
                    "Possible causes: (a) these features are zero-variance in the training "
                    "sample you supplied -- enlarge the sample, (b) the selected_features.json "
                    "was generated from a different dataset sample than the one used here. "
                    "Re-run run_feature_selection.py on the same data to regenerate it."
                )
            active_df = active_df[self.selected_features]
            self.feature_names_out_ = list(self.selected_features)
        else:
            self.feature_names_out_ = list(active_df.columns)

        # 4. Fit scaler on active features
        if self.scaler_type == "robust":
            self.scaler_ = RobustScaler()
        elif self.scaler_type == "standard":
            self.scaler_ = StandardScaler()
        elif self.scaler_type == "none" or self.scaler_type is None:
            self.scaler_ = None
        else:
            raise ValueError(f"Unknown scaler_type '{self.scaler_type}'. Choose 'robust', 'standard', or 'none'.")

        if self.scaler_ is not None:
            self.scaler_.fit(active_df)

        self.is_fitted_ = True
        logger.info(f"Pipeline fitted successfully. Feature count: {len(self.feature_names_out_)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw validation, test, or inference data using fitted statistics.
        Guarantees identical feature ordering, imputation, and scaling.
        """
        if not self.is_fitted_:
            raise RuntimeError("PreprocessingPipeline must be fitted before calling transform().")

        df = X.copy()
        df.columns = [clean_column_name(c) for c in df.columns]

        # Isolate from target column if present
        if TARGET_COLUMN in df.columns:
            df = df.drop(columns=[TARGET_COLUMN])

        # Clean using same rules (no row-dropping in inference)
        cleaned_df, _ = self.cleaner.clean_dataframe(df, is_training=False)
        cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)

        # Align with training input columns
        for col in self.feature_names_in_:
            if col not in cleaned_df.columns:
                cleaned_df[col] = self.impute_medians_.get(col, 0.0)

        numeric_df = cleaned_df[self.feature_names_in_].copy()

        # Apply fitted training medians for imputation
        for col, median_val in self.impute_medians_.items():
            numeric_df[col] = numeric_df[col].fillna(median_val)

        # Drop zero-variance columns identified during fit
        if self.zero_variance_cols_:
            numeric_df = numeric_df.drop(columns=[c for c in self.zero_variance_cols_ if c in numeric_df.columns])

        # Align to final selected feature ordering
        final_df = numeric_df[self.feature_names_out_].copy()

        # Apply fitted scaler
        if self.scaler_ is not None:
            scaled_vals = self.scaler_.transform(final_df)
            result_df = pd.DataFrame(scaled_vals, columns=self.feature_names_out_, index=final_df.index)
        else:
            result_df = final_df

        return result_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit on X and return transformed DataFrame."""
        return self.fit(X, y).transform(X)

    @staticmethod
    def encode_labels(
        y: pd.Series,
        mode: str = "binary"
    ) -> np.ndarray:
        """
        Encode string labels to integers.
        
        Args:
            y: Series of raw string labels
            mode: 'binary' (0: Benign, 1: Attack),
                  'multiclass' (0..14),
                  or 'category' (7 high-level classes)
                  
        Returns:
            NumPy array of integer label codes
        """
        cleaned_y = y.apply(normalize_label)

        if mode == "binary":
            return (cleaned_y != "BENIGN").astype(int).values
        elif mode == "multiclass":
            return cleaned_y.map(LABEL_MAPPING).fillna(-1).astype(int).values
        elif mode == "category":
            category_names = sorted(list(set(ATTACK_CATEGORY_MAPPING.values())))
            cat_map = {cat: i for i, cat in enumerate(category_names)}
            return cleaned_y.map(ATTACK_CATEGORY_MAPPING).map(cat_map).fillna(-1).astype(int).values
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'binary', 'multiclass', or 'category'.")

    @staticmethod
    def decode_labels(
        y_encoded: Union[np.ndarray, List[int]],
        mode: str = "binary"
    ) -> List[str]:
        """Decode integer predictions back to readable string labels."""
        if mode == "binary":
            return ["BENIGN" if val == 0 else "ATTACK" for val in y_encoded]
        elif mode == "multiclass":
            return [INVERSE_LABEL_MAPPING.get(val, "UNKNOWN") for val in y_encoded]
        elif mode == "category":
            category_names = sorted(list(set(ATTACK_CATEGORY_MAPPING.values())))
            return [category_names[val] if 0 <= val < len(category_names) else "UNKNOWN" for val in y_encoded]
        else:
            raise ValueError(f"Unknown mode '{mode}'.")

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Serialize fitted pipeline to disk.

        BUG-6 FIX: also saves a sidecar JSON schema file
        (<artifact>.schema.json) alongside the .joblib so that schema
        mismatches can be detected at load time.
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted pipeline.")
        out_path = Path(filepath) if filepath else (MODELS_DIR / "preprocessor.joblib")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path)
        # Save schema sidecar
        schema = {
            "feature_names_in_": self.feature_names_in_,
            "feature_names_out_": self.feature_names_out_,
            "zero_variance_cols_": self.zero_variance_cols_,
            "selected_features": self.selected_features,
            "scaler_type": self.scaler_type,
        }
        schema_path = out_path.with_suffix(".schema.json")
        import json as _json
        with open(schema_path, "w", encoding="utf-8") as f:
            _json.dump(schema, f, indent=2)
        logger.info(f"Saved fitted PreprocessingPipeline to '{out_path}'.")
        logger.info(f"Saved pipeline schema sidecar to '{schema_path}'.")
        return out_path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "PreprocessingPipeline":
        """Load fitted pipeline from disk.

        BUG-6 FIX: validates the sidecar schema if it exists so that
        incompatible artifacts are detected at load time rather than causing
        silent mispredictions.
        """
        in_path = Path(filepath) if filepath else (MODELS_DIR / "preprocessor.joblib")
        if not in_path.exists():
            raise FileNotFoundError(f"Pipeline artifact not found at '{in_path}'.")
        pipeline = joblib.load(in_path)
        # Validate sidecar schema if present
        schema_path = in_path.with_suffix(".schema.json")
        if schema_path.exists():
            import json as _json
            with open(schema_path, "r", encoding="utf-8") as f:
                saved_schema = _json.load(f)
            live_out = list(pipeline.feature_names_out_)
            saved_out = saved_schema.get("feature_names_out_", [])
            if live_out != saved_out:
                raise RuntimeError(
                    f"Pipeline artifact schema mismatch: the loaded .joblib has "
                    f"{len(live_out)} output features but the sidecar schema records "
                    f"{len(saved_out)}. The artifact may be stale. Refit the pipeline."
                )
        logger.info(f"Loaded PreprocessingPipeline from '{in_path}'.")
        return pipeline
