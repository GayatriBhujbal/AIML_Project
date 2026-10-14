"""Data cleaning, validation, anomaly handling, and deduplication for CIC-IDS2017."""
import logging
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd

from src.dataset.schema import (
    TARGET_COLUMN,
    DUPLICATE_COLUMNS,
    CONSTANT_COLUMNS,
    clean_column_name,
    normalize_label
)

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles data cleaning, duplicate detection, infinite value conversion,
    and invalid negative value handling.
    """

    def __init__(
        self,
        drop_duplicate_cols: bool = True,
        drop_constant_cols: bool = True,
        clip_negative_durations: bool = True,
        keep_duplicates: str = "first"
    ):
        self.drop_duplicate_cols = drop_duplicate_cols
        self.drop_constant_cols = drop_constant_cols
        self.clip_negative_durations = clip_negative_durations
        self.keep_duplicates = keep_duplicates

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full cleaning pipeline on DataFrame.
        
        Args:
            df: Input DataFrame
            is_training: If True, executes deduplication. In inference, rows are not dropped.
            
        Returns:
            Tuple of (cleaned_df, cleaning_audit_dict)
        """
        audit: Dict[str, Any] = {
            "initial_rows": len(df),
            "initial_cols": len(df.columns),
            "exact_duplicates_dropped": 0,
            "feature_duplicates_dropped": 0,
            "infs_detected": {},
            "nans_detected": {},
            "negatives_handled": {},
            "cols_dropped": []
        }

        data = df.copy()

        # 1. Standardize column names
        data.columns = [clean_column_name(c) for c in data.columns]

        # 2. Normalize target labels if present
        if TARGET_COLUMN in data.columns:
            data[TARGET_COLUMN] = data[TARGET_COLUMN].apply(normalize_label)

        # 3. Deduplication (applied on training/batch data)
        if is_training:
            init_n = len(data)
            # Remove exact duplicates
            exact_dup_mask = data.duplicated(keep=self.keep_duplicates)
            exact_dup_cnt = int(exact_dup_mask.sum())
            if exact_dup_cnt > 0:
                data = data[~exact_dup_mask].reset_index(drop=True)
                audit["exact_duplicates_dropped"] = exact_dup_cnt
                logger.info(f"Removed {exact_dup_cnt:,} exact duplicate rows.")

            # If target is present, check feature-level duplicates with conflicting labels
            if TARGET_COLUMN in data.columns:
                feature_cols = [c for c in data.columns if c != TARGET_COLUMN]
                feat_dup_mask = data.duplicated(subset=feature_cols, keep=self.keep_duplicates)
                feat_dup_cnt = int(feat_dup_mask.sum())
                if feat_dup_cnt > 0:
                    data = data[~feat_dup_mask].reset_index(drop=True)
                    audit["feature_duplicates_dropped"] = feat_dup_cnt
                    logger.info(f"Removed {feat_dup_cnt:,} feature-level duplicate rows with conflicting labels.")

        # 4. Detect and handle infinite values in numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = int(np.isinf(data[col]).sum())
            if inf_count > 0:
                audit["infs_detected"][col] = inf_count
                data[col] = data[col].replace([np.inf, -np.inf], np.nan)
                logger.info(f"Replaced {inf_count} infinite values in '{col}' with NaN.")

        # 5. Record NaNs
        nan_series = data.isna().sum()
        for col, count in nan_series[nan_series > 0].items():
            audit["nans_detected"][col] = int(count)

        # 6. Handle impossible negative values in network features
        # Note: Init_Win_bytes values of -1 are valid TCP sentinel values (no window)
        # But Flow Duration, Packet Lengths, and IATs cannot physically be negative.
        physical_positive_cols = [
            "Flow Duration", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
            "Fwd IAT Min", "Fwd Header Length", "Bwd Header Length", "min_seg_size_forward"
        ]
        if self.clip_negative_durations:
            for col in physical_positive_cols:
                if col in data.columns:
                    neg_mask = (data[col] < 0) & data[col].notna()
                    neg_count = int(neg_mask.sum())
                    if neg_count > 0:
                        audit["negatives_handled"][col] = neg_count
                        # Clip negative values to zero
                        data.loc[neg_mask, col] = 0.0
                        logger.info(f"Clipped {neg_count} negative values to 0 in '{col}'.")

        # 7. Drop identical duplicate columns (e.g. 'Fwd Header Length.1')
        if self.drop_duplicate_cols:
            for col in DUPLICATE_COLUMNS:
                if col in data.columns:
                    data = data.drop(columns=[col])
                    audit["cols_dropped"].append(f"{col} (duplicate)")
                    logger.info(f"Dropped duplicate column: '{col}'.")

        # 8. Drop constant (zero variance) columns
        if self.drop_constant_cols:
            for col in CONSTANT_COLUMNS:
                if col in data.columns:
                    data = data.drop(columns=[col])
                    audit["cols_dropped"].append(f"{col} (constant)")
                    logger.info(f"Dropped constant column: '{col}'.")

        audit["final_rows"] = len(data)
        audit["final_cols"] = len(data.columns)
        return data, audit

    def separate_features_and_target(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Strictly isolate feature matrix X from target vector y.
        Guarantees that the label is never included in the input feature space.
        """
        if TARGET_COLUMN in df.columns:
            X = df.drop(columns=[TARGET_COLUMN]).copy()
            y = df[TARGET_COLUMN].copy()
            return X, y
        return df.copy(), None
