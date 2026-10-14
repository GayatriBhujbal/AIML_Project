"""Reproducible, leakage-safe stratified data splitting for CIC-IDS2017."""
import logging
from typing import Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.dataset_config import (
    DEFAULT_RANDOM_STATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)
from src.dataset.schema import TARGET_COLUMN

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Creates stratified, leakage-free training, validation, and test splits.
    Ensures severe class imbalance is accounted for so rare classes (e.g. Heartbleed, Infiltration)
    are present across all subsets.
    """

    def __init__(
        self,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
        random_state: int = DEFAULT_RANDOM_STATE
    ):
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split ratios must sum to 1.0. Given: {train_ratio} + {val_ratio} + {test_ratio} = {total}")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def split_dataframe(
        self,
        df: pd.DataFrame,
        stratify_col: str = TARGET_COLUMN,
        reset_index: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Split a full DataFrame into train, val, and test subsets with stratification.
        
        Returns:
            Tuple of (train_df, val_df, test_df, split_report_dict)
        """
        if stratify_col not in df.columns:
            raise KeyError(f"Stratification column '{stratify_col}' not found in DataFrame.")

        # Compute temporary test ratio (val + test combined)
        temp_ratio = self.val_ratio + self.test_ratio
        # Ratio of test within the temporary portion
        test_within_temp = self.test_ratio / temp_ratio

        # Identify classes with extremely low sample counts (< 3 samples)
        class_counts = df[stratify_col].value_counts()
        rare_classes = class_counts[class_counts < 3].index.tolist()

        if rare_classes:
            logger.warning(
                f"Classes {rare_classes} have fewer than 3 samples. "
                "Stratified splitting will allocate samples carefully."
            )
            # For ultra-rare classes (< 3 rows), put them in train to ensure model sees them
            regular_mask = ~df[stratify_col].isin(rare_classes)
            df_regular = df[regular_mask]
            df_rare = df[~regular_mask]

            train_reg, temp_reg = train_test_split(
                df_regular,
                test_size=temp_ratio,
                random_state=self.random_state,
                stratify=df_regular[stratify_col]
            )

            val_reg, test_reg = train_test_split(
                temp_reg,
                test_size=test_within_temp,
                random_state=self.random_state,
                stratify=temp_reg[stratify_col]
            )

            train_df = pd.concat([train_reg, df_rare])
            val_df = val_reg
            test_df = test_reg
        else:
            train_df, temp_df = train_test_split(
                df,
                test_size=temp_ratio,
                random_state=self.random_state,
                stratify=df[stratify_col]
            )

            val_df, test_df = train_test_split(
                temp_df,
                test_size=test_within_temp,
                random_state=self.random_state,
                stratify=temp_df[stratify_col]
            )

        # Generate split report
        report = {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_percent": (len(train_df) / len(df)) * 100,
            "val_percent": (len(val_df) / len(df)) * 100,
            "test_percent": (len(test_df) / len(df)) * 100,
            "train_class_counts": train_df[stratify_col].value_counts().to_dict(),
            "val_class_counts": val_df[stratify_col].value_counts().to_dict(),
            "test_class_counts": test_df[stratify_col].value_counts().to_dict()
        }

        if reset_index:
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

        return train_df, val_df, test_df, report

    def split_features_and_target(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        reset_index: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split X and y matrices directly into (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        df = X.copy()
        df["_target_temp"] = y.values
        train_df, val_df, test_df, _ = self.split_dataframe(
            df, stratify_col="_target_temp", reset_index=reset_index
        )

        X_train = train_df.drop(columns=["_target_temp"])
        y_train = train_df["_target_temp"]

        X_val = val_df.drop(columns=["_target_temp"])
        y_val = val_df["_target_temp"]

        X_test = test_df.drop(columns=["_target_temp"])
        y_test = test_df["_target_temp"]

        return X_train, X_val, X_test, y_train, y_val, y_test
