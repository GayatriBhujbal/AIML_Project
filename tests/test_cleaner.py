"""Unit tests for DataCleaner."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import DataCleaner
from src.dataset.schema import TARGET_COLUMN


@pytest.fixture
def sample_raw_dataframe():
    """Create a synthetic DataFrame with typical CIC-IDS2017 anomalies."""
    return pd.DataFrame({
        " Destination Port": [80, 443, 22, 80, 80, 80],
        " Flow Duration": [100.0, -50.0, 200.0, 100.0, 100.0, 150.0],  # Has negative value
        " Flow Bytes/s": [10.0, np.inf, 20.0, 10.0, 10.0, np.nan],       # Has inf and nan
        " Flow Packets/s": [2.0, -np.inf, 4.0, 2.0, 2.0, 1.0],           # Has -inf
        "Fwd Header Length": [32, -10, 40, 32, 32, 20],                  # Has negative
        " Fwd Header Length.1": [32, -10, 40, 32, 32, 20],                # Duplicate column
        " Bwd PSH Flags": [0, 0, 0, 0, 0, 0],                             # Constant 0
        " Label": ["BENIGN", "DoS Hulk", "PortScan", "BENIGN", "BENIGN", "Web Attack \ufffd Brute Force"]
    })


def test_cleaner_removes_infs_and_negatives(sample_raw_dataframe):
    cleaner = DataCleaner()
    cleaned_df, audit = cleaner.clean_dataframe(sample_raw_dataframe, is_training=True)

    # Infs must be replaced with NaN
    assert not np.isinf(cleaned_df["Flow Bytes/s"]).any()
    assert not np.isinf(cleaned_df["Flow Packets/s"]).any()

    # Negative flow duration and header length must be clipped to 0
    assert (cleaned_df["Flow Duration"] >= 0).all()
    assert (cleaned_df["Fwd Header Length"] >= 0).all()

    # Duplicate column and constant column must be dropped
    assert "Fwd Header Length.1" not in cleaned_df.columns
    assert "Bwd PSH Flags" not in cleaned_df.columns

    # Labels must be normalized
    assert "Web Attack - Brute Force" in cleaned_df[TARGET_COLUMN].values


def test_cleaner_deduplication(sample_raw_dataframe):
    cleaner = DataCleaner()
    cleaned_df, audit = cleaner.clean_dataframe(sample_raw_dataframe, is_training=True)

    # In sample_raw_dataframe, row 0, 3, and 4 are exact duplicates: 2 must be dropped
    assert audit["exact_duplicates_dropped"] == 2
    assert len(cleaned_df) == len(sample_raw_dataframe) - 2


def test_separate_features_and_target(sample_raw_dataframe):
    cleaner = DataCleaner()
    cleaned_df, _ = cleaner.clean_dataframe(sample_raw_dataframe, is_training=False)

    X, y = cleaner.separate_features_and_target(cleaned_df)

    assert TARGET_COLUMN not in X.columns
    assert y is not None
    assert len(X) == len(y)
    assert len(X) == len(cleaned_df)
