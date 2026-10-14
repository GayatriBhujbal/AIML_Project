"""Unit tests for DataSplitter."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.splitter import DataSplitter
from src.dataset.schema import TARGET_COLUMN


@pytest.fixture
def synthetic_classification_dataset():
    """Create a balanced and imbalanced synthetic dataset for split testing."""
    np.random.seed(42)
    n_samples = 300
    df = pd.DataFrame({
        "Feature_1": np.random.randn(n_samples),
        "Feature_2": np.random.uniform(10, 100, n_samples),
        TARGET_COLUMN: ["BENIGN"] * 240 + ["DoS Hulk"] * 40 + ["PortScan"] * 15 + ["Heartbleed"] * 5
    })
    return df


def test_split_proportions(synthetic_classification_dataset):
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    train_df, val_df, test_df, report = splitter.split_dataframe(synthetic_classification_dataset)

    total_len = len(synthetic_classification_dataset)
    assert len(train_df) + len(val_df) + len(test_df) == total_len

    # Proportions should be approximately 70%, 15%, 15%
    assert np.isclose(len(train_df) / total_len, 0.70, atol=0.03)
    assert np.isclose(len(val_df) / total_len, 0.15, atol=0.03)
    assert np.isclose(len(test_df) / total_len, 0.15, atol=0.03)


def test_minority_class_preservation(synthetic_classification_dataset):
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    train_df, val_df, test_df, report = splitter.split_dataframe(synthetic_classification_dataset)

    # Heartbleed (minority class with 5 samples) should be present in train
    assert "Heartbleed" in train_df[TARGET_COLUMN].values
    assert report["train_class_counts"]["Heartbleed"] > 0


def test_no_leakage_across_splits(synthetic_classification_dataset):
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42)
    X = synthetic_classification_dataset.drop(columns=[TARGET_COLUMN])
    y = synthetic_classification_dataset[TARGET_COLUMN]

    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_features_and_target(X, y)

    # Ensure no rows overlap by index
    assert len(set(X_train.index).intersection(set(X_val.index))) == 0
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0
    assert len(set(X_val.index).intersection(set(X_test.index))) == 0
