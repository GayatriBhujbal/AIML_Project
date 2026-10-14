"""Unit tests for PreprocessingPipeline."""
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import PreprocessingPipeline
from src.dataset.schema import TARGET_COLUMN


@pytest.fixture
def mock_dataset():
    """Create a sample dataset with missing and infinite values."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "Flow Duration": np.random.uniform(10, 1000, n),
        "Total Fwd Packets": np.random.randint(1, 50, n).astype(float),
        "Flow Bytes/s": [np.inf if i == 0 else (np.nan if i == 1 else 100.0) for i in range(n)],
        "Constant_Feature": [0.0] * n,  # Zero-variance
        TARGET_COLUMN: ["BENIGN"] * 80 + ["DDoS"] * 20
    })
    return df


def test_pipeline_fit_transform_consistency(mock_dataset):
    X = mock_dataset.drop(columns=[TARGET_COLUMN])
    y = mock_dataset[TARGET_COLUMN]

    pipeline = PreprocessingPipeline(scaler_type="robust")
    X_trans = pipeline.fit_transform(X)

    # Constant feature should be dropped
    assert "Constant_Feature" not in X_trans.columns

    # No NaN or Inf should remain in output
    assert not X_trans.isna().any().any()
    assert not np.isinf(X_trans.values).any()

    # Exact same column count and order on transform of new data
    new_data = X.iloc[:5].copy()
    new_trans = pipeline.transform(new_data)
    assert list(new_trans.columns) == list(X_trans.columns)
    assert len(new_trans) == 5


def test_pipeline_selected_features(mock_dataset):
    X = mock_dataset.drop(columns=[TARGET_COLUMN])
    pipeline = PreprocessingPipeline(selected_features=["Flow Duration", "Total Fwd Packets"])
    X_trans = pipeline.fit_transform(X)

    assert list(X_trans.columns) == ["Flow Duration", "Total Fwd Packets"]


def test_label_encoding_and_decoding():
    labels = pd.Series(["BENIGN", "DDoS", "PortScan", "BENIGN", "Web Attack \ufffd Brute Force"])

    # Binary mode
    y_bin = PreprocessingPipeline.encode_labels(labels, mode="binary")
    assert np.array_equal(y_bin, [0, 1, 1, 0, 1])

    decoded_bin = PreprocessingPipeline.decode_labels(y_bin, mode="binary")
    assert decoded_bin == ["BENIGN", "ATTACK", "ATTACK", "BENIGN", "ATTACK"]

    # Multiclass mode
    y_multi = PreprocessingPipeline.encode_labels(labels, mode="multiclass")
    assert y_multi[0] == 0  # BENIGN
    assert y_multi[1] == 3  # DDoS
    assert y_multi[2] == 2  # PortScan
    assert y_multi[4] == 10 # Web Attack - Brute Force


def test_pipeline_save_and_load(mock_dataset, tmp_path):
    X = mock_dataset.drop(columns=[TARGET_COLUMN])
    pipeline = PreprocessingPipeline(scaler_type="robust")
    pipeline.fit(X)

    save_path = tmp_path / "test_preprocessor.joblib"
    pipeline.save(save_path)

    loaded_pipeline = PreprocessingPipeline.load(save_path)
    assert loaded_pipeline.is_fitted_

    trans_orig = pipeline.transform(X.iloc[:10])
    trans_loaded = loaded_pipeline.transform(X.iloc[:10])
    pd.testing.assert_frame_equal(trans_orig, trans_loaded)
