"""Unit tests for DecisionTreeBaseline and RandomForestBinaryDetector."""
import numpy as np
import pandas as pd
import pytest

from src.models.binary_detector import DecisionTreeBaseline, RandomForestBinaryDetector


@pytest.fixture
def synthetic_binary_data():
    """Create reproducible synthetic classification data."""
    np.random.seed(42)
    n_samples = 150
    n_features = 10
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)]
    )
    # Binary target with slight imbalance (80% 0, 20% 1)
    y = np.array([0] * 120 + [1] * 30)
    return X, y


def test_decision_tree_fit_predict(synthetic_binary_data, tmp_path):
    X, y = synthetic_binary_data
    dt = DecisionTreeBaseline(max_depth=5, random_state=42)
    dt.fit(X, y)

    assert dt.is_fitted_

    preds = dt.predict(X)
    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})

    probs = dt.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Test serialization
    save_file = tmp_path / "dt_test.joblib"
    dt.save(save_file)

    loaded_dt = DecisionTreeBaseline.load(save_file)
    assert loaded_dt.is_fitted_
    assert np.array_equal(dt.predict(X), loaded_dt.predict(X))


def test_random_forest_fit_predict_threshold(synthetic_binary_data, tmp_path):
    X, y = synthetic_binary_data
    rf = RandomForestBinaryDetector(n_estimators=20, max_depth=6, random_state=42)
    rf.fit(X, y)

    assert rf.is_fitted_

    preds_default = rf.predict(X)
    assert len(preds_default) == len(y)
    assert set(np.unique(preds_default)).issubset({0, 1})

    probs = rf.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Test custom threshold: lower threshold should increase or keep same positive predictions
    rf.set_threshold(0.20)
    preds_low_thresh = rf.predict(X)
    assert np.sum(preds_low_thresh == 1) >= np.sum(preds_default == 1)

    # Test serialization
    save_file = tmp_path / "rf_test.joblib"
    rf.save(save_file)

    loaded_rf = RandomForestBinaryDetector.load(save_file)
    assert loaded_rf.is_fitted_
    assert loaded_rf.decision_threshold == 0.20
    assert np.array_equal(rf.predict(X), loaded_rf.predict(X))
