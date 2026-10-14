"""Unit and integration tests for ThreatDetectionInferenceEngine and Member 4 / 3 interfaces."""
import json
import numpy as np
import pandas as pd
import pytest

from config.dataset_config import MODELS_DIR, SCHEMAS_DIR
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.binary_detector import RandomForestBinaryDetector
from src.models.predictor import ThreatDetectionInferenceEngine


@pytest.fixture
def mock_trained_system(tmp_path):
    """Set up temporary preprocessor and binary detector for testing."""
    with open(SCHEMAS_DIR / "selected_features.json", "r", encoding="utf-8") as f:
        selected_features = json.load(f)["selected_features"]

    # Create dummy raw data containing the 25 selected features
    n = 30
    df = pd.DataFrame({
        feat: np.random.uniform(1.0, 100.0, n) for feat in selected_features
    })
    y = np.array([0] * 20 + [1] * 10)

    # Fit pipeline
    pipeline = PreprocessingPipeline(scaler_type="robust", selected_features=selected_features)
    pipeline.fit(df)
    p_path = tmp_path / "preprocessor.joblib"
    pipeline.save(p_path)

    # Fit detector
    X_trans = pipeline.transform(df)
    detector = RandomForestBinaryDetector(n_estimators=10, max_depth=4, random_state=42)
    detector.fit(X_trans, y)
    m_path = tmp_path / "binary_rf.joblib"
    detector.save(m_path)

    engine = ThreatDetectionInferenceEngine(
        preprocessor_path=p_path,
        model_path=m_path,
        selected_features_path=SCHEMAS_DIR / "selected_features.json"
    )
    return engine, df, selected_features


def test_inference_engine_predict_dataframe(mock_trained_system):
    engine, df, _ = mock_trained_system
    preds = engine.predict(df)
    assert len(preds) == len(df)
    assert set(np.unique(preds)).issubset({0, 1})

    labels = engine.predict_labels(df)
    assert len(labels) == len(df)
    assert all(l in ["BENIGN", "ATTACK"] for l in labels)

    probs = engine.predict_proba(df)
    assert probs.shape == (len(df), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_inference_engine_predict_dict(mock_trained_system):
    engine, df, _ = mock_trained_system
    single_record = df.iloc[0].to_dict()

    pred = engine.predict(single_record)
    assert len(pred) == 1
    assert pred[0] in [0, 1]

    label = engine.predict_labels(single_record)
    assert label[0] in ["BENIGN", "ATTACK"]


def test_inference_engine_empty_input_error(mock_trained_system):
    engine, _, _ = mock_trained_system
    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        engine.predict(pd.DataFrame())

    with pytest.raises(ValueError, match="Input list of records cannot be empty"):
        engine.predict([])


def test_inference_engine_missing_critical_features(mock_trained_system):
    engine, _, _ = mock_trained_system
    invalid_df = pd.DataFrame({"Unknown_Feature": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing critical flow features"):
        engine.predict(invalid_df)


def test_member_4_explainer_interface(mock_trained_system):
    """Verify contract for Member 4: returns preprocessed DataFrame and sklearn estimator."""
    engine, df, selected_features = mock_trained_system
    X_trans, estimator = engine.get_explainer_inputs(df.iloc[:5])

    assert isinstance(X_trans, pd.DataFrame)
    assert list(X_trans.columns) == selected_features
    assert X_trans.shape == (5, len(selected_features))
    assert hasattr(estimator, "estimators_")  # Confirms it is a valid RandomForestClassifier


def test_member_3_routing_interface(mock_trained_system):
    """Verify contract for Member 3: routes only attack flows to XGBoost."""
    engine, df, selected_features = mock_trained_system
    attack_feats, attack_indices, all_trans = engine.route_to_attack_classifier(df)

    assert isinstance(attack_feats, pd.DataFrame)
    assert len(attack_feats) == len(attack_indices)
    assert list(attack_feats.columns) == selected_features
    # Attack features must be a subset of all flows
    assert len(attack_feats) <= len(all_trans)
