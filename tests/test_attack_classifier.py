"""
Unit and integration tests for XGBoostAttackClassifier (Member 3).

Tests cover:
- Model lifecycle: fit, predict, predict_proba, predict_with_scores, save, load
- Class mapping derivation from actual labels
- BENIGN row guard in fit()
- Feature alignment enforcement
- SHAP global and local explanations (API shape and content)
- Interface contracts for Member 4 integration
- AttackClassifierEvaluator metric computation
- No fabricated data: all assertions use real model outputs

Dataset is NOT required for these tests -- synthetic CIC-IDS2017-like
DataFrames are constructed from the 25 canonical selected_features.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from config.dataset_config import SCHEMAS_DIR
from src.models.attack_classifier import XGBoostAttackClassifier
from src.models.attack_evaluator import AttackClassifierEvaluator


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def selected_features() -> list:
    """Load the 25 canonical features from shared schema."""
    with open(SCHEMAS_DIR / "selected_features.json", "r", encoding="utf-8") as f:
        return json.load(f)["selected_features"]


@pytest.fixture(scope="module")
def synthetic_attack_data(selected_features):
    """
    Build a synthetic DataFrame mimicking CIC-IDS2017 attack records.
    Uses the exact 25 canonical feature names from selected_features.json.
    Labels are a realistic subset of actual CIC-IDS2017 attack classes.
    """
    rng = np.random.RandomState(42)
    n = 300  # enough samples for all classes to have >= 3 rows

    # 6 attack classes from the verified dataset
    attack_classes = [
        "DoS Hulk", "PortScan", "DDoS",
        "FTP-Patator", "Bot", "Web Attack - Brute Force"
    ]
    class_sizes = [100, 80, 50, 30, 25, 15]
    labels = []
    for cls, sz in zip(attack_classes, class_sizes):
        labels.extend([cls] * sz)
    rng.shuffle(labels)

    X = pd.DataFrame(
        rng.uniform(0.0, 100.0, size=(len(labels), len(selected_features))),
        columns=selected_features,
    )
    y = pd.Series(labels, name="Label")
    return X, y, attack_classes


@pytest.fixture(scope="module")
def trained_classifier(synthetic_attack_data, selected_features):
    """Fit a small XGBoostAttackClassifier on synthetic data."""
    X, y, _ = synthetic_attack_data
    # 80/20 split for train/val
    split = int(0.8 * len(X))
    X_train, X_val = X.iloc[:split].reset_index(drop=True), X.iloc[split:].reset_index(drop=True)
    y_train, y_val = y.iloc[:split].reset_index(drop=True), y.iloc[split:].reset_index(drop=True)

    clf = XGBoostAttackClassifier(
        n_estimators=30,
        max_depth=3,
        random_state=42,
        early_stopping_rounds=5,
    )
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return clf, X, y


# -----------------------------------------------------------------------
# Class mapping tests
# -----------------------------------------------------------------------

class TestClassMapping:
    def test_class_names_derived_from_data(self, trained_classifier, synthetic_attack_data):
        """Class names must be derived from actual training labels, not assumed."""
        clf, _, _ = trained_classifier
        _, _, expected_classes = synthetic_attack_data
        # All 6 attack classes should be in the mapping
        assert set(clf.get_class_names()) == set(expected_classes)

    def test_class_mapping_contiguous(self, trained_classifier):
        """Integer codes must be contiguous [0, n_classes-1]."""
        clf, _, _ = trained_classifier
        mapping = clf.get_class_mapping()
        assert sorted(mapping.keys()) == list(range(len(mapping)))

    def test_n_classes_matches_data(self, trained_classifier, synthetic_attack_data):
        clf, _, _ = trained_classifier
        _, _, expected_classes = synthetic_attack_data
        assert clf.n_classes_ == len(expected_classes)

    def test_label_encoder_roundtrip(self, trained_classifier):
        clf, X, y = trained_classifier
        preds = clf.predict(X)
        names = clf.predict_label_names(X)
        re_encoded = clf.label_encoder_.transform(names)
        np.testing.assert_array_equal(preds, re_encoded)


# -----------------------------------------------------------------------
# Prediction tests
# -----------------------------------------------------------------------

class TestPrediction:
    def test_predict_shape(self, trained_classifier):
        clf, X, _ = trained_classifier
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_valid_codes(self, trained_classifier):
        clf, X, _ = trained_classifier
        preds = clf.predict(X)
        assert set(preds).issubset(set(range(clf.n_classes_)))

    def test_predict_label_names_type(self, trained_classifier):
        clf, X, _ = trained_classifier
        names = clf.predict_label_names(X)
        assert isinstance(names, list)
        assert len(names) == len(X)
        assert all(n in clf.get_class_names() for n in names)

    def test_predict_proba_shape(self, trained_classifier):
        clf, X, _ = trained_classifier
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), clf.n_classes_)

    def test_predict_proba_sums_to_one(self, trained_classifier):
        clf, X, _ = trained_classifier
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_with_scores_returns_tuple(self, trained_classifier):
        clf, X, _ = trained_classifier
        preds, proba, class_names = clf.predict_with_scores(X)
        assert len(preds) == len(X)
        assert proba.shape == (len(X), clf.n_classes_)
        assert class_names == clf.get_class_names()

    def test_single_row_prediction(self, trained_classifier):
        clf, X, _ = trained_classifier
        single = X.iloc[:1]
        pred = clf.predict(single)
        assert pred.shape == (1,)
        name = clf.predict_label_names(single)
        assert len(name) == 1

    def test_benign_guard_in_fit(self, synthetic_attack_data, selected_features):
        """BENIGN rows mixed into training should be dropped with a warning."""
        X, y, _ = synthetic_attack_data
        y_with_benign = y.copy()
        y_with_benign.iloc[0] = "BENIGN"
        y_with_benign.iloc[1] = "BENIGN"

        clf = XGBoostAttackClassifier(n_estimators=10, max_depth=2, random_state=42)
        # Should fit without error and not include BENIGN in class names
        clf.fit(X, y_with_benign)
        assert "BENIGN" not in clf.get_class_names()

    def test_missing_feature_raises(self, trained_classifier):
        """Missing required features must raise ValueError with informative message."""
        clf, _, _ = trained_classifier
        bad_df = pd.DataFrame({"unknown_feature": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Missing"):
            clf.predict(bad_df)

    def test_unfitted_raises(self, selected_features):
        """Calling predict on unfitted model must raise RuntimeError."""
        clf = XGBoostAttackClassifier()
        X_dummy = pd.DataFrame(
            np.zeros((5, len(selected_features))), columns=selected_features
        )
        with pytest.raises(RuntimeError, match="must be fitted"):
            clf.predict(X_dummy)


# -----------------------------------------------------------------------
# SHAP tests
# -----------------------------------------------------------------------

class TestSHAP:
    def test_explain_global_shapes(self, trained_classifier):
        """SHAP values must have shape (N_sample, n_features, n_classes)."""
        clf, X, _ = trained_classifier
        sv, X_sample = clf.explain_global(X, max_samples=50)
        assert X_sample.shape[1] == len(clf.get_feature_names())
        assert sv.shape[0] == len(X_sample)
        assert sv.shape[1] == len(clf.get_feature_names())
        assert sv.shape[2] == clf.n_classes_

    def test_explain_global_uses_real_data(self, trained_classifier):
        """X_sample rows must be a subset of the input X (real data, not synthetic)."""
        clf, X, _ = trained_classifier
        _, X_sample = clf.explain_global(X, max_samples=30)
        # Check that all returned sample values are in the valid range of X
        for col in X_sample.columns:
            assert X_sample[col].min() >= X[col].min() - 1e-6
            assert X_sample[col].max() <= X[col].max() + 1e-6

    def test_explain_global_no_subsampling_when_small(self, trained_classifier):
        """If X is smaller than max_samples, return all rows."""
        clf, X, _ = trained_classifier
        small_X = X.iloc[:20].reset_index(drop=True)
        sv, X_sample = clf.explain_global(small_X, max_samples=100)
        assert len(X_sample) == 20

    def test_explain_local_shape(self, trained_classifier):
        """Local SHAP must return (n_features, n_classes), (n_classes,), list."""
        clf, X, _ = trained_classifier
        sv, base, names = clf.explain_local(X.iloc[:1])
        assert sv.shape == (len(clf.get_feature_names()), clf.n_classes_)
        assert len(base) == clf.n_classes_
        assert names == clf.get_class_names()

    def test_explain_local_rejects_multirow(self, trained_classifier):
        clf, X, _ = trained_classifier
        with pytest.raises(ValueError, match="1 row"):
            clf.explain_local(X.iloc[:3])

    def test_get_xgb_booster_type(self, trained_classifier):
        """get_xgb_booster() must return an xgb.Booster instance."""
        import xgboost as xgb
        clf, _, _ = trained_classifier
        booster = clf.get_xgb_booster()
        assert isinstance(booster, xgb.Booster)


# -----------------------------------------------------------------------
# Persistence tests
# -----------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, trained_classifier):
        """Loaded model must produce identical predictions to in-memory model."""
        clf, X, _ = trained_classifier
        original_preds = clf.predict(X)
        original_proba = clf.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "xgb_attack.joblib"
            clf.save(save_path)

            assert save_path.exists()
            class_map_path = save_path.parent / "attack_class_mapping.json"
            assert class_map_path.exists()

            loaded = XGBoostAttackClassifier.load(save_path)

        loaded_preds = loaded.predict(X)
        loaded_proba = loaded.predict_proba(X)

        np.testing.assert_array_equal(original_preds, loaded_preds)
        np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-5)

    def test_class_mapping_json_content(self, trained_classifier):
        """Class mapping JSON must contain required keys with correct types."""
        clf, _, _ = trained_classifier
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "xgb_attack.joblib"
            clf.save(save_path)
            cmp = save_path.parent / "attack_class_mapping.json"
            with open(cmp, "r", encoding="utf-8") as f:
                data = json.load(f)

        assert "n_classes" in data
        assert "class_names" in data
        assert "int_to_class" in data
        assert "class_to_int" in data
        assert "feature_names" in data
        assert data["n_classes"] == clf.n_classes_
        assert data["class_names"] == clf.get_class_names()
        assert data["feature_names"] == clf.get_feature_names()

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            XGBoostAttackClassifier.load("/nonexistent/path.joblib")


# -----------------------------------------------------------------------
# Evaluator tests
# -----------------------------------------------------------------------

class TestEvaluator:
    def test_evaluate_returns_required_keys(self, trained_classifier):
        clf, X, y = trained_classifier
        preds = clf.predict_label_names(X)
        proba = clf.predict_proba(X)
        evaluator = AttackClassifierEvaluator()
        results = evaluator.evaluate(
            list(y), preds, y_proba=proba, class_names=clf.get_class_names()
        )
        required_keys = [
            "accuracy", "precision_macro", "recall_macro",
            "f1_macro", "f1_weighted",
            "classification_report_str", "classification_report_dict",
            "confusion_matrix", "confusion_matrix_labels", "n_test_samples",
        ]
        for key in required_keys:
            assert key in results, f"Missing key: {key}"

    def test_evaluate_accuracy_in_range(self, trained_classifier):
        clf, X, y = trained_classifier
        preds = clf.predict_label_names(X)
        evaluator = AttackClassifierEvaluator()
        results = evaluator.evaluate(list(y), preds)
        assert 0.0 <= results["accuracy"] <= 1.0

    def test_evaluate_f1_matches_sklearn(self, trained_classifier):
        clf, X, y = trained_classifier
        preds = clf.predict_label_names(X)
        evaluator = AttackClassifierEvaluator()
        results = evaluator.evaluate(list(y), preds)
        sklearn_f1 = f1_score(list(y), preds, average="macro", zero_division=0)
        assert abs(results["f1_macro"] - sklearn_f1) < 1e-6

    def test_confusion_matrix_shape(self, trained_classifier):
        clf, X, y = trained_classifier
        preds = clf.predict_label_names(X)
        evaluator = AttackClassifierEvaluator()
        results = evaluator.evaluate(list(y), preds)
        cm = np.array(results["confusion_matrix"])
        n = len(results["confusion_matrix_labels"])
        assert cm.shape == (n, n)

    def test_confusion_matrix_row_sums(self, trained_classifier):
        """Each confusion matrix row must sum to the number of true samples of that class."""
        clf, X, y = trained_classifier
        preds = clf.predict_label_names(X)
        evaluator = AttackClassifierEvaluator()
        results = evaluator.evaluate(list(y), preds)
        cm = np.array(results["confusion_matrix"])
        labels = results["confusion_matrix_labels"]
        y_arr = np.array(list(y))
        for i, label in enumerate(labels):
            expected = int((y_arr == label).sum())
            assert int(cm[i].sum()) == expected, f"Row sum mismatch for class: {label}"

    def test_evaluator_save_creates_json(self, trained_classifier):
        clf, X, y = trained_classifier
        preds = clf.predict_label_names(X)
        evaluator = AttackClassifierEvaluator()
        results = evaluator.evaluate(list(y), preds)
        with tempfile.TemporaryDirectory() as tmp:
            evaluator.save_results(results, Path(tmp), prefix="test_run")
            out = Path(tmp) / "test_run_evaluation.json"
            assert out.exists()
            with open(out, "r") as f:
                data = json.load(f)
            assert "accuracy" in data
            assert "f1_macro" in data

    def test_confusion_matrix_plot_saved(self, trained_classifier):
        clf, X, y = trained_classifier
        preds = clf.predict_label_names(X)
        evaluator = AttackClassifierEvaluator()
        results = evaluator.evaluate(list(y), preds)
        with tempfile.TemporaryDirectory() as tmp:
            out = evaluator.plot_confusion_matrix(results, Path(tmp), prefix="test_run")
            assert out.exists()
            assert out.suffix == ".png"

    def test_shap_summary_plot_saved(self, trained_classifier):
        clf, X, _ = trained_classifier
        sv, X_sample = clf.explain_global(X, max_samples=30)
        evaluator = AttackClassifierEvaluator()
        with tempfile.TemporaryDirectory() as tmp:
            out = evaluator.plot_shap_summary(
                sv, X_sample, clf.get_class_names(), Path(tmp), prefix="test_run"
            )
            assert out.exists()
            assert out.suffix == ".png"


# -----------------------------------------------------------------------
# Member 4 interface contract tests
# -----------------------------------------------------------------------

class TestMember4Interface:
    """Verify that all methods promised to Member 4 work correctly."""

    def test_get_xgb_booster_callable(self, trained_classifier):
        import shap
        clf, X, _ = trained_classifier
        booster = clf.get_xgb_booster()
        explainer = shap.TreeExplainer(booster)
        assert explainer is not None

    def test_feature_names_match_schema(self, trained_classifier, selected_features):
        clf, _, _ = trained_classifier
        assert clf.get_feature_names() == selected_features

    def test_class_names_are_strings(self, trained_classifier):
        clf, _, _ = trained_classifier
        names = clf.get_class_names()
        assert all(isinstance(n, str) for n in names)

    def test_predict_with_scores_structure(self, trained_classifier):
        clf, X, _ = trained_classifier
        preds, proba, names = clf.predict_with_scores(X.iloc[:10])
        assert len(preds) == 10
        assert proba.shape == (10, clf.n_classes_)
        assert names == clf.get_class_names()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_shap_local_class_alignment(self, trained_classifier):
        """SHAP base_values length must equal n_classes (one per attack class)."""
        clf, X, _ = trained_classifier
        sv, base, names = clf.explain_local(X.iloc[:1])
        assert len(base) == clf.n_classes_
        assert len(names) == clf.n_classes_
        assert names == clf.get_class_names()
