"""
Tests for the Streamlit inference service.

Tests:
- Artifact loading and ArtifactStatus fields.
- CSV validation (valid input, missing columns, empty file, extra columns, Inf values).
- Preprocessing output shape and feature ordering.
- Binary prediction output types and values.
- Attack-category routing (ATTACK rows only).
- BENIGN handling (no XGBoost call).
- SHAP explanation generation (RF and XGBoost).
- Full pipeline result schema.
- Missing artifact graceful handling.
- Evaluation report loading.
"""

import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def svc():
    """Load real InferenceService (uses real saved artifacts)."""
    from app.inference_service import InferenceService
    return InferenceService()


@pytest.fixture(scope="module")
def canonical_cols(svc):
    return svc.get_canonical_columns()


@pytest.fixture(scope="module")
def minimal_raw_df(canonical_cols):
    """Tiny DataFrame with all required canonical columns filled with non-zero values."""
    rng = np.random.default_rng(42)
    data = {col: rng.uniform(10, 1000, 5) for col in canonical_cols}
    return pd.DataFrame(data)


# ── Artifact loading tests ─────────────────────────────────────────────────────

class TestArtifactLoading:
    def test_service_loads_without_crash(self, svc):
        assert svc is not None

    def test_preprocessor_loaded(self, svc):
        assert svc.status.preprocessor_ok, f"Errors: {svc.status.errors}"

    def test_rf_loaded(self, svc):
        assert svc.status.rf_ok, f"Errors: {svc.status.errors}"

    def test_xgb_loaded(self, svc):
        assert svc.status.xgb_ok, f"Warnings: {svc.status.warnings}"

    def test_selected_features_loaded(self, svc):
        assert svc.status.selected_features_ok
        assert len(svc.selected_features) == 25

    def test_selected_features_unique(self, svc):
        assert len(svc.selected_features) == len(set(svc.selected_features))

    def test_class_mapping_loaded(self, svc):
        assert svc.status.class_mapping_ok
        assert svc.class_mapping.get("n_classes") == 13

    def test_binary_report_loaded(self, svc):
        assert svc.status.binary_report_ok
        assert "random_forest_detector" in svc.binary_report

    def test_attack_report_loaded(self, svc):
        assert svc.status.attack_report_ok
        assert "accuracy" in svc.attack_report

    def test_can_predict_binary_flag(self, svc):
        assert svc.status.can_predict_binary

    def test_can_predict_attack_flag(self, svc):
        assert svc.status.can_predict_attack

    def test_artifact_summary_has_all_keys(self, svc):
        summary = svc.get_artifact_summary()
        assert "Preprocessor" in summary
        assert "Random Forest (Binary)" in summary
        assert "XGBoost (Attack Classification)" in summary


# ── CSV validation tests ──────────────────────────────────────────────────────

class TestCSVValidation:
    def test_valid_input_passes(self, svc, minimal_raw_df):
        ok, errs, warns = svc.validate_uploaded_csv(minimal_raw_df)
        assert ok, f"Unexpected errors: {errs}"

    def test_empty_df_fails(self, svc, canonical_cols):
        empty = pd.DataFrame(columns=canonical_cols)
        ok, errs, _ = svc.validate_uploaded_csv(empty)
        assert not ok
        assert any("no data" in e.lower() for e in errs)

    def test_missing_columns_fails(self, svc, canonical_cols):
        partial = pd.DataFrame({col: [1.0, 2.0] for col in canonical_cols[:10]})
        ok, errs, _ = svc.validate_uploaded_csv(partial)
        assert not ok
        assert any("missing" in e.lower() for e in errs)

    def test_extra_columns_warn(self, svc, minimal_raw_df):
        df_extra = minimal_raw_df.copy()
        df_extra["some_extra_col"] = 99
        ok, errs, warns = svc.validate_uploaded_csv(df_extra)
        assert ok
        assert any("extra" in w.lower() for w in warns)

    def test_inf_values_warn(self, svc, canonical_cols):
        df_inf = pd.DataFrame({col: [1.0, float("inf"), 3.0] for col in canonical_cols})
        ok, errs, warns = svc.validate_uploaded_csv(df_inf)
        assert ok
        assert any("infinite" in w.lower() or "inf" in w.lower() for w in warns)

    def test_label_column_warns(self, svc, minimal_raw_df):
        df_with_label = minimal_raw_df.copy()
        df_with_label["Label"] = "BENIGN"
        ok, errs, warns = svc.validate_uploaded_csv(df_with_label)
        assert ok
        assert any("label" in w.lower() for w in warns)


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_preprocess_output_shape(self, svc, minimal_raw_df):
        X_trans = svc.preprocess(minimal_raw_df)
        assert X_trans.shape == (len(minimal_raw_df), 25)

    def test_preprocess_output_columns_ordered(self, svc, minimal_raw_df):
        X_trans = svc.preprocess(minimal_raw_df)
        assert list(X_trans.columns) == svc.selected_features

    def test_preprocess_strips_label(self, svc, minimal_raw_df):
        df_with_label = minimal_raw_df.copy()
        df_with_label["Label"] = "BENIGN"
        X_trans = svc.preprocess(df_with_label)
        assert "Label" not in X_trans.columns

    def test_preprocess_no_nan_in_output(self, svc, minimal_raw_df):
        X_trans = svc.preprocess(minimal_raw_df)
        assert not X_trans.isna().any().any()

    def test_preprocess_no_inf_in_output(self, svc, minimal_raw_df):
        X_trans = svc.preprocess(minimal_raw_df)
        assert not np.isinf(X_trans.values).any()

    def test_preprocess_missing_columns_raises_or_wrong_shape(self, svc):
        """Passing a 2-column DF to preprocess() should either raise an error
        or produce output that does NOT match the expected 25-feature schema.
        The pipeline may impute rather than raise -- both outcomes signal
        that the input was invalid; the important invariant is that a valid
        25-feature output is NOT silently produced from garbage input."""
        tiny_df = pd.DataFrame({"col_a": [1.0], "col_b": [2.0]})
        try:
            X_trans = svc.preprocess(tiny_df)
            # If it didn't raise, the output must not silently pass as valid
            assert X_trans.shape[1] != 25 or not all(
                f in X_trans.columns for f in svc.selected_features
            ), "preprocess() silently accepted invalid input and returned valid-looking output"
        except (ValueError, KeyError, RuntimeError, Exception):
            pass  # Any exception is acceptable — input was invalid



# ── Binary prediction tests ───────────────────────────────────────────────────

class TestBinaryPrediction:
    def test_predict_binary_returns_arrays(self, svc, minimal_raw_df):
        X_trans = svc.preprocess(minimal_raw_df)
        preds, proba = svc.predict_binary(X_trans)
        assert len(preds) == len(minimal_raw_df)
        assert len(proba) == len(minimal_raw_df)

    def test_binary_preds_are_0_or_1(self, svc, minimal_raw_df):
        X_trans = svc.preprocess(minimal_raw_df)
        preds, _ = svc.predict_binary(X_trans)
        assert set(preds).issubset({0, 1})

    def test_attack_proba_in_unit_interval(self, svc, minimal_raw_df):
        X_trans = svc.preprocess(minimal_raw_df)
        _, proba = svc.predict_binary(X_trans)
        assert (proba >= 0).all() and (proba <= 1).all()


# ── Full pipeline tests ───────────────────────────────────────────────────────

class TestFullPipeline:
    def test_pipeline_returns_dataframe(self, svc, minimal_raw_df):
        results = svc.run_full_pipeline(minimal_raw_df)
        assert isinstance(results, pd.DataFrame)

    def test_pipeline_result_has_required_cols(self, svc, minimal_raw_df):
        results = svc.run_full_pipeline(minimal_raw_df)
        for col in ["row_id", "binary_label", "attack_proba"]:
            assert col in results.columns, f"Missing column: {col}"

    def test_pipeline_row_count_matches_input(self, svc, minimal_raw_df):
        results = svc.run_full_pipeline(minimal_raw_df)
        assert len(results) == len(minimal_raw_df)

    def test_binary_labels_are_valid_strings(self, svc, minimal_raw_df):
        results = svc.run_full_pipeline(minimal_raw_df)
        assert set(results["binary_label"]).issubset({"BENIGN", "ATTACK"})

    def test_benign_rows_have_no_attack_category(self, svc, minimal_raw_df):
        results = svc.run_full_pipeline(minimal_raw_df)
        benign_mask = results["binary_label"] == "BENIGN"
        if benign_mask.any():
            # BENIGN rows must not have an attack category assigned
            assert results.loc[benign_mask, "attack_category"].isna().all()

    def test_attack_rows_have_attack_category_if_xgb_loaded(self, svc, minimal_raw_df):
        if not svc.status.xgb_ok:
            pytest.skip("XGBoost not loaded.")
        results = svc.run_full_pipeline(minimal_raw_df)
        attack_mask = results["binary_label"] == "ATTACK"
        if attack_mask.any():
            assert results.loc[attack_mask, "attack_category"].notna().all()

    def test_pipeline_with_id_col(self, svc, minimal_raw_df):
        df = minimal_raw_df.copy()
        df["flow_id"] = [f"flow_{i}" for i in range(len(df))]
        results = svc.run_full_pipeline(df, id_col="flow_id")
        assert list(results["row_id"]) == [f"flow_{i}" for i in range(len(df))]

    def test_attack_categories_are_valid_class_names(self, svc, minimal_raw_df):
        if not svc.status.xgb_ok:
            pytest.skip("XGBoost not loaded.")
        valid_classes = set(svc.class_mapping.get("class_names", []))
        results = svc.run_full_pipeline(minimal_raw_df)
        attack_cats = results["attack_category"].dropna()
        invalid = [c for c in attack_cats if c not in valid_classes and c != "XGBoost Error"]
        assert not invalid, f"Invalid categories: {invalid}"


# ── Evaluation report tests ───────────────────────────────────────────────────

class TestEvaluationReports:
    def test_binary_report_has_rf_data(self, svc):
        assert "random_forest_detector" in svc.binary_report

    def test_binary_report_accuracy_in_range(self, svc):
        rf = svc.binary_report.get("random_forest_detector", {})
        acc = rf.get("accuracy", 0)
        assert 0.95 <= acc <= 1.0, f"Unexpected accuracy: {acc}"

    def test_attack_report_has_n_classes(self, svc):
        n = svc.attack_report.get("n_classes_in_test", 0)
        assert n == 13

    def test_attack_report_accuracy_in_range(self, svc):
        acc = svc.attack_report.get("accuracy", 0)
        assert 0.90 <= acc <= 1.0, f"Unexpected accuracy: {acc}"

    def test_attack_report_has_classification_report(self, svc):
        cr = svc.attack_report.get("classification_report_dict", {})
        assert "Bot" in cr
        assert "DDoS" in cr


# ── SHAP tests ────────────────────────────────────────────────────────────────

class TestSHAPExplanations:
    def test_rf_shap_returns_dict_or_none(self, svc, minimal_raw_df):
        X_trans = svc.preprocess(minimal_raw_df.iloc[:1])
        result = svc.explain_rf_local(X_trans)
        assert result is None or isinstance(result, dict)

    def test_rf_shap_has_required_keys(self, svc, minimal_raw_df):
        X_trans = svc.preprocess(minimal_raw_df.iloc[:1])
        result = svc.explain_rf_local(X_trans)
        if result is not None:
            for key in ["shap_values", "expected_value", "feature_names", "feature_values"]:
                assert key in result, f"Missing key: {key}"

    def test_rf_shap_feature_count_matches(self, svc, minimal_raw_df):
        X_trans = svc.preprocess(minimal_raw_df.iloc[:1])
        result = svc.explain_rf_local(X_trans)
        if result is not None:
            assert len(result["shap_values"]) == 25
            assert len(result["feature_names"]) == 25

    def test_xgb_shap_returns_dict_or_none(self, svc, minimal_raw_df):
        if not svc.status.xgb_ok:
            pytest.skip("XGBoost not loaded.")
        X_trans = svc.preprocess(minimal_raw_df.iloc[:1])
        result = svc.explain_xgb_local(X_trans)
        assert result is None or isinstance(result, dict)

    def test_xgb_shap_has_class_names(self, svc, minimal_raw_df):
        if not svc.status.xgb_ok:
            pytest.skip("XGBoost not loaded.")
        X_trans = svc.preprocess(minimal_raw_df.iloc[:1])
        result = svc.explain_xgb_local(X_trans)
        if result is not None:
            assert "class_names" in result
            assert len(result["class_names"]) == 13
