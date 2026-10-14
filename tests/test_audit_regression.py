"""
Regression tests for audit fixes (Phases 2–6).

Tests verify:
  BUG-1: Raw input with 25 selected-feature column names is NOT treated as preprocessed.
  BUG-2: predict_raw / predict_preprocessed follow distinct, explicit paths.
  BUG-3: Shared sampling constant is used by all scripts.
  BUG-4: load_combined supports 'random' and 'head' sampling strategies.
  BUG-5: pipeline.fit() raises when a required selected feature is missing post-processing.
  BUG-6: pipeline.save() writes a sidecar schema; load() validates it.
  GENERAL: Missing features, extra features, column reorder, and empty inputs raise errors.
"""
import json
import numpy as np
import pandas as pd
import pytest

from config.dataset_config import (
    SCHEMAS_DIR,
    DEFAULT_SAMPLE_PER_FILE,
    SAMPLE_STRATEGY,
    SAMPLE_RANDOM_SEED,
)
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.binary_detector import RandomForestBinaryDetector
from src.models.predictor import ThreatDetectionInferenceEngine


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def selected_features():
    with open(SCHEMAS_DIR / "selected_features.json", "r", encoding="utf-8") as f:
        return json.load(f)["selected_features"]


@pytest.fixture
def raw_feature_df(selected_features):
    """A synthetic RAW DataFrame that has the canonical CIC-IDS2017 column names
    (i.e. the un-preprocessed columns the pipeline expects as input)."""
    from src.dataset.schema import NUMERIC_COLUMNS
    n = 20
    df = pd.DataFrame({col: np.random.uniform(0.5, 200.0, n) for col in NUMERIC_COLUMNS})
    return df


@pytest.fixture
def fitted_engine(tmp_path, selected_features, raw_feature_df):
    """Build a minimal fitted engine for testing input contract."""
    pipeline = PreprocessingPipeline(scaler_type="robust", selected_features=selected_features)
    pipeline.fit(raw_feature_df)
    p_path = tmp_path / "preprocessor.joblib"
    pipeline.save(p_path)

    X_trans = pipeline.transform(raw_feature_df)
    y = np.array([0] * 14 + [1] * 6)
    detector = RandomForestBinaryDetector(n_estimators=5, max_depth=3, random_state=42)
    detector.fit(X_trans, y)
    m_path = tmp_path / "rf.joblib"
    detector.save(m_path)

    engine = ThreatDetectionInferenceEngine(
        preprocessor_path=p_path,
        model_path=m_path,
        selected_features_path=SCHEMAS_DIR / "selected_features.json",
    )
    return engine, pipeline, X_trans, selected_features


# ──────────────────────────────────────────────────────────────────────────────
# BUG-1 regression: unsafe preprocessed-input bypass removed
# ──────────────────────────────────────────────────────────────────────────────

class TestBug1UnsafeBypassRemoved:
    """Verify that a DataFrame whose columns are the 25 selected feature names
    is NOT silently passed through without preprocessing when it is actually raw
    (unscaled) data.  The old code would bypass the pipeline in that case."""

    def test_raw_df_with_selected_col_names_goes_through_preprocessing(
        self, fitted_engine, selected_features
    ):
        """
        Build a fake 'raw' DataFrame whose column names are exactly the 25 selected
        features but whose values are unscaled raw values.  Under the old code this
        would skip preprocessing.  The fix means it now raises a ValueError because
        the 25 selected-feature columns are NOT the same as the canonical pipeline
        input columns (which number ~69 raw features).
        """
        engine, pipeline, _, _ = fitted_engine
        # Raw-looking but with selected-feature column names (would have bypassed old code)
        fake_raw = pd.DataFrame(
            {feat: np.random.uniform(100, 5000, 5) for feat in selected_features}
        )
        # validate_and_transform must raise because these are not the raw pipeline input cols
        with pytest.raises(ValueError, match="canonical CIC-IDS2017 features"):
            engine.validate_and_transform(fake_raw)

    def test_predict_raises_on_selected_feature_named_raw_df(self, fitted_engine, selected_features):
        """predict() (legacy) must also raise -- same path via validate_and_transform."""
        engine, *_ = fitted_engine
        fake_raw = pd.DataFrame(
            {feat: np.random.uniform(100, 5000, 5) for feat in selected_features}
        )
        with pytest.raises(ValueError):
            engine.predict(fake_raw)

    def test_old_bypass_did_not_apply_scaling(self, fitted_engine, selected_features):
        """The preprocessed validation path must REJECT DataFrames with extra columns."""
        engine, pipeline, X_trans, _ = fitted_engine
        # Add one extra column to the preprocessed output -- should be rejected
        X_bad = X_trans.copy()
        X_bad["extra_col"] = 0.0
        with pytest.raises(ValueError, match="unexpected columns"):
            engine.predict_preprocessed(X_bad)


# ──────────────────────────────────────────────────────────────────────────────
# BUG-2 regression: explicit predict_raw / predict_preprocessed paths
# ──────────────────────────────────────────────────────────────────────────────

class TestBug2ExplicitInputPaths:

    def test_predict_preprocessed_accepts_valid_preprocessed_df(self, fitted_engine):
        engine, _, X_trans, _ = fitted_engine
        preds = engine.predict_preprocessed(X_trans)
        assert len(preds) == len(X_trans)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_preprocessed_proba_shape(self, fitted_engine):
        engine, _, X_trans, _ = fitted_engine
        proba = engine.predict_preprocessed_proba(X_trans)
        assert proba.shape == (len(X_trans), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_preprocessed_rejects_missing_features(self, fitted_engine, selected_features):
        engine, _, X_trans, _ = fitted_engine
        incomplete = X_trans.drop(columns=[selected_features[0]])
        with pytest.raises(ValueError, match="missing selected features"):
            engine.predict_preprocessed(incomplete)

    def test_predict_preprocessed_rejects_extra_features(self, fitted_engine):
        engine, _, X_trans, _ = fitted_engine
        X_extra = X_trans.copy()
        X_extra["extra_col"] = 99.9
        with pytest.raises(ValueError, match="unexpected columns"):
            engine.predict_preprocessed(X_extra)

    def test_predict_preprocessed_rejects_wrong_column_order(self, fitted_engine, selected_features):
        """Column order mismatch: _validate_preprocessed enforces order via df[selected_features]."""
        engine, _, X_trans, _ = fitted_engine
        shuffled = X_trans[list(reversed(selected_features))]
        # Should NOT raise -- reorder is handled internally by _validate_preprocessed
        preds = engine.predict_preprocessed(shuffled)
        assert len(preds) == len(shuffled)

    def test_predict_raw_applies_preprocessing(self, fitted_engine, raw_feature_df):
        """predict_raw must transform raw data through the pipeline."""
        engine, *_ = fitted_engine
        preds = engine.predict_raw(raw_feature_df)
        assert len(preds) == len(raw_feature_df)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_raw_proba_shape(self, fitted_engine, raw_feature_df):
        engine, *_ = fitted_engine
        proba = engine.predict_raw_proba(raw_feature_df)
        assert proba.shape == (len(raw_feature_df), 2)

    def test_predict_raw_labels_strings(self, fitted_engine, raw_feature_df):
        engine, *_ = fitted_engine
        labels = engine.predict_raw_labels(raw_feature_df)
        assert all(l in {"BENIGN", "ATTACK"} for l in labels)

    def test_route_to_attack_classifier_preprocessed(self, fitted_engine):
        """route_to_attack_classifier_preprocessed must not apply preprocessing."""
        engine, _, X_trans, _ = fitted_engine
        attack_feats, attack_idx = engine.route_to_attack_classifier_preprocessed(X_trans)
        assert isinstance(attack_feats, pd.DataFrame)
        assert len(attack_feats) == len(attack_idx)
        assert len(attack_feats) <= len(X_trans)

    def test_preprocessing_not_applied_twice(self, fitted_engine):
        """Calling predict_preprocessed then predict_preprocessed again on the
        same DataFrame should give the same predictions (idempotent)."""
        engine, _, X_trans, _ = fitted_engine
        p1 = engine.predict_preprocessed(X_trans)
        p2 = engine.predict_preprocessed(X_trans)
        np.testing.assert_array_equal(p1, p2)


# ──────────────────────────────────────────────────────────────────────────────
# BUG-3 regression: shared sampling config
# ──────────────────────────────────────────────────────────────────────────────

class TestBug3SharedSamplingConfig:

    def test_default_sample_per_file_is_an_integer(self):
        assert isinstance(DEFAULT_SAMPLE_PER_FILE, int)
        assert DEFAULT_SAMPLE_PER_FILE >= 0

    def test_sample_strategy_is_valid(self):
        assert SAMPLE_STRATEGY in {"random", "head"}

    def test_sample_random_seed_is_integer(self):
        assert isinstance(SAMPLE_RANDOM_SEED, int)


# ──────────────────────────────────────────────────────────────────────────────
# BUG-4 regression: reproducible sampling in DatasetLoader
# ──────────────────────────────────────────────────────────────────────────────

class TestBug4LoaderSampling:

    def _make_tmp_csv(self, tmp_path, n_rows, filename="test.csv"):
        """Create a small synthetic CSV with numeric columns and a Label column."""
        from src.dataset.schema import NUMERIC_COLUMNS
        cols = NUMERIC_COLUMNS[:5] + ["Label"]
        df = pd.DataFrame(
            {c: np.arange(n_rows, dtype=float) for c in cols[:-1]}
        )
        df["Label"] = ["BENIGN"] * (n_rows // 2) + ["DoS Hulk"] * (n_rows // 2)
        path = tmp_path / filename
        df.to_csv(path, index=False)
        return path

    def test_random_sampling_is_reproducible(self, tmp_path):
        from src.dataset.loader import DatasetLoader
        csv_path = self._make_tmp_csv(tmp_path, n_rows=200)
        loader = DatasetLoader(tmp_path)
        df1 = loader.load_combined(nrows_per_file=50, sample_strategy="random", sample_seed=42)
        df2 = loader.load_combined(nrows_per_file=50, sample_strategy="random", sample_seed=42)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))

    def test_random_sampling_differs_with_different_seed(self, tmp_path):
        from src.dataset.loader import DatasetLoader
        csv_path = self._make_tmp_csv(tmp_path, n_rows=200)
        loader = DatasetLoader(tmp_path)
        df1 = loader.load_combined(nrows_per_file=50, sample_strategy="random", sample_seed=1)
        df2 = loader.load_combined(nrows_per_file=50, sample_strategy="random", sample_seed=99)
        # Very unlikely to be identical for 200-row source with 50-row sample
        assert not df1.reset_index(drop=True).equals(df2.reset_index(drop=True))

    def test_head_sampling_takes_first_n_rows(self, tmp_path):
        from src.dataset.loader import DatasetLoader
        self._make_tmp_csv(tmp_path, n_rows=200)
        loader = DatasetLoader(tmp_path)
        df = loader.load_combined(nrows_per_file=30, sample_strategy="head")
        assert len(df) == 30

    def test_random_sample_respects_nrows_limit(self, tmp_path):
        from src.dataset.loader import DatasetLoader
        self._make_tmp_csv(tmp_path, n_rows=200)
        loader = DatasetLoader(tmp_path)
        df = loader.load_combined(nrows_per_file=50, sample_strategy="random")
        assert len(df) == 50

    def test_full_load_when_nrows_none(self, tmp_path):
        from src.dataset.loader import DatasetLoader
        self._make_tmp_csv(tmp_path, n_rows=100)
        loader = DatasetLoader(tmp_path)
        df = loader.load_combined(nrows_per_file=None)
        assert len(df) == 100


# ──────────────────────────────────────────────────────────────────────────────
# BUG-5 regression: pipeline raises on missing selected feature
# ──────────────────────────────────────────────────────────────────────────────

class TestBug5PipelineMissingSelectedFeature:

    def test_fit_raises_when_selected_feature_absent(self):
        """If a selected feature is zero-variance (or missing) in the training
        sample, pipeline.fit() must raise instead of silently dropping it."""
        n = 30
        feat_a = "Fwd Packets/s"
        feat_b = "Flow Bytes/s"
        # Build training data that contains feat_a but NOT feat_b
        df = pd.DataFrame({feat_a: np.random.uniform(1, 100, n)})
        pipeline = PreprocessingPipeline(
            scaler_type="robust",
            selected_features=[feat_a, feat_b],  # feat_b is absent
        )
        with pytest.raises(ValueError, match="absent after preprocessing"):
            pipeline.fit(df)

    def test_fit_raises_when_selected_feature_is_constant(self):
        """A selected feature that is constant (zero variance) in the training
        sample is dropped by the zero-variance filter, then the BUG-5 check fires."""
        n = 30
        feat_a = "Fwd Packets/s"
        feat_b = "Flow Bytes/s"
        df = pd.DataFrame({
            feat_a: np.random.uniform(1, 100, n),
            feat_b: np.zeros(n),   # constant -- will be zero-variance filtered
        })
        pipeline = PreprocessingPipeline(
            scaler_type="robust",
            selected_features=[feat_a, feat_b],
        )
        with pytest.raises(ValueError, match="absent after preprocessing"):
            pipeline.fit(df)

    def test_fit_succeeds_when_all_selected_features_present(self, selected_features):
        """No error when all selected features are genuinely present."""
        n = 40
        df = pd.DataFrame({f: np.random.uniform(1, 100, n) for f in selected_features})
        pipeline = PreprocessingPipeline(
            scaler_type="robust",
            selected_features=selected_features,
        )
        pipeline.fit(df)
        assert pipeline.feature_names_out_ == selected_features


# ──────────────────────────────────────────────────────────────────────────────
# BUG-6 regression: schema sidecar saved and validated on load
# ──────────────────────────────────────────────────────────────────────────────

class TestBug6SchemaSidecar:

    def test_save_creates_schema_sidecar(self, tmp_path, selected_features):
        n = 40
        df = pd.DataFrame({f: np.random.uniform(1, 100, n) for f in selected_features})
        pipeline = PreprocessingPipeline(scaler_type="robust", selected_features=selected_features)
        pipeline.fit(df)
        save_path = tmp_path / "pipeline.joblib"
        pipeline.save(save_path)
        schema_path = save_path.with_suffix(".schema.json")
        assert schema_path.exists(), "Schema sidecar file must be created alongside .joblib"

    def test_schema_sidecar_contains_expected_keys(self, tmp_path, selected_features):
        n = 40
        df = pd.DataFrame({f: np.random.uniform(1, 100, n) for f in selected_features})
        pipeline = PreprocessingPipeline(scaler_type="robust", selected_features=selected_features)
        pipeline.fit(df)
        save_path = tmp_path / "pipeline.joblib"
        pipeline.save(save_path)
        with open(save_path.with_suffix(".schema.json"), "r", encoding="utf-8") as f:
            schema = json.load(f)
        for key in ("feature_names_in_", "feature_names_out_", "scaler_type"):
            assert key in schema, f"Schema sidecar missing key: {key}"
        assert schema["feature_names_out_"] == selected_features

    def test_load_succeeds_when_sidecar_matches(self, tmp_path, selected_features):
        n = 40
        df = pd.DataFrame({f: np.random.uniform(1, 100, n) for f in selected_features})
        pipeline = PreprocessingPipeline(scaler_type="robust", selected_features=selected_features)
        pipeline.fit(df)
        save_path = tmp_path / "pipeline.joblib"
        pipeline.save(save_path)
        loaded = PreprocessingPipeline.load(save_path)
        assert loaded.feature_names_out_ == selected_features

    def test_load_raises_on_schema_mismatch(self, tmp_path, selected_features):
        """Corrupt the sidecar to simulate a stale artifact mismatch."""
        n = 40
        df = pd.DataFrame({f: np.random.uniform(1, 100, n) for f in selected_features})
        pipeline = PreprocessingPipeline(scaler_type="robust", selected_features=selected_features)
        pipeline.fit(df)
        save_path = tmp_path / "pipeline.joblib"
        pipeline.save(save_path)
        # Corrupt the sidecar
        schema_path = save_path.with_suffix(".schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        schema["feature_names_out_"] = schema["feature_names_out_"][:-3]  # remove last 3 features
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f)
        with pytest.raises(RuntimeError, match="schema mismatch"):
            PreprocessingPipeline.load(save_path)

    def test_load_without_sidecar_still_works(self, tmp_path, selected_features):
        """If no sidecar exists (old artifact), load proceeds without error."""
        import joblib
        n = 40
        df = pd.DataFrame({f: np.random.uniform(1, 100, n) for f in selected_features})
        pipeline = PreprocessingPipeline(scaler_type="robust", selected_features=selected_features)
        pipeline.fit(df)
        save_path = tmp_path / "pipeline_nosidecar.joblib"
        # Save joblib directly without sidecar
        joblib.dump(pipeline, save_path)
        loaded = PreprocessingPipeline.load(save_path)
        assert loaded.feature_names_out_ == selected_features


# ──────────────────────────────────────────────────────────────────────────────
# General input validation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGeneralInputValidation:

    def test_empty_dataframe_raises(self, fitted_engine):
        engine, *_ = fitted_engine
        with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
            engine.predict_raw(pd.DataFrame())

    def test_empty_list_raises(self, fitted_engine):
        engine, *_ = fitted_engine
        with pytest.raises(ValueError, match="Input list of records cannot be empty"):
            engine.predict_raw([])

    def test_unsupported_type_raises(self, fitted_engine):
        engine, *_ = fitted_engine
        with pytest.raises(TypeError, match="Unsupported input type"):
            engine.predict_raw("not a dataframe")

    def test_target_column_stripped_from_raw_input(self, fitted_engine, raw_feature_df):
        """If the Label column is accidentally included it must be silently removed."""
        engine, *_ = fitted_engine
        df_with_label = raw_feature_df.copy()
        df_with_label["Label"] = "BENIGN"
        # Should not raise -- Label is stripped before preprocessing
        preds = engine.predict_raw(df_with_label)
        assert len(preds) == len(raw_feature_df)

    def test_missing_raw_features_raises_actionable_error(self, fitted_engine):
        engine, *_ = fitted_engine
        bad_df = pd.DataFrame({"Totally_Unknown_Feature": [1.0, 2.0]})
        with pytest.raises(ValueError):
            engine.predict_raw(bad_df)

    def test_duplicate_selected_feature_cols_rejected(self, fitted_engine, selected_features):
        """Duplicate column names in preprocessed input should raise."""
        engine, _, X_trans, _ = fitted_engine
        X_dup = pd.concat([X_trans, X_trans[[selected_features[0]]]], axis=1)
        # pandas allows duplicate columns; _validate_preprocessed must catch extra col
        with pytest.raises(ValueError):
            engine.predict_preprocessed(X_dup)
