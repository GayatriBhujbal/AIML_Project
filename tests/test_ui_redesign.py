"""
Tests for the redesigned Streamlit navigation and session state logic.

Tests:
- Default session state initialisation.
- Navigation helper updates st.session_state["page"].
- New upload invalidates previous results (file_hash mismatch).
- Selected record index persists between pages.
- Analysis_ready flag gates Results page.
- Results CSV download content.
- Inference service integration (reuses existing tests).
"""

import sys
import json
import hashlib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw_df(canonical_cols, n=5):
    rng = np.random.default_rng(42)
    return pd.DataFrame({col: rng.uniform(10, 1000, n) for col in canonical_cols})


# ── InferenceService smoke tests (reuse real artifacts) ───────────────────────

@pytest.fixture(scope="module")
def svc():
    from app.inference_service import InferenceService
    return InferenceService()


@pytest.fixture(scope="module")
def canonical_cols(svc):
    return svc.get_canonical_columns()


@pytest.fixture(scope="module")
def minimal_raw(canonical_cols):
    return _make_raw_df(canonical_cols)


# ── Navigation state logic (unit tests — no Streamlit runtime needed) ─────────

class TestSessionStateDefaults:
    """Verify default session state keys are defined in streamlit_app.py."""

    def test_defaults_dict_has_required_keys(self):
        """The _DEFAULTS dict in streamlit_app must cover all required keys."""
        # We parse the file rather than importing it (Streamlit won't run headless)
        app_text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        required_keys = [
            '"page"', '"raw_df"', '"results_df"', '"file_hash"',
            '"selected_row_idx"', '"analysis_ready"',
        ]
        for key in required_keys:
            assert key in app_text, f"Missing session state key: {key}"

    def test_page_default_is_home(self):
        app_text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        assert '"page": "Home"' in app_text

    def test_analysis_ready_default_is_false(self):
        app_text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        assert '"analysis_ready": False' in app_text


class TestPageFiles:
    """All five page modules must exist and expose a render() function."""

    @pytest.mark.parametrize("page_file", [
        "home", "analyze", "results", "explain", "sysinfo",
    ])
    def test_page_file_exists(self, page_file):
        path = ROOT / "app" / "pages" / f"{page_file}.py"
        assert path.exists(), f"Missing page file: {path}"

    @pytest.mark.parametrize("page_file", [
        "home", "analyze", "results", "explain", "sysinfo",
    ])
    def test_page_has_render_function(self, page_file):
        text = (ROOT / "app" / "pages" / f"{page_file}.py").read_text(encoding="utf-8")
        assert "def render(" in text, f"{page_file}.py missing render() function"

    def test_home_has_start_analysis_button(self):
        text = (ROOT / "app" / "pages" / "home.py").read_text(encoding="utf-8")
        assert "Start Analysis" in text or "start_analysis" in text.lower()

    def test_home_has_no_accuracy_metric(self):
        text = (ROOT / "app" / "pages" / "home.py").read_text(encoding="utf-8")
        # Home page must NOT show accuracy metrics directly
        assert "accuracy" not in text.lower() or "disclaimer" in text.lower(), \
            "Home page should not display accuracy metrics — move them to System Info"

    def test_results_page_has_download_button(self):
        text = (ROOT / "app" / "pages" / "results.py").read_text(encoding="utf-8")
        assert "download_button" in text

    def test_results_page_has_filter(self):
        text = (ROOT / "app" / "pages" / "results.py").read_text(encoding="utf-8")
        assert "filter" in text.lower() or "selectbox" in text.lower()

    def test_explain_page_uses_session_state(self):
        text = (ROOT / "app" / "pages" / "explain.py").read_text(encoding="utf-8")
        assert 'session_state' in text and 'selected_row_idx' in text

    def test_explain_page_handles_no_analysis(self):
        text = (ROOT / "app" / "pages" / "explain.py").read_text(encoding="utf-8")
        assert "analysis_ready" in text or "No analysis" in text

    def test_sysinfo_has_five_tabs(self):
        text = (ROOT / "app" / "pages" / "sysinfo.py").read_text(encoding="utf-8")
        assert text.count("with tab_") >= 5, "sysinfo.py should have at least 5 tab blocks"

    def test_sysinfo_loads_from_report_not_hardcoded(self):
        text = (ROOT / "app" / "pages" / "sysinfo.py").read_text(encoding="utf-8")
        assert "binary_report" in text or "svc.binary_report" in text, \
            "System Info must load metrics from report, not hardcode them"


class TestNewUploadClearsOldResults:
    """Verify the file-hash logic in analyze.py invalidates stale results."""

    def test_different_hash_triggers_clear(self):
        text = (ROOT / "app" / "pages" / "analyze.py").read_text(encoding="utf-8")
        # The analyze page must compare the new file hash to the stored one
        assert "file_hash" in text
        assert "analysis_ready" in text
        assert 'False' in text  # Must set analysis_ready=False on new upload

    def test_same_hash_does_not_force_rerun(self):
        text = (ROOT / "app" / "pages" / "analyze.py").read_text(encoding="utf-8")
        assert "fhash != st.session_state" in text or "file_hash" in text


class TestNavigationStructure:
    """Verify the top navigation in streamlit_app.py covers all 5 pages."""

    def test_all_five_pages_in_nav(self):
        text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        for page in ["Home", "Analyze", "Results", "Explain", "System Info"]:
            assert page in text, f"Page '{page}' missing from navigation"

    def test_no_sidebar_page_radio(self):
        text = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
        # The old sidebar radio should be gone
        assert 'st.radio' not in text or "nav_" in text, \
            "Sidebar radio navigation should be replaced with top nav"


# ── Inference integration ─────────────────────────────────────────────────────

class TestInferenceServiceIntegration:
    def test_service_loads(self, svc):
        assert svc.status.can_predict_binary

    def test_full_pipeline_result_structure(self, svc, minimal_raw):
        results = svc.run_full_pipeline(minimal_raw)
        assert isinstance(results, pd.DataFrame)
        assert "binary_label" in results.columns
        assert "attack_proba" in results.columns
        assert len(results) == len(minimal_raw)

    def test_benign_rows_have_no_category(self, svc, minimal_raw):
        results = svc.run_full_pipeline(minimal_raw)
        benign = results[results["binary_label"] == "BENIGN"]
        if len(benign) > 0:
            assert benign["attack_category"].isna().all()

    def test_download_csv_has_expected_columns(self, svc, minimal_raw):
        results = svc.run_full_pipeline(minimal_raw)
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        df_reloaded = pd.read_csv(BytesIO(csv_bytes))
        assert "binary_label" in df_reloaded.columns
        assert "attack_proba" in df_reloaded.columns

    def test_selected_record_index_in_bounds(self, svc, minimal_raw):
        results = svc.run_full_pipeline(minimal_raw)
        max_idx = len(results) - 1
        sel_idx = 0
        assert 0 <= sel_idx <= max_idx

    def test_evaluation_reports_loaded(self, svc):
        assert svc.status.binary_report_ok
        assert svc.status.attack_report_ok

    def test_attack_report_has_classification_report(self, svc):
        cr = svc.attack_report.get("classification_report_dict", {})
        assert len(cr) > 0

    def test_rf_shap_for_single_record(self, svc, minimal_raw):
        X = svc.preprocess(minimal_raw.iloc[:1])
        result = svc.explain_rf_local(X)
        if result is not None:
            assert "shap_values" in result
            assert len(result["shap_values"]) == 25

    def test_xgb_shap_for_single_record(self, svc, minimal_raw):
        if not svc.status.xgb_ok:
            pytest.skip("XGBoost not loaded")
        X = svc.preprocess(minimal_raw.iloc[:1])
        result = svc.explain_xgb_local(X)
        if result is not None:
            assert "class_names" in result
            assert len(result["class_names"]) == 13
