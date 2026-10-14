"""Page E — Artifacts and System Information."""

import sys
import platform
import streamlit as st
import pandas as pd
from pathlib import Path


def render(svc):
    st.title("⚙️ Artifacts & System Information")
    st.markdown("Current state of all loaded model artifacts, schema files, and environment dependencies.")

    # ── Artifact status table ─────────────────────────────────────────────────
    st.subheader("📦 Loaded Artifacts")
    artifact_summary = svc.get_artifact_summary()
    rows = []
    for name, status_text in artifact_summary.items():
        ok = "MISSING" not in status_text and "Error" not in status_text
        rows.append({"Artifact": name, "Status": status_text, "OK": "✅" if ok else "❌"})
    art_df = pd.DataFrame(rows)
    st.dataframe(art_df.set_index("OK"), use_container_width=True)

    # ── Errors and warnings ───────────────────────────────────────────────────
    if svc.status.errors:
        st.error("**Critical Errors** — prediction unavailable until resolved:")
        for e in svc.status.errors:
            st.code(e, language=None)

    if svc.status.warnings:
        with st.expander("⚠️ Warnings (non-critical)"):
            for w in svc.status.warnings:
                st.warning(w)

    # ── Artifact file details ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📁 Artifact File Details")

    ROOT = Path(__file__).resolve().parents[2]
    ARTIFACTS = ROOT / "artifacts"

    file_rows = []
    for f in sorted(ARTIFACTS.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            rel = str(f.relative_to(ROOT))
            file_rows.append({
                "File": rel,
                "Size": f"{size_kb:.1f} KB",
                "Exists": "✅",
            })
    if file_rows:
        st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No artifact files found.")

    # ── Feature schema details ────────────────────────────────────────────────
    if svc.selected_features:
        st.markdown("---")
        st.subheader(f"🎯 Selected Features ({len(svc.selected_features)} features)")
        feat_df = pd.DataFrame({
            "#": range(1, len(svc.selected_features) + 1),
            "Feature Name": svc.selected_features,
        }).set_index("#")
        st.dataframe(feat_df, use_container_width=True)

    # ── Class mapping ─────────────────────────────────────────────────────────
    if svc.status.class_mapping_ok:
        st.markdown("---")
        st.subheader(f"🎯 Attack Class Mapping ({svc.class_mapping.get('n_classes','?')} classes)")
        int_to_class = svc.class_mapping.get("int_to_class", {})
        cls_df = pd.DataFrame([
            {"Code": int(k), "Attack Class": v}
            for k, v in int_to_class.items()
        ]).sort_values("Code")
        st.dataframe(cls_df, use_container_width=True, hide_index=True)

    # ── Preprocessor details ──────────────────────────────────────────────────
    if svc.preprocessor is not None:
        st.markdown("---")
        st.subheader("🔧 Preprocessor Details")
        p = svc.preprocessor
        pc1, pc2 = st.columns(2)
        pc1.markdown(f"""
- **Scaler type:** {p.scaler_type}
- **Input features:** {len(p.feature_names_in_)}
- **Output features:** {len(p.feature_names_out_)}
- **Zero-variance cols removed:** {len(p.zero_variance_cols_)}
- **Is fitted:** {p.is_fitted_}
        """)
        pc2.markdown(f"""
- **Selected features configured:** {len(p.selected_features) if p.selected_features else 'None'}
- **Drop duplicate cols:** {p.drop_duplicate_cols}
- **Drop constant cols:** {p.drop_constant_cols}
- **Clip negative durations:** {p.clip_negative_durations}
        """)

        with st.expander("Imputation medians (training set)"):
            med_df = pd.DataFrame([
                {"Feature": k, "Training Median": round(v, 6)}
                for k, v in p.impute_medians_.items()
                if k in svc.selected_features
            ])
            st.dataframe(med_df, use_container_width=True, hide_index=True)

    # ── Environment info ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🐍 Environment")

    env_rows = []
    packages = [
        ("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
        ("Platform", platform.system()),
    ]
    try:
        import numpy; packages.append(("NumPy", numpy.__version__))
    except Exception: pass
    try:
        import pandas; packages.append(("Pandas", pandas.__version__))
    except Exception: pass
    try:
        import sklearn; packages.append(("scikit-learn", sklearn.__version__))
    except Exception: pass
    try:
        import xgboost; packages.append(("XGBoost", xgboost.__version__))
    except Exception: pass
    try:
        import shap; packages.append(("SHAP", shap.__version__))
    except Exception: pass
    try:
        import joblib; packages.append(("joblib", joblib.__version__))
    except Exception: pass
    try:
        import streamlit; packages.append(("Streamlit", streamlit.__version__))
    except Exception: pass

    env_df = pd.DataFrame(packages, columns=["Package", "Version"])
    st.dataframe(env_df, use_container_width=True, hide_index=True)

    # ── Setup and troubleshooting ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🚀 Setup & Commands")
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt
pip install streamlit plotly

# 2. Prepare dataset (requires MachineLearningCVE/ CSVs)
python scripts/prepare_dataset.py

# 3. Run feature selection
python scripts/run_feature_selection.py

# 4. Train binary models
python scripts/train_binary_models.py
python scripts/evaluate_binary_models.py

# 5. Train XGBoost attack classifier
python scripts/train_attack_classifier.py

# 6. Run all tests (93 expected to pass)
python -m pytest tests/ -v

# 7. Launch Streamlit application
streamlit run app/streamlit_app.py
    """, language="bash")

    st.subheader("📂 Expected Artifact Locations")
    st.code("""
artifacts/
├── models/
│   ├── preprocessor.joblib           ← Fitted preprocessing pipeline
│   ├── binary_random_forest.joblib   ← RF binary detector
│   ├── binary_decision_tree.joblib   ← DT baseline
│   ├── attack_xgboost_classifier.joblib ← XGBoost multiclass
│   └── attack_class_mapping.json     ← Attack category mapping
├── schemas/
│   ├── selected_features.json        ← 25 selected feature names
│   └── feature_schema.json           ← Full CIC-IDS2017 schema
└── reports/
    ├── binary_model_evaluation.json  ← RF & DT metrics
    └── attack_model_evaluation.json  ← XGBoost metrics
    """, language="text")

    st.subheader("📋 CSV Input Format")
    canonical = svc.get_canonical_columns()
    if canonical:
        st.markdown(
            f"Your CSV must contain these **{len(canonical)} columns** "
            "(matching canonical CIC-IDS2017 column names):"
        )
        with st.expander("View required columns"):
            st.dataframe(
                pd.DataFrame({"Column": canonical, "#": range(1, len(canonical)+1)}).set_index("#"),
                use_container_width=True
            )
    else:
        st.warning("Column schema unavailable (preprocessor not loaded).")
