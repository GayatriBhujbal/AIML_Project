"""Page 5 — System Info & Evaluation: all technical content in 5 tabs."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import sys
import platform
from pathlib import Path


def render(svc, nav_to):
    st.markdown("## ⚙️  System Information & Evaluation")
    st.caption(
        "Technical details, model performance, evaluation reports, "
        "artifacts, and known limitations."
    )

    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs([
        "🏗️  System Overview",
        "🤖  Models & Pipeline",
        "📈  Performance Evaluation",
        "📦  Artifacts & Setup",
        "⚠️  Limitations",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB A — System Overview
    # ══════════════════════════════════════════════════════════════════════════
    with tab_a:
        st.subheader("Project Purpose")
        st.markdown("""
This project implements an **explainable AI-based network intrusion detection system** using
the CIC-IDS2017 benchmark dataset.  The goal is to automatically classify network flows as
**BENIGN or ATTACK**, and for attack traffic to further identify **one of 13 specific attack
categories** — with a transparent explanation of which features drove each decision.

The system is designed as an undergraduate engineering mini-project demonstrating:
- Classical ML pipeline engineering (preprocessing, feature selection, modelling)
- Two-stage hierarchical classification
- Model explainability via SHAP (SHapley Additive exPlanations)
        """)

        st.markdown("---")
        st.subheader("Two-Stage Architecture")

        col1, arrow, col2 = st.columns([1, 0.12, 1])
        with col1:
            st.markdown("""
**Stage 1 — Binary Detection**
- Model: Random Forest Classifier (100 trees)
- Baseline: Decision Tree (comparison)
- Task: BENIGN vs ATTACK
- Input: 25 selected network flow features
- Threshold: 0.60 (calibrated on validation set)
- Explainability: SHAP TreeExplainer → P(ATTACK) contributions
            """)
        with arrow:
            st.markdown(
                "<div style='text-align:center;font-size:2rem;padding-top:60px'>→</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown("""
**Stage 2 — Attack Classification**
- Model: XGBoost Multiclass (300 estimators, max_depth=6)
- Task: 13 attack categories
- Input: Same 25 features (ATTACK records only)
- Imbalance: Inverse-frequency sample weights
- Explainability: SHAP TreeExplainer → per-class contributions
            """)

        st.info(
            "BENIGN traffic is filtered at Stage 1 and **never** passed to Stage 2. "
            "Attack categories are only displayed when Stage 1 predicts ATTACK.",
            icon="ℹ️",
        )

        st.markdown("---")
        st.subheader("Input → Output Flow")
        st.markdown("""
```
Raw CIC-IDS2017 CSV  (79 raw network flow features)
         ↓  DataCleaner  (NaN, Inf, duplicates removed)
         ↓  PreprocessingPipeline  (RobustScaler, imputation, feature selection)
         ↓  25 selected features
         ↓
Stage 1: RandomForestBinaryDetector.predict()
         ↓ 0 = BENIGN (stop)        ↓ 1 = ATTACK (continue)
         ↓                           ↓
  Result: BENIGN             Stage 2: XGBoostAttackClassifier.predict()
                                      ↓  one of 13 attack labels
                                      ↓
                               SHAP TreeExplainer (per-record, on request)
```
        """)

        st.markdown("---")
        st.subheader("Role of Explainability")
        st.markdown("""
SHAP (SHapley Additive exPlanations) decomposes each model prediction into per-feature
contributions.  For each flow record:

- **Random Forest SHAP** shows which of the 25 features pushed the prediction toward ATTACK
  and which pushed it toward BENIGN, relative to the model's baseline output.
- **XGBoost SHAP** shows, for the predicted attack class, which features were most influential.
  A cross-class heatmap is also available showing SHAP values for all 13 classes simultaneously.

SHAP values reflect **learned statistical patterns** in the training data.  They do not
establish causality.
        """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB B — Models & Pipeline
    # ══════════════════════════════════════════════════════════════════════════
    with tab_b:
        st.subheader("Random Forest Binary Detector")
        st.markdown("""
| Parameter | Value |
|---|---|
| Estimators | 100 trees |
| Max depth | 16 |
| Min samples split | 10 |
| Min samples leaf | 4 |
| Class weight | balanced |
| Decision threshold | 0.60 (calibrated on validation set) |
| n_jobs | -1 (all CPU cores) |
| random_state | 42 |
        """)

        st.subheader("Decision Tree Baseline")
        st.markdown("""
| Parameter | Value |
|---|---|
| Criterion | gini |
| Max depth | 12 |
| Min samples split | 10 |
| Min samples leaf | 5 |
| Class weight | balanced |
| random_state | 42 |

Used as an interpretability baseline to justify the additional complexity of the Random Forest.
        """)

        st.subheader("XGBoost Attack Classifier")
        st.markdown("""
| Parameter | Value |
|---|---|
| Estimators | 300 (with early stopping, round=20) |
| Max depth | 6 |
| Learning rate | 0.1 |
| Subsample | 0.8 |
| Colsample bytree | 0.8 |
| Objective | multi:softprob (13 classes) |
| Class imbalance | Inverse-frequency sample_weight per record |
| random_state | 42 |
        """)

        st.subheader("Preprocessing Pipeline")
        if svc.preprocessor is not None:
            p = svc.preprocessor
            col_a, col_b = st.columns(2)
            col_a.markdown(f"""
- Scaler: **{p.scaler_type}** (RobustScaler — median/IQR)
- Input features: **{len(p.feature_names_in_)}** raw columns
- Output features: **{len(p.feature_names_out_)}** after selection
- Zero-variance columns removed: **{len(p.zero_variance_cols_)}**
- Is fitted on training data: **{p.is_fitted_}**
            """)
            col_b.markdown("""
- Missing values: replaced by training-set column median
- Infinite values: replaced by column median (clipped)
- Negative durations: clipped to 0
- Label column: stripped before any numeric operation
- Re-fit on inference data: **NEVER**
            """)

            with st.expander(f"Imputation medians for {len(svc.selected_features)} selected features"):
                med_df = pd.DataFrame([
                    {"Feature": k, "Training Median": round(v, 6)}
                    for k, v in p.impute_medians_.items()
                    if k in svc.selected_features
                ])
                st.dataframe(med_df, width='stretch', hide_index=True)
        else:
            st.warning("Preprocessor not loaded.")

        st.subheader("Feature Selection")
        st.markdown("""
Feature selection was performed **on training data only** (no validation or test leakage):

1. Remove zero-variance columns from training set
2. Compute Pearson correlation matrix on training set — drop lower-importance partner in each
   highly corlinear pair (|r| > 0.95), retaining the higher Gini-importance feature
3. Fit `RandomForestClassifier` on training set; rank by Gini feature importance
4. Select the top **25** features from the de-collinear ranking
5. Validation set used **only** for F1 benchmark comparison — never to select features
        """)

        if svc.selected_features:
            feat_df = pd.DataFrame({
                "#": range(1, len(svc.selected_features)+1),
                "Feature": svc.selected_features,
            }).set_index("#")
            with st.expander(f"All {len(svc.selected_features)} selected features"):
                st.dataframe(feat_df, width='stretch')

        st.subheader("Attack Class Mapping")
        if svc.status.class_mapping_ok:
            int_to_class = svc.class_mapping.get("int_to_class", {})
            cls_df = pd.DataFrame([
                {"Code": int(k), "Attack Category": v}
                for k, v in int_to_class.items()
            ]).sort_values("Code")
            st.dataframe(cls_df, width='stretch', hide_index=True)
        else:
            st.warning("Class mapping not loaded.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB C — Performance Evaluation
    # ══════════════════════════════════════════════════════════════════════════
    with tab_c:
        if not svc.status.binary_report_ok and not svc.status.attack_report_ok:
            st.error(
                "No evaluation reports found. Run training scripts first. "
                "See **Artifacts & Setup** tab."
            )

        # ── Binary ────────────────────────────────────────────────────────────
        if svc.status.binary_report_ok:
            st.subheader("Stage 1 — Binary Detection (BENIGN vs ATTACK)")
            report = svc.binary_report
            splits = report.get("split_sizes", {})
            rf = report.get("random_forest_detector", {})
            dt = report.get("decision_tree_baseline", {})

            # Split sizes
            if splits:
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Train",      f"{splits.get('train', '?'):,}")
                s2.metric("Validation", f"{splits.get('val', '?'):,}")
                s3.metric("Test",       f"{splits.get('test', '?'):,}")
                s4.metric("Total",      f"{sum(splits.values()):,}")
                st.caption("Split: 70% train / 15% validation / 15% test (stratified)")

            # Metrics side-by-side
            col_rf, col_dt = st.columns(2)

            def _metric_block(col, title, data):
                with col:
                    st.markdown(f"##### {title}")
                    a, b = st.columns(2)
                    a.metric("Accuracy",  f"{data.get('accuracy',0)*100:.4f}%")
                    b.metric("Attack F1", f"{data.get('f1_attack',0)*100:.4f}%")
                    c, d = st.columns(2)
                    c.metric("Precision", f"{data.get('precision_attack',0)*100:.4f}%")
                    d.metric("Recall",    f"{data.get('recall_attack',0)*100:.4f}%")
                    e, f_ = st.columns(2)
                    e.metric("ROC-AUC",   f"{data.get('roc_auc',0):.6f}")
                    f_.metric("PR-AUC",   f"{data.get('pr_auc',0):.6f}")
                    st.caption(
                        f"Threshold: {data.get('threshold','?')} · "
                        f"Test n: {data.get('test_samples','?'):,}"
                    )

            _metric_block(col_rf, "🌲 Random Forest (Production)", rf)
            _metric_block(col_dt, "🌳 Decision Tree (Baseline)", dt)

            # Comparison bar chart
            if rf and dt:
                metrics  = ["accuracy", "f1_attack", "precision_attack", "recall_attack", "roc_auc"]
                mlabels  = ["Accuracy", "Attack F1", "Precision", "Recall", "ROC-AUC"]
                rf_vals  = [rf.get(m, 0) for m in metrics]
                dt_vals  = [dt.get(m, 0) for m in metrics]
                fig_cmp  = go.Figure()
                fig_cmp.add_bar(name="Random Forest", x=mlabels, y=rf_vals,
                                marker_color="#2563eb",
                                text=[f"{v:.4f}" for v in rf_vals], textposition="outside")
                fig_cmp.add_bar(name="Decision Tree", x=mlabels, y=dt_vals,
                                marker_color="#94a3b8",
                                text=[f"{v:.4f}" for v in dt_vals], textposition="outside")
                fig_cmp.update_layout(
                    barmode="group", title="RF vs Decision Tree — Binary Detection",
                    yaxis=dict(range=[0.97, 1.002], title="Score"),
                    height=360, template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_cmp, width='stretch')

            # Confusion matrices
            st.markdown("##### Confusion Matrices")
            cm_col1, cm_col2 = st.columns(2)

            def _plot_cm(col, data, title):
                with col:
                    cm = data.get("confusion_matrix", {})
                    if cm:
                        tn, fp = cm["tn"], cm["fp"]
                        fn, tp = cm["fn"], cm["tp"]
                        z    = [[tn, fp], [fn, tp]]
                        text = [[f"TN={tn:,}", f"FP={fp:,}"],
                                [f"FN={fn:,}", f"TP={tp:,}"]]
                        fig  = ff.create_annotated_heatmap(
                            z,
                            x=["Pred BENIGN", "Pred ATTACK"],
                            y=["True BENIGN", "True ATTACK"],
                            annotation_text=text,
                            colorscale=[[0,"#d1fae5"],[0.5,"#fef3c7"],[1,"#fee2e2"]],
                            showscale=False,
                        )
                        fig.update_layout(title=title, height=280)
                        st.plotly_chart(fig, width='stretch')

            _plot_cm(cm_col1, rf, "Random Forest")
            _plot_cm(cm_col2, dt, "Decision Tree")

            # Classification reports
            with st.expander("Classification Reports (full)"):
                def _cr_df(data, model):
                    cr   = data.get("classification_report", {})
                    rows = []
                    for cls in ["BENIGN", "ATTACK", "macro avg", "weighted avg"]:
                        if cls in cr:
                            d = cr[cls]
                            rows.append({
                                "Class": cls, "Model": model,
                                "Precision": round(d.get("precision",0),4),
                                "Recall":    round(d.get("recall",0),4),
                                "F1-Score":  round(d.get("f1-score",0),4),
                                "Support":   int(d.get("support",0)),
                            })
                    return pd.DataFrame(rows)
                combined = pd.concat([
                    _cr_df(rf, "Random Forest"),
                    _cr_df(dt, "Decision Tree"),
                ])
                st.dataframe(combined.set_index(["Model","Class"]), width='stretch')

        # ── Attack ────────────────────────────────────────────────────────────
        if svc.status.attack_report_ok:
            st.markdown("---")
            st.subheader("Stage 2 — Attack Classification (XGBoost, 13 classes)")
            atk = svc.attack_report

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Accuracy",       f"{atk.get('accuracy',0)*100:.4f}%")
            a2.metric("Macro F1",       f"{atk.get('f1_macro',0)*100:.4f}%")
            a3.metric("Weighted F1",    f"{atk.get('f1_weighted',0)*100:.4f}%")
            a4.metric("ROC-AUC (macro)",f"{atk.get('roc_auc_macro_ovr',0):.4f}")
            b1, b2 = st.columns(2)
            b1.metric("Macro Precision",f"{atk.get('precision_macro',0)*100:.4f}%")
            b2.metric("Macro Recall",   f"{atk.get('recall_macro',0)*100:.4f}%")
            st.caption(
                f"Test samples: {atk.get('n_test_samples','?'):,} · "
                f"Classes in test: {atk.get('n_classes_in_test','?')}"
            )

            # Per-class F1 chart
            cr = atk.get("classification_report_dict", {})
            classes = [k for k in cr if k not in ("accuracy","macro avg","weighted avg")]
            if classes:
                cr_rows = []
                for cls in classes:
                    d = cr[cls]
                    cr_rows.append({
                        "Attack Class": cls,
                        "Precision": round(d.get("precision",0),4),
                        "Recall":    round(d.get("recall",0),4),
                        "F1-Score":  round(d.get("f1-score",0),4),
                        "Test Support": int(d.get("support",0)),
                    })
                cr_df = pd.DataFrame(cr_rows).sort_values("F1-Score", ascending=False)

                fig_pc = px.bar(
                    cr_df.sort_values("F1-Score"),
                    x="F1-Score", y="Attack Class", orientation="h",
                    color="F1-Score", color_continuous_scale="RdYlGn", range_color=[0,1],
                    text="F1-Score",
                    title="Per-Class F1-Score (XGBoost Test Set)",
                    hover_data=["Precision","Recall","Test Support"],
                )
                fig_pc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_pc.update_layout(height=450, template="plotly_white",
                                     showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_pc, width='stretch')

                def _row_style(row):
                    if row["Test Support"] < 10:
                        return ["background-color:#fef3c7"] * len(row)
                    elif row["F1-Score"] < 0.75:
                        return ["background-color:#fee2e2"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    cr_df.style.apply(_row_style, axis=1),
                    width='stretch', hide_index=True,
                )
                st.caption("Yellow = <10 test samples (statistically unreliable). Red = F1 < 0.75")

            # Confusion matrix
            cm_labels = atk.get("confusion_matrix_labels", [])
            cm_data   = atk.get("confusion_matrix", [])
            if cm_labels and cm_data:
                cm_arr  = np.array(cm_data)
                row_sum = cm_arr.sum(axis=1, keepdims=True)
                cm_norm = np.where(row_sum > 0, cm_arr / row_sum, 0)
                fig_cm  = go.Figure(go.Heatmap(
                    z=cm_norm.tolist(), x=cm_labels, y=cm_labels,
                    colorscale="Blues",
                    text=cm_arr.tolist(), texttemplate="%{text}",
                    hovertemplate="True:%{y}<br>Pred:%{x}<br>Count:%{text}<extra></extra>",
                ))
                fig_cm.update_layout(
                    title="XGBoost Confusion Matrix (normalised by row, counts shown)",
                    xaxis_title="Predicted", yaxis_title="True",
                    xaxis_tickangle=-40, height=560, template="plotly_white",
                )
                st.plotly_chart(fig_cm, width='stretch')

    # ══════════════════════════════════════════════════════════════════════════
    # TAB D — Artifacts & Setup
    # ══════════════════════════════════════════════════════════════════════════
    with tab_d:
        st.subheader("Artifact Status")
        summary = svc.get_artifact_summary()
        rows = [
            {"Artifact": name,
             "Status": status,
             "OK": "✅" if "MISSING" not in status and "Error" not in status else "❌"}
            for name, status in summary.items()
        ]
        st.dataframe(pd.DataFrame(rows).set_index("OK"), width='stretch')

        if svc.status.errors:
            st.error("**Critical errors:**\n" + "\n".join(f"• {e}" for e in svc.status.errors))
        if svc.status.warnings:
            with st.expander("⚠️ Warnings"):
                for w in svc.status.warnings:
                    st.warning(w)

        # Artifact files
        ROOT = Path(__file__).resolve().parents[2]
        ARTS = ROOT / "artifacts"
        file_rows = []
        for f in sorted(ARTS.rglob("*")):
            if f.is_file():
                file_rows.append({
                    "File": str(f.relative_to(ROOT)),
                    "Size": f"{f.stat().st_size/1024:.1f} KB",
                })
        if file_rows:
            with st.expander("Artifact files on disk"):
                st.dataframe(pd.DataFrame(file_rows), width='stretch', hide_index=True)

        st.markdown("---")
        st.subheader("Environment")
        pkgs = [("Python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
                ("Platform", platform.system())]
        for name, mod in [("numpy","numpy"),("pandas","pandas"),
                           ("scikit-learn","sklearn"),("xgboost","xgboost"),
                           ("shap","shap"),("joblib","joblib"),("streamlit","streamlit")]:
            try:
                import importlib; m = importlib.import_module(mod)
                pkgs.append((name, getattr(m,"__version__","?")))
            except Exception: pass
        st.dataframe(pd.DataFrame(pkgs, columns=["Package","Version"]),
                     width='stretch', hide_index=True)

        st.markdown("---")
        st.subheader("Setup & Launch Commands")
        st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset  (requires MachineLearningCVE/ directory)
python scripts/prepare_dataset.py

# 3. Feature selection
python scripts/run_feature_selection.py

# 4. Train binary models
python scripts/train_binary_models.py
python scripts/evaluate_binary_models.py

# 5. Train XGBoost attack classifier
python scripts/train_attack_classifier.py

# 6. Run test suite  (138 tests expected)
python -m pytest tests/ -v

# 7. Launch Streamlit app
streamlit run app/streamlit_app.py
        """, language="bash")

        st.subheader("Expected Artifact Locations")
        st.code("""
artifacts/
├── models/
│   ├── preprocessor.joblib               ← required
│   ├── binary_random_forest.joblib       ← required
│   ├── binary_decision_tree.joblib       ← optional (baseline)
│   ├── attack_xgboost_classifier.joblib  ← required for Stage 2
│   └── attack_class_mapping.json
├── schemas/
│   ├── selected_features.json            ← required
│   └── feature_schema.json
└── reports/
    ├── binary_model_evaluation.json
    └── attack_model_evaluation.json
        """, language="text")

        st.subheader("Required CSV Format")
        canonical = svc.get_canonical_columns()
        if canonical:
            with st.expander(f"All {len(canonical)} required columns"):
                st.dataframe(
                    pd.DataFrame({"#": range(1,len(canonical)+1),
                                  "Column Name": canonical}).set_index("#"),
                    width='stretch',
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB E — Limitations
    # ══════════════════════════════════════════════════════════════════════════
    with tab_e:
        st.subheader("Known Limitations")

        st.markdown("#### Rare-Class Reliability")
        st.markdown("""
| Class | Test Samples | Issue |
|---|---|---|
| Heartbleed | 2 | F1=1.00 is statistically meaningless — only 2 samples |
| Web Attack — SQL Injection | 4 | Metrics should be treated with extreme caution |
| Infiltration | 0–1 | May not appear in test set at all |
        """)
        st.warning(
            "Do not cite Heartbleed or SQL Injection F1 scores as strong results. "
            "Report the test support count alongside the metric.",
        )

        st.markdown("#### Web Attack Class Confusion")
        st.markdown("""
The three Web Attack subtypes (Brute Force, XSS, SQL Injection) share similar HTTP flow
profiles (port 80/443, similar packet sizes, overlapping timing statistics).  CIC-IDS2017
flow statistics do not include payload content, so separating these classes without
deep-packet inspection is fundamentally difficult.

- XSS F1 ≈ 0.72 — some XSS records are classified as Brute Force
- SQL Injection F1 ≈ 0.67 — very small sample size compounds the issue
        """)

        st.markdown("#### Dataset & Training Limitations")
        st.markdown("""
- **Sampling:** 30,000 rows per file (random, seed=42) from 8 CIC-IDS2017 CSV files.
  Monday's file is 100% BENIGN — class balance reflects the dataset, not a bug.
- **After deduplication:** 194,485 records remain (45,515 duplicates removed pre-split).
  CIC-IDS2017 is known to contain large numbers of exact duplicate flow records.
- **Feature scope:** 25 features selected from 79 raw CIC-IDS2017 statistics.
  These are derived network flow statistics, not raw packet payloads.
- **Concept drift:** The model was trained on 2017 traffic patterns.  Modern attack tools
  may produce different flow signatures not represented in the training data.
- **No live capture:** This system analyses uploaded CSV files only.
  It does not capture live network traffic or integrate with packet-capture tools.
        """)

        st.markdown("#### SHAP Limitations")
        st.markdown("""
- SHAP values reflect the model's learned statistical associations — not causal relationships.
- A high SHAP value for `Destination Port` means the model found port number predictive in
  the training data, not that the port *caused* the attack.
- SHAP computation uses `shap.TreeExplainer` on actual saved model artifacts.
  No SHAP values are fabricated or hardcoded.
- For the XGBoost multiclass model, SHAP values are computed per class.
  The displayed waterfall shows values for the predicted class only.
        """)

        st.markdown("#### Model Confidence Interpretation")
        st.markdown("""
- `P(ATTACK)` is the Random Forest's predicted probability for the ATTACK class.
  It is **not** a calibrated probability; the model was not Platt-scaled or isotonic-calibrated.
- `Category Prob.` is XGBoost's max class probability from `predict_proba()`.
  Same caveat applies.
- Neither score should be interpreted as a guaranteed accuracy percentage for that record.
        """)
