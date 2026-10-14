"""Page A — Project Overview and Dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render(svc):
    st.title("🛡️ Explainable AI-Based Network Threat Detection")
    st.caption("CIC-IDS2017 Dataset  ·  Two-Stage Detection Architecture  ·  SHAP Explainability")

    st.warning(
        "⚠️ **Research Demonstration** — This system is an undergraduate ML engineering project. "
        "It is not a certified security product and should not be used as a sole indicator "
        "for production network security decisions.",
        icon="⚠️",
    )

    # ── Architecture diagram ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔀 Two-Stage Detection Architecture")

    col1, col2, col3 = st.columns([1, 0.15, 1])
    with col1:
        st.markdown("""
**Stage 1 — Binary Detection**
- **Model:** Random Forest Classifier
- **Baseline:** Decision Tree (comparison)
- **Task:** BENIGN vs ATTACK
- **Input:** 25 selected network flow features
- **Threshold:** 0.60 (calibrated on validation set)
- **Explainability:** SHAP TreeExplainer
        """)
    with col2:
        st.markdown("<div style='text-align:center; font-size:2rem; padding-top:60px'>→</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("""
**Stage 2 — Attack Classification**
- **Model:** XGBoost Multiclass Classifier
- **Task:** 13 attack categories
- **Input:** Same 25 features (ATTACK records only)
- **Imbalance handling:** Inverse-frequency sample weights
- **Explainability:** SHAP TreeExplainer (per class)
        """)

    st.info(
        "**Design principle:** BENIGN traffic is filtered at Stage 1 and never passed to Stage 2. "
        "Attack categories are only shown when Stage 1 predicts ATTACK.",
        icon="ℹ️",
    )

    # ── Model status cards ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📦 Artifact Status")

    c1, c2, c3, c4 = st.columns(4)
    def status_card(col, label, ok, ok_text="Loaded", fail_text="NOT LOADED"):
        with col:
            if ok:
                st.success(f"✅ **{label}**\n\n{ok_text}")
            else:
                st.error(f"❌ **{label}**\n\n{fail_text}")

    status_card(c1, "Preprocessor",  svc.status.preprocessor_ok, "RobustScaler + 25-feature schema")
    status_card(c2, "Random Forest", svc.status.rf_ok, f"Binary detector — threshold 0.60")
    status_card(c3, "XGBoost",       svc.status.xgb_ok,
                f"{svc.class_mapping.get('n_classes','?')} attack classes" if svc.status.class_mapping_ok else "Attack classifier")
    status_card(c4, "Decision Tree", svc.status.dt_ok, "Baseline model", "Missing (optional)")

    if svc.status.errors:
        st.error("**Critical errors — prediction is unavailable:**\n" + "\n".join(f"• {e}" for e in svc.status.errors))
    if svc.status.warnings:
        with st.expander("⚠️ Warnings", expanded=False):
            for w in svc.status.warnings:
                st.warning(w)

    # ── Key metrics from evaluation reports ───────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Evaluation Metrics (Held-Out Test Set)")

    if svc.status.binary_report_ok:
        rf = svc.binary_report.get("random_forest_detector", {})
        dt = svc.binary_report.get("decision_tree_baseline", {})
        splits = svc.binary_report.get("split_sizes", {})

        st.markdown("#### Stage 1 — Binary Detection (BENIGN vs ATTACK)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RF Accuracy",  f"{rf.get('accuracy', 0)*100:.2f}%", delta=f"+{(rf.get('accuracy',0)-dt.get('accuracy',0))*100:.2f}% vs DT")
        m2.metric("RF Attack F1", f"{rf.get('f1_attack', 0)*100:.2f}%")
        m3.metric("RF Precision", f"{rf.get('precision_attack', 0)*100:.2f}%")
        m4.metric("RF ROC-AUC",   f"{rf.get('roc_auc', 0):.4f}")

        if splits:
            st.caption(
                f"Test set: {splits.get('test', '?'):,} records  ·  "
                f"Train: {splits.get('train', '?'):,}  ·  "
                f"Val: {splits.get('val', '?'):,}  ·  "
                f"Split: 70/15/15 stratified"
            )

        # RF vs DT comparison bar chart
        if dt and rf:
            fig = go.Figure()
            metrics = ["accuracy", "f1_attack", "precision_attack", "recall_attack", "roc_auc"]
            labels = ["Accuracy", "F1 (Attack)", "Precision", "Recall", "ROC-AUC"]
            rf_vals = [rf.get(m, 0) for m in metrics]
            dt_vals = [dt.get(m, 0) for m in metrics]

            fig.add_bar(name="Random Forest", x=labels, y=rf_vals,
                        marker_color="#2563EB", text=[f"{v:.4f}" for v in rf_vals], textposition="outside")
            fig.add_bar(name="Decision Tree", x=labels, y=dt_vals,
                        marker_color="#94A3B8", text=[f"{v:.4f}" for v in dt_vals], textposition="outside")
            fig.update_layout(
                barmode="group", title="Random Forest vs Decision Tree (Binary Detection)",
                yaxis=dict(range=[0.97, 1.001], title="Score"),
                height=380, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Binary evaluation report not available.")

    if svc.status.attack_report_ok:
        st.markdown("#### Stage 2 — Attack Classification (XGBoost, 13 classes)")
        atk = svc.attack_report
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Accuracy",    f"{atk.get('accuracy', 0)*100:.2f}%")
        a2.metric("Macro F1",    f"{atk.get('f1_macro', 0)*100:.2f}%")
        a3.metric("Macro Prec.", f"{atk.get('precision_macro', 0)*100:.2f}%")
        a4.metric("ROC-AUC",     f"{atk.get('roc_auc_macro_ovr', 0):.4f}")

        # Per-class bar chart
        cr = atk.get("classification_report_dict", {})
        classes = [k for k in cr if k not in ("accuracy", "macro avg", "weighted avg")]
        if classes:
            f1_vals = [cr[c].get("f1-score", 0) for c in classes]
            support = [int(cr[c].get("support", 0)) for c in classes]
            colors = ["#16A34A" if v >= 0.9 else "#EAB308" if v >= 0.7 else "#DC2626" for v in f1_vals]
            fig2 = go.Figure()
            fig2.add_bar(
                x=classes, y=f1_vals, marker_color=colors,
                text=[f"{v:.2f}<br>(n={s})" for v, s in zip(f1_vals, support)],
                textposition="outside",
            )
            fig2.update_layout(
                title="XGBoost Per-Class F1-Score (Test Set)",
                yaxis=dict(range=[0, 1.15], title="F1-Score"),
                xaxis_tickangle=-35, height=420, template="plotly_white",
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Green ≥ 0.90 · Yellow ≥ 0.70 · Red < 0.70")
    else:
        st.info("Attack evaluation report not available.")

    # ── Dataset info ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📁 Dataset & Feature Pipeline")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Dataset:** CIC-IDS2017 (Canadian Institute for Cybersecurity)
- 8 CSV files (Monday–Friday, multiple captures per day)
- 15 attack types + BENIGN traffic
- ~2.8 million raw flow records (full dataset)
- **Sampled:** 30,000 rows/file (random, seed=42)
- **After deduplication:** 194,485 records
- **Split:** 70% train / 15% val / 15% test (stratified)
        """)
    with col_b:
        st.markdown(f"""
**Feature Pipeline:**
- Raw CIC-IDS2017 features: 79 columns
- After zero-variance removal: ~69 numeric
- After correlation pruning (|r| > 0.95): reduced set
- **Final selected features: {len(svc.selected_features)}** (RF Gini importance top-25)
- Scaling: RobustScaler (median/IQR, training data only)
- Imputation: Training median for NaN/Inf values
        """)

    if svc.selected_features:
        with st.expander(f"View all {len(svc.selected_features)} selected features"):
            feat_df = pd.DataFrame({"Feature": svc.selected_features, "Index": range(1, len(svc.selected_features)+1)})
            st.dataframe(feat_df.set_index("Index"), use_container_width=True)

    # ── Attack classes ────────────────────────────────────────────────────────
    if svc.status.class_mapping_ok:
        st.markdown("---")
        st.subheader("🎯 Detectable Attack Categories (Stage 2)")
        class_names = svc.class_mapping.get("class_names", [])
        cols = st.columns(4)
        for i, name in enumerate(class_names):
            cols[i % 4].markdown(f"• {name}")
