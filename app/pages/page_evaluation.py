"""Page D — Evaluation Results loaded from saved report files."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px


def render(svc):
    st.title("📈 Evaluation Results")
    st.markdown(
        "All metrics are loaded from saved evaluation report files generated during training. "
        "No metrics are fabricated or estimated."
    )

    tab_binary, tab_attack = st.tabs(["Stage 1 — Binary Detection", "Stage 2 — Attack Classification"])

    # ── STAGE 1 ───────────────────────────────────────────────────────────────
    with tab_binary:
        if not svc.status.binary_report_ok:
            st.error("Binary evaluation report not found. Run `python scripts/evaluate_binary_models.py` to generate it.")
            return

        report = svc.binary_report
        splits = report.get("split_sizes", {})
        rf = report.get("random_forest_detector", {})
        dt = report.get("decision_tree_baseline", {})

        st.subheader("Dataset Split Sizes")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Train", f"{splits.get('train', '?'):,}")
        sc2.metric("Validation", f"{splits.get('val', '?'):,}")
        sc3.metric("Test", f"{splits.get('test', '?'):,}")
        sc4.metric("Total", f"{sum(splits.values()):,}" if splits else "?")
        st.caption("Split ratio: 70% train / 15% validation / 15% test (stratified on Label)")

        # ── Side-by-side metrics ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Model Comparison")

        col_rf, col_dt = st.columns(2)

        def metric_block(col, title, data):
            with col:
                st.markdown(f"#### {title}")
                m1, m2 = st.columns(2)
                m1.metric("Accuracy",    f"{data.get('accuracy', 0)*100:.4f}%")
                m2.metric("Attack F1",   f"{data.get('f1_attack', 0)*100:.4f}%")
                m3, m4 = st.columns(2)
                m3.metric("Precision",   f"{data.get('precision_attack', 0)*100:.4f}%")
                m4.metric("Recall",      f"{data.get('recall_attack', 0)*100:.4f}%")
                m5, m6 = st.columns(2)
                m5.metric("ROC-AUC",     f"{data.get('roc_auc', 0):.6f}")
                m6.metric("PR-AUC",      f"{data.get('pr_auc', 0):.6f}")
                st.caption(f"Test samples: {data.get('test_samples', '?'):,} · "
                           f"Threshold: {data.get('threshold', '?')}")

        metric_block(col_rf, "🌲 Random Forest (Production)", rf)
        metric_block(col_dt, "🌳 Decision Tree (Baseline)", dt)

        # ── Confusion matrices ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Confusion Matrices")
        cm_col1, cm_col2 = st.columns(2)

        def plot_cm(col, data, title):
            with col:
                cm = data.get("confusion_matrix", {})
                if cm:
                    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
                    z = [[tn, fp], [fn, tp]]
                    text = [[f"TN={tn:,}", f"FP={fp:,}"], [f"FN={fn:,}", f"TP={tp:,}"]]
                    fig = ff.create_annotated_heatmap(
                        z, x=["Pred BENIGN", "Pred ATTACK"],
                        y=["True BENIGN", "True ATTACK"],
                        annotation_text=text,
                        colorscale=[[0, "#D1FAE5"], [0.5, "#FEF3C7"], [1, "#FEE2E2"]],
                        showscale=False,
                    )
                    fig.update_layout(title=title, height=300)
                    st.plotly_chart(fig, width='stretch')
                    st.caption(f"FP={fp} (BENIGN predicted as ATTACK), FN={fn} (ATTACK missed)")

        plot_cm(cm_col1, rf, "Random Forest Confusion Matrix")
        plot_cm(cm_col2, dt, "Decision Tree Confusion Matrix")

        # ── Classification report ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("Classification Reports")

        def class_report_df(data, model_name):
            cr = data.get("classification_report", {})
            rows = []
            for cls in ["BENIGN", "ATTACK", "macro avg", "weighted avg"]:
                if cls in cr:
                    d = cr[cls]
                    rows.append({
                        "Class": cls, "Model": model_name,
                        "Precision": round(d.get("precision", 0), 4),
                        "Recall": round(d.get("recall", 0), 4),
                        "F1-Score": round(d.get("f1-score", 0), 4),
                        "Support": int(d.get("support", 0)),
                    })
            return pd.DataFrame(rows)

        combined_cr = pd.concat([class_report_df(rf, "Random Forest"), class_report_df(dt, "Decision Tree")])
        st.dataframe(combined_cr.set_index(["Model", "Class"]), width='stretch')

        # ── Inference latency ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Inference Latency")
        lat_data = {
            "Model": ["Random Forest", "Decision Tree"],
            "ms/sample": [rf.get("inference_latency_ms_per_sample", 0), dt.get("inference_latency_ms_per_sample", 0)],
            "Total sec (test set)": [rf.get("total_inference_time_sec", 0), dt.get("total_inference_time_sec", 0)],
        }
        st.dataframe(pd.DataFrame(lat_data), width='stretch', hide_index=True)

    # ── STAGE 2 ───────────────────────────────────────────────────────────────
    with tab_attack:
        if not svc.status.attack_report_ok:
            st.error("Attack evaluation report not found. Run `python scripts/train_attack_classifier.py` to generate it.")
            return

        atk = svc.attack_report

        st.subheader("Overall XGBoost Multiclass Metrics")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Accuracy",        f"{atk.get('accuracy', 0)*100:.4f}%")
        a2.metric("Macro F1",        f"{atk.get('f1_macro', 0)*100:.4f}%")
        a3.metric("Weighted F1",     f"{atk.get('f1_weighted', 0)*100:.4f}%")
        a4.metric("ROC-AUC (macro)", f"{atk.get('roc_auc_macro_ovr', 0):.4f}")

        b1, b2 = st.columns(2)
        b1.metric("Macro Precision", f"{atk.get('precision_macro', 0)*100:.4f}%")
        b2.metric("Macro Recall",    f"{atk.get('recall_macro', 0)*100:.4f}%")
        st.caption(f"Test samples: {atk.get('n_test_samples', '?'):,} · Classes: {atk.get('n_classes_in_test', '?')}")
        st.caption(f"ROC note: {atk.get('roc_auc_note', 'One-vs-Rest macro')}")

        # ── Per-class classification report ─────────────────────────────────
        st.markdown("---")
        st.subheader("Per-Class Metrics")
        cr = atk.get("classification_report_dict", {})
        classes = [k for k in cr if k not in ("accuracy", "macro avg", "weighted avg")]

        if classes:
            cr_rows = []
            for cls in classes:
                d = cr[cls]
                support = int(d.get("support", 0))
                cr_rows.append({
                    "Attack Class": cls,
                    "Precision": round(d.get("precision", 0), 4),
                    "Recall": round(d.get("recall", 0), 4),
                    "F1-Score": round(d.get("f1-score", 0), 4),
                    "Test Support": support,
                    "Note": "⚠️ Very few test samples" if support < 10 else "",
                })
            cr_df = pd.DataFrame(cr_rows).sort_values("F1-Score", ascending=False)

            # Highlight weak classes
            def highlight_row(row):
                if row["Test Support"] < 10:
                    return ["background-color: #FEF3C7"] * len(row)
                elif row["F1-Score"] < 0.75:
                    return ["background-color: #FEE2E2"] * len(row)
                return [""] * len(row)

            st.dataframe(
                cr_df.style.apply(highlight_row, axis=1),
                width='stretch', hide_index=True,
            )
            st.caption("Yellow = < 10 test samples (unreliable). Red = F1 < 0.75")

            # Bar chart
            fig_pc = px.bar(
                cr_df.sort_values("F1-Score"),
                x="F1-Score", y="Attack Class", orientation="h",
                color="F1-Score", color_continuous_scale="RdYlGn",
                range_color=[0, 1],
                text="F1-Score", title="Per-Class F1-Score (XGBoost Test Set)",
                hover_data=["Precision", "Recall", "Test Support"],
            )
            fig_pc.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig_pc.update_layout(height=450, template="plotly_white", showlegend=False)
            st.plotly_chart(fig_pc, width='stretch')

        # ── Confusion matrix ─────────────────────────────────────────────────
        cm_labels = atk.get("confusion_matrix_labels", [])
        cm_data = atk.get("confusion_matrix", [])
        if cm_labels and cm_data:
            st.markdown("---")
            st.subheader("Confusion Matrix (XGBoost, 13 classes)")
            cm_arr = np.array(cm_data)
            # Normalise by row for readability
            row_sums = cm_arr.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm_arr / row_sums, 0)

            fig_cm = go.Figure(go.Heatmap(
                z=cm_norm.tolist(),
                x=cm_labels, y=cm_labels,
                colorscale="Blues",
                text=cm_arr.tolist(),
                texttemplate="%{text}",
                hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>",
            ))
            fig_cm.update_layout(
                title="Normalised Confusion Matrix (cell = count, shade = row %)",
                xaxis_title="Predicted", yaxis_title="True",
                xaxis_tickangle=-40,
                height=550, template="plotly_white",
            )
            st.plotly_chart(fig_cm, width='stretch')

        # ── Known limitations ─────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Known Limitations")
        st.markdown("""
| Class | Test Samples | Note |
|---|---|---|
| Heartbleed | 2 | F1=1.00 is statistically unreliable with 2 samples |
| Web Attack — Sql Injection | 4 | Minimal test support; treat metrics with caution |
| Web Attack — XSS | 204 | Confused with Brute Force (similar HTTP flow patterns) |
| Web Attack — Brute Force | 410 | Absorbs some XSS misclassifications |

**Root cause of Web Attack confusion:** All three Web Attack types share similar HTTP port/packet-size patterns.
Separating them reliably would require deep-packet inspection beyond CIC-IDS2017 flow statistics.
        """)
