"""Page 4 — Explain: Feature contribution explanations for the selected record."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ── SHAP waterfall helper ─────────────────────────────────────────────────────

def _shap_waterfall(shap_vals, feat_names, feat_vals, expected_val, title, n_top=12):
    sv = np.asarray(shap_vals).reshape(-1)
    feat_names = list(feat_names)
    feat_vals = np.asarray(feat_vals).reshape(-1).tolist()

    # Prevent feature-index errors if the explanation arrays differ in length
    n_features = min(len(sv), len(feat_names), len(feat_vals))
    if n_features == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No feature contribution data available to display.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color="#64748b"),
        )
        fig.update_layout(
            title=dict(text=title, font_size=14),
            height=320,
            template="plotly_white",
        )
        return fig

    sv = sv[:n_features]
    feat_names = feat_names[:n_features]
    feat_vals = feat_vals[:n_features]

    n_top = max(1, min(int(n_top), n_features))
    order = np.argsort(np.abs(sv))[::-1][:n_top]

    names = [feat_names[int(i)] for i in order]
    vals = [float(sv[int(i)]) for i in order]
    fvals = [float(feat_vals[int(i)]) for i in order]

    # Sort by contribution value for readability in horizontal bar chart
    srt = np.argsort(vals)
    names = [names[i] for i in srt]
    vals = [vals[i] for i in srt]
    fvals = [fvals[i] for i in srt]

    colors = ["#dc2626" if v > 0 else "#2563eb" for v in vals]
    ylabels = [f"{n}  <i style='color:#64748b'>({v:.3g})</i>" for n, v in zip(names, fvals)]

    base_num = float(expected_val) if hasattr(expected_val, "__float__") or isinstance(expected_val, (int, float)) else 0.0

    fig = go.Figure(go.Bar(
        x=vals, y=ylabels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font_size=14),
        xaxis_title="Feature Influence  (→ Red pushes toward Threat Detected · Blue pushes toward Safe)",
        height=max(320, n_top * 38),
        template="plotly_white",
        margin=dict(l=10, r=60, t=50, b=40),
        annotations=[dict(
            x=0.01, y=-0.12, xref="paper", yref="paper",
            text=f"Baseline Expected Value: {base_num:.4f}",
            showarrow=False, font=dict(size=10, color="#94a3b8"),
        )],
    )
    return fig


def render(svc, nav_to):
    st.markdown("## 🔍  Why Was This Result Given?")

    results  = st.session_state.get("results_df")
    raw_df   = st.session_state.get("raw_df")
    analysis = st.session_state.get("analysis_ready", False)

    if not analysis or results is None or raw_df is None:
        st.info(
            "No traffic analysis has been run yet. Please go to **Analyze** to check for threats first.",
            icon="📂",
        )
        if st.button("📂  Go to Analyze", type="primary", key="explain_goto_analyze"):
            nav_to("Analyze")
        return

    if not svc.status.can_predict_binary:
        st.error("Model artifacts are missing — explanation unavailable.")
        return

    st.caption(
        "Feature contributions show how individual network measurements influenced the detection decision. "
        "🔴 Red indicates factors that increase the likelihood of **Threat Detected**, "
        "while 🔵 Blue indicates factors that support **No Threat Detected**."
    )
    st.markdown("---")

    # ── Record selector ───────────────────────────────────────────────────────
    labels = []
    for _, row in results.iterrows():
        tag = "Threat Detected" if row["binary_label"] == "ATTACK" else "No Threat Detected"
        cat = row.get("attack_category") or ""
        lbl = f"Record {row['row_id']} — {tag}"
        if cat and cat != "XGBoost Error":
            lbl += f" ({cat})"
        labels.append(lbl)

    max_idx = len(labels) - 1
    cur_idx = min(st.session_state.get("selected_row_idx", 0), max_idx)

    chosen = st.selectbox("Select record to inspect:", labels, index=cur_idx, key="explain_sel")
    sel_idx = labels.index(chosen)
    st.session_state["selected_row_idx"] = sel_idx

    sel_result = results.iloc[sel_idx]
    binary_label = sel_result["binary_label"]
    attack_proba = float(sel_result["attack_proba"])
    row_id       = sel_result["row_id"]

    # ── Record summary cards ──────────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Record ID", str(row_id))
    display_pred = "Threat Detected" if binary_label == "ATTACK" else "No Threat Detected"
    d2.metric("Prediction", display_pred)
    d3.metric("Threat Probability", f"{attack_proba:.1%}")

    if binary_label == "ATTACK" and sel_result.get("attack_category"):
        cat = sel_result["attack_category"]
        if cat != "XGBoost Error":
            d4.metric("Possible Attack Type", cat)
            if sel_result.get("category_proba"):
                st.caption(f"Attack type confidence: {float(sel_result['category_proba']):.1%}")
        else:
            d4.warning("Classification unavailable")
    elif binary_label == "BENIGN":
        d4.success("Normal traffic — safe")

    # Feature values viewer
    sel_raw = raw_df.iloc[sel_idx: sel_idx + 1].copy()
    with st.expander("📋  View Raw & Normalized Feature Values"):
        try:
            X_trans = svc.preprocess(sel_raw)
            comp_df = pd.DataFrame({
                "Feature Name": list(X_trans.columns),
                "Normalized Value": [round(float(v), 4) for v in X_trans.iloc[0].values],
            })
            st.dataframe(comp_df, width='stretch', hide_index=True)
        except Exception as e:
            st.warning(f"Feature values display notice: {e}")

    # ── Stage 1: Threat Detection Factors (Random Forest) ─────────────────────
    st.markdown("---")
    st.subheader("Stage 1 · Threat Detection Factors")
    st.caption("How network measurements influenced the decision between **Threat Detected** and **No Threat Detected**.")

    with st.spinner("Calculating feature contributions…"):
        try:
            X_trans = svc.preprocess(sel_raw)
            rf_exp  = svc.explain_rf_local(X_trans)
        except Exception as e:
            rf_exp = None
            st.error(f"Explanation calculation notice: {e}")

    if rf_exp is not None:
        n_top_rf = st.slider("Number of top features to show", 5, 25, 10, key="shap_rf_n")
        fig_rf = _shap_waterfall(
            rf_exp["shap_values"], rf_exp["feature_names"],
            rf_exp["feature_values"], rf_exp["expected_value"],
            title=f"Feature Contributions for Record {row_id}",
            n_top=n_top_rf,
        )
        st.plotly_chart(fig_rf, width='stretch')

        sv_arr = np.asarray(rf_exp["shap_values"]).reshape(-1)
        f_names = list(rf_exp.get("feature_names", []))
        f_vals = np.asarray(rf_exp.get("feature_values", [])).reshape(-1)
        n_table = min(len(sv_arr), len(f_names), len(f_vals))
        if n_table > 0:
            sv_arr = sv_arr[:n_table]
            f_names = f_names[:n_table]
            f_vals = f_vals[:n_table]
            k_top = max(1, min(int(n_top_rf), n_table))
            order = np.argsort(np.abs(sv_arr))[::-1][:k_top]
            shap_df = pd.DataFrame({
                "Feature Name":      [f_names[int(i)] for i in order],
                "Observed Value":    [round(float(f_vals[int(i)]), 4) for i in order],
                "Contribution":      [round(float(sv_arr[int(i)]), 5) for i in order],
                "Influence":         ["Pushes toward Threat Detected" if sv_arr[int(i)] > 0 else "Pushes toward No Threat Detected" for i in order],
            })
            with st.expander("Detailed Feature Contribution Table"):
                st.dataframe(shap_df, width='stretch', hide_index=True)

        st.caption(
            f"Baseline Expected Value: {rf_exp['expected_value']:.4f} · "
            "Evaluated by Random Forest detector."
        )
    else:
        st.warning("Could not generate feature contributions for this record.")

    # ── Stage 2: Attack Type Classification Factors (XGBoost) ─────────────────
    if binary_label == "ATTACK" and svc.status.xgb_ok:
        st.markdown("---")
        st.subheader("Stage 2 · Possible Attack Type Factors")
        cat_name = sel_result.get("attack_category") or ""
        if cat_name and cat_name != "XGBoost Error":
            st.markdown(f"Explaining factors indicating **Possible Attack Type: {cat_name}**")

        with st.spinner("Calculating attack classification factors…"):
            try:
                X_trans_xgb = svc.preprocess(sel_raw)
                xgb_exp     = svc.explain_xgb_local(X_trans_xgb)
            except Exception as e:
                xgb_exp = None
                st.error(f"Attack classification explanation notice: {e}")

        if xgb_exp is not None:
            pred_class  = xgb_exp["predicted_class_name"]
            class_names = xgb_exp["class_names"]
            sv_matrix   = xgb_exp["shap_values"]
            fnames      = xgb_exp["feature_names"]
            fvals       = xgb_exp["feature_values"]
            base_vals   = xgb_exp["expected_value"]
            pred_idx    = xgb_exp["predicted_class_idx"]

            st.success(f"**Identified Attack Type:** {pred_class}")

            sv_pred = sv_matrix[:, pred_idx] if sv_matrix.ndim == 2 else sv_matrix
            base_ev = float(base_vals[pred_idx]) \
                if hasattr(base_vals, "__len__") else float(base_vals)

            n_top_xgb = st.slider("Number of top features to show (Attack Type)", 5, 25, 10, key="shap_xgb_n")
            fig_xgb = _shap_waterfall(
                sv_pred, fnames, fvals, base_ev,
                title=f"Attack Type Factors: {pred_class} (Record {row_id})",
                n_top=n_top_xgb,
            )
            st.plotly_chart(fig_xgb, width='stretch')

            # Cross-class heatmap
            if sv_matrix.ndim == 2 and len(class_names) > 1:
                with st.expander("Compare Features Across All Attack Types"):
                    k_heat = min(10, len(fnames), sv_matrix.shape[0])
                    top10 = np.argsort(np.abs(sv_matrix[:len(fnames), :]).mean(axis=1))[::-1][:k_heat]
                    heat  = sv_matrix[top10, :]
                    hfeat = [fnames[int(i)] for i in top10]
                    heat_fig = go.Figure(go.Heatmap(
                        z=heat.tolist(), x=class_names, y=hfeat,
                        colorscale="RdBu_r", zmid=0,
                        text=[[f"{v:.3f}" for v in row] for row in heat],
                        texttemplate="%{text}",
                    ))
                    heat_fig.update_layout(
                        title="Feature Influence Across Attack Types",
                        height=400, xaxis_tickangle=-35, template="plotly_white",
                    )
                    st.plotly_chart(heat_fig, width='stretch')
        else:
            st.info(
                "Attack type explanation could not be computed. The Threat Detected result above remains valid."
            )

    elif binary_label == "ATTACK" and not svc.status.xgb_ok:
        st.info(
            "Attack classification model is not loaded. The Threat Detected result above remains valid.",
            icon="ℹ️",
        )
    else:
        st.info(
            "This network record is **No Threat Detected (Safe)**. Attack classification is only performed when threats are detected.",
            icon="✅",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Results", key="explain_back"):
        nav_to("Results")
