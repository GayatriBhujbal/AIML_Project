"""Page 4 — Explain: SHAP explanations for the record selected on the Results page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ── SHAP waterfall helper ─────────────────────────────────────────────────────

def _shap_waterfall(shap_vals, feat_names, feat_vals, expected_val, title, n_top=12):
    sv = np.array(shap_vals).flatten()          # always 1-D regardless of SHAP output shape
    feat_vals = list(np.array(feat_vals).flatten())
    n_top = min(n_top, len(sv))
    order = np.argsort(np.abs(sv))[::-1][:n_top]
    names = [feat_names[int(i)] for i in order]
    vals  = [float(sv[int(i)])  for i in order]
    fvals = [float(feat_vals[int(i)]) for i in order]

    # Sort by magnitude for readability
    srt = np.argsort(vals)
    names = [names[i] for i in srt]
    vals  = [vals[i]  for i in srt]
    fvals = [fvals[i] for i in srt]

    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in vals]
    ylabels = [f"{n}  <i style='color:#64748b'>({v:.3g})</i>"
               for n, v in zip(names, fvals)]

    fig = go.Figure(go.Bar(
        x=vals, y=ylabels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=title, font_size=14),
        xaxis_title="SHAP value  →  pushes toward ATTACK",
        height=max(320, n_top * 38),
        template="plotly_white",
        margin=dict(l=10, r=60, t=50, b=40),
        annotations=[dict(
            x=0.01, y=-0.12, xref="paper", yref="paper",
            text=f"Base value: {expected_val:.4f}",
            showarrow=False, font=dict(size=10, color="#94a3b8"),
        )],
    )
    return fig


def render(svc, nav_to):
    st.markdown("## 🔍  Explain a Prediction")

    results  = st.session_state.get("results_df")
    raw_df   = st.session_state.get("raw_df")
    analysis = st.session_state.get("analysis_ready", False)

    if not analysis or results is None or raw_df is None:
        st.info(
            "No analysis available. Go to **Analyze** first to upload a CSV and run detection.",
            icon="📂",
        )
        if st.button("📂  Go to Analyze", type="primary", key="explain_goto_analyze"):
            nav_to("Analyze")
        return

    if not svc.status.can_predict_binary:
        st.error("Models not loaded — explanation unavailable.")
        return

    st.caption(
        "SHAP values show how much each feature contributes to this prediction. "
        "🔴 Red pushes toward ATTACK · 🔵 Blue pushes toward BENIGN."
    )
    st.markdown("---")

    # ── Record selector ───────────────────────────────────────────────────────
    labels = []
    for _, row in results.iterrows():
        tag = row["binary_label"]
        cat = row.get("attack_category") or ""
        lbl = f"Row {row['row_id']} — {tag}"
        if cat and cat != "XGBoost Error":
            lbl += f" ({cat})"
        labels.append(lbl)

    max_idx = len(labels) - 1
    cur_idx = min(st.session_state.get("selected_row_idx", 0), max_idx)

    chosen = st.selectbox("Select record:", labels, index=cur_idx, key="explain_sel")
    sel_idx = labels.index(chosen)
    st.session_state["selected_row_idx"] = sel_idx

    sel_result = results.iloc[sel_idx]
    binary_label = sel_result["binary_label"]
    attack_proba = sel_result["attack_proba"]
    row_id       = sel_result["row_id"]

    # ── Record summary cards ──────────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Row ID",     str(row_id))
    d2.metric("Prediction", binary_label)
    d3.metric("P(ATTACK)",  f"{attack_proba:.4f}")
    if binary_label == "ATTACK" and sel_result.get("attack_category"):
        cat = sel_result["attack_category"]
        if cat != "XGBoost Error":
            d4.metric("Attack Category", cat)
            if sel_result.get("category_proba"):
                st.caption(f"Category score: {sel_result['category_proba']:.4f}")
        else:
            d4.warning("XGBoost error")
    elif binary_label == "BENIGN":
        d4.success("BENIGN — Stage 2 not invoked")

    # Show preprocessed feature values on demand
    sel_raw = raw_df.iloc[sel_idx: sel_idx + 1].copy()
    with st.expander("Preprocessed feature values"):
        try:
            X_trans = svc.preprocess(sel_raw)
            st.dataframe(
                X_trans.T.rename(columns={X_trans.index[0]: "Scaled value"}),
                width='stretch'
            )
        except Exception as e:
            st.warning(f"Could not display preprocessed values: {e}")

    # ── Stage 1: RF SHAP ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Stage 1 · Random Forest — Binary Detection")

    with st.spinner("Computing RF SHAP values…"):
        try:
            X_trans = svc.preprocess(sel_raw)
            rf_exp  = svc.explain_rf_local(X_trans)
        except Exception as e:
            rf_exp = None
            st.error(f"Preprocessing failed: {e}")

    if rf_exp is not None:
        n_top_rf = st.slider("Features to display (RF)", 5, 25, 12, key="shap_rf_n")
        fig_rf = _shap_waterfall(
            rf_exp["shap_values"], rf_exp["feature_names"],
            rf_exp["feature_values"], rf_exp["expected_value"],
            title=f"RF SHAP — P(ATTACK) contributions  (Row {row_id})",
            n_top=n_top_rf,
        )
        st.plotly_chart(fig_rf, width='stretch')

        sv_arr = np.array(rf_exp["shap_values"])
        order  = np.argsort(np.abs(sv_arr))[::-1][:n_top_rf]
        shap_df = pd.DataFrame({
            "Feature":           [rf_exp["feature_names"][i] for i in order],
            "Scaled Value":      [round(rf_exp["feature_values"][i], 4)   for i in order],
            "SHAP Contribution": [round(float(sv_arr[i]), 6)               for i in order],
            "Direction":         ["→ ATTACK" if sv_arr[i] > 0 else "→ BENIGN" for i in order],
        })
        with st.expander("SHAP contribution table (RF)"):
            st.dataframe(shap_df, width='stretch', hide_index=True)

        st.caption(
            f"Base value: {rf_exp['expected_value']:.4f} · "
            f"Explaining: P(ATTACK) · Model: {rf_exp['model']}"
        )
    else:
        st.error("RF SHAP explanation could not be generated.")

    # ── Stage 2: XGBoost SHAP (only for ATTACK) ──────────────────────────────
    if binary_label == "ATTACK" and svc.status.xgb_ok:
        st.markdown("---")
        st.subheader("Stage 2 · XGBoost — Attack Classification")
        cat_name = sel_result.get("attack_category") or ""
        if cat_name and cat_name != "XGBoost Error":
            st.markdown(f"Explaining predicted class: **{cat_name}**")

        with st.spinner("Computing XGBoost SHAP values…"):
            try:
                X_trans_xgb = svc.preprocess(sel_raw)
                xgb_exp     = svc.explain_xgb_local(X_trans_xgb)
            except Exception as e:
                xgb_exp = None
                st.error(f"XGBoost SHAP error: {e}")

        if xgb_exp is not None:
            pred_class  = xgb_exp["predicted_class_name"]
            class_names = xgb_exp["class_names"]
            sv_matrix   = xgb_exp["shap_values"]   # (n_features, n_classes)
            fnames      = xgb_exp["feature_names"]
            fvals       = xgb_exp["feature_values"]
            base_vals   = xgb_exp["expected_value"]
            pred_idx    = xgb_exp["predicted_class_idx"]

            st.success(f"**Predicted attack category: {pred_class}**")

            sv_pred = sv_matrix[:, pred_idx] if sv_matrix.ndim == 2 else sv_matrix
            base_ev = float(base_vals[pred_idx]) \
                if hasattr(base_vals, "__len__") else float(base_vals)

            n_top_xgb = st.slider("Features to display (XGBoost)", 5, 25, 12, key="shap_xgb_n")
            fig_xgb = _shap_waterfall(
                sv_pred, fnames, fvals, base_ev,
                title=f"XGBoost SHAP — Class: {pred_class}  (Row {row_id})",
                n_top=n_top_xgb,
            )
            st.plotly_chart(fig_xgb, width='stretch')

            # Cross-class heatmap
            if sv_matrix.ndim == 2 and len(class_names) > 1:
                with st.expander("SHAP values across all 13 attack classes"):
                    top10 = np.argsort(np.abs(sv_matrix).mean(axis=1))[::-1][:10]
                    heat  = sv_matrix[top10, :]
                    hfeat = [fnames[i] for i in top10]
                    heat_fig = go.Figure(go.Heatmap(
                        z=heat.tolist(), x=class_names, y=hfeat,
                        colorscale="RdBu_r", zmid=0,
                        text=[[f"{v:.3f}" for v in row] for row in heat],
                        texttemplate="%{text}",
                    ))
                    heat_fig.update_layout(
                        title="Feature SHAP values × Attack Classes",
                        height=400, xaxis_tickangle=-35, template="plotly_white",
                    )
                    st.plotly_chart(heat_fig, width='stretch')

            st.caption(f"Model: {xgb_exp['model']}")
        else:
            st.warning(
                "XGBoost SHAP could not be generated. "
                "The binary ATTACK prediction above is still valid."
            )

    elif binary_label == "ATTACK" and not svc.status.xgb_ok:
        st.info(
            "XGBoost not loaded — attack-category SHAP unavailable. "
            "Binary ATTACK prediction is valid.",
            icon="ℹ️",
        )
    else:
        st.info(
            "This record is **BENIGN** — Stage 2 was not invoked. "
            "Only the Random Forest explanation above is applicable.",
            icon="✅",
        )

    st.divider()
    st.caption(
        "⚠️ SHAP values reflect learned model patterns, not causal relationships. "
        "A high SHAP value means the model weighted that feature heavily — "
        "it does not prove that feature caused the network attack."
    )

    if st.button("← Back to Results", key="explain_back"):
        nav_to("Results")
