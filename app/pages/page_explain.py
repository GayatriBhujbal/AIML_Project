"""Page C — Explain a Prediction using SHAP."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def _shap_waterfall(shap_values, feature_names, feature_values, expected_value,
                    title="SHAP Waterfall", n_top=12):
    """Build a Plotly waterfall chart from SHAP values."""
    sv = np.array(shap_values)
    order = np.argsort(np.abs(sv))[::-1][:n_top]
    top_names = [feature_names[i] for i in order]
    top_sv = [sv[i] for i in order]
    top_fv = [feature_values[i] for i in order]

    # Sort by value for cleaner display
    sorted_idx = np.argsort(top_sv)
    top_names = [top_names[i] for i in sorted_idx]
    top_sv = [top_sv[i] for i in sorted_idx]
    top_fv = [top_fv[i] for i in sorted_idx]

    colors = ["#DC2626" if v > 0 else "#2563EB" for v in top_sv]
    labels = [
        f"{n}<br><i>val={v:.3g}</i>" for n, v in zip(top_names, top_fv)
    ]

    fig = go.Figure(go.Bar(
        x=top_sv,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in top_sv],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="SHAP Value (contribution to prediction)",
        height=max(350, len(top_names) * 36),
        template="plotly_white",
        annotations=[dict(
            x=0, y=-0.12, xref="paper", yref="paper",
            text=f"Base value: {expected_value:.4f}",
            showarrow=False, font=dict(size=11, color="gray")
        )],
    )
    return fig


def render(svc):
    st.title("🔍 Explain a Prediction")
    st.markdown(
        "Select a record from the last uploaded batch to inspect its SHAP-based explanation. "
        "SHAP values show **how much each feature contributed** to the model's output — "
        "positive values push toward ATTACK, negative values push toward BENIGN."
    )

    # ── Check for uploaded data ───────────────────────────────────────────────
    if "last_results" not in st.session_state or "last_raw_df" not in st.session_state:
        st.info(
            "No predictions available yet. Go to **B · Predict from CSV** and upload a file first.",
            icon="📂",
        )
        return

    results = st.session_state["last_results"]
    raw_df = st.session_state["last_raw_df"]

    if not svc.status.can_predict_binary:
        st.error("Models not loaded — explanation unavailable.")
        return

    # ── Record selector ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("1️⃣ Select a Record")

    # Build selector labels
    labels = []
    for _, row in results.iterrows():
        tag = row["binary_label"]
        cat = row.get("attack_category") or ""
        labels.append(f"Row {row['row_id']} — {tag}{' (' + cat + ')' if cat else ''}")

    selected_label = st.selectbox("Choose a record:", labels)
    sel_idx = labels.index(selected_label)

    sel_result = results.iloc[sel_idx]
    sel_raw = raw_df.iloc[sel_idx : sel_idx + 1].copy()
    sel_row_id = sel_result["row_id"]
    binary_label = sel_result["binary_label"]
    attack_proba = sel_result["attack_proba"]

    # ── Record details ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2️⃣ Record Details")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Row ID", str(sel_row_id))
    d2.metric("Binary Prediction", binary_label)
    d3.metric("P(ATTACK)", f"{attack_proba:.4f}")
    if binary_label == "ATTACK" and sel_result.get("attack_category"):
        d4.metric("Attack Category", sel_result["attack_category"])
        if sel_result.get("category_proba"):
            st.caption(f"Category probability: {sel_result['category_proba']:.4f}")
    elif binary_label == "BENIGN":
        d4.info("BENIGN — Stage 2 not invoked")

    # Show preprocessed feature values
    with st.expander("Preprocessed feature values used by models"):
        try:
            X_trans = svc.preprocess(sel_raw)
            st.dataframe(X_trans.T.rename(columns={X_trans.index[0]: "scaled value"}),
                         use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display preprocessed values: {e}")

    # ── Random Forest SHAP ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3️⃣ Random Forest SHAP Explanation (Binary Detection)")
    st.caption("SHAP values show each feature's contribution to P(ATTACK). "
               "Red bars increase P(ATTACK); blue bars decrease it.")

    with st.spinner("Computing Random Forest SHAP values…"):
        try:
            X_trans = svc.preprocess(sel_raw)
            rf_explanation = svc.explain_rf_local(X_trans)
        except Exception as e:
            rf_explanation = None
            st.error(f"Preprocessing failed: {e}")

    if rf_explanation is not None:
        sv = rf_explanation["shap_values"]
        ev = rf_explanation["expected_value"]
        fnames = rf_explanation["feature_names"]
        fvals = rf_explanation["feature_values"]

        n_top = st.slider("Number of top features to display (RF)", 5, 25, 12)
        fig_rf = _shap_waterfall(sv, fnames, fvals, ev,
                                 title=f"RF SHAP — P(ATTACK) contribution (Record {sel_row_id})",
                                 n_top=n_top)
        st.plotly_chart(fig_rf, use_container_width=True)

        # Top contributors table
        sv_arr = np.array(sv)
        order = np.argsort(np.abs(sv_arr))[::-1][:n_top]
        shap_df = pd.DataFrame({
            "Feature": [fnames[i] for i in order],
            "Scaled Value": [round(fvals[i], 4) for i in order],
            "SHAP Contribution": [round(float(sv_arr[i]), 6) for i in order],
            "Direction": ["→ ATTACK" if sv_arr[i] > 0 else "→ BENIGN" for i in order],
        })
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

        st.caption(
            f"Base value (expected model output): {ev:.4f} | "
            f"Model: {rf_explanation['model']} | "
            f"Explaining: {rf_explanation['class_explained']}"
        )
    else:
        st.error("SHAP explanation for Random Forest could not be generated.")

    # ── XGBoost SHAP (only for ATTACK records) ───────────────────────────────
    if binary_label == "ATTACK" and svc.status.xgb_ok:
        st.markdown("---")
        st.subheader("4️⃣ XGBoost SHAP Explanation (Attack Classification)")
        st.caption(
            "SHAP values show each feature's contribution to the predicted attack class. "
            "Each class gets its own SHAP vector; the class with the highest sum is predicted."
        )

        with st.spinner("Computing XGBoost SHAP values…"):
            try:
                X_trans_xgb = svc.preprocess(sel_raw)
                xgb_explanation = svc.explain_xgb_local(X_trans_xgb)
            except Exception as e:
                xgb_explanation = None
                st.error(f"XGBoost SHAP error: {e}")

        if xgb_explanation is not None:
            pred_class = xgb_explanation["predicted_class_name"]
            class_names = xgb_explanation["class_names"]
            sv_matrix = xgb_explanation["shap_values"]   # (n_features, n_classes)
            fnames = xgb_explanation["feature_names"]
            fvals = xgb_explanation["feature_values"]
            base_vals = xgb_explanation["expected_value"]
            pred_idx = xgb_explanation["predicted_class_idx"]

            st.success(f"**Predicted attack category: {pred_class}**")

            # Show SHAP for predicted class
            sv_pred_class = sv_matrix[:, pred_idx] if sv_matrix.ndim == 2 else sv_matrix

            n_top_xgb = st.slider("Number of top features to display (XGBoost)", 5, 25, 12)
            base_ev = float(base_vals[pred_idx]) if hasattr(base_vals, "__len__") else float(base_vals)
            fig_xgb = _shap_waterfall(
                sv_pred_class, fnames, fvals, base_ev,
                title=f"XGBoost SHAP — Predicted class: {pred_class} (Record {sel_row_id})",
                n_top=n_top_xgb,
            )
            st.plotly_chart(fig_xgb, use_container_width=True)

            # Heatmap: all classes vs top features
            if sv_matrix.ndim == 2 and len(class_names) > 1:
                with st.expander("View SHAP values across all attack classes"):
                    top_feat_idx = np.argsort(np.abs(sv_matrix).mean(axis=1))[::-1][:10]
                    heat_data = sv_matrix[top_feat_idx, :]
                    heat_feats = [fnames[i] for i in top_feat_idx]

                    heat_fig = go.Figure(go.Heatmap(
                        z=heat_data.tolist(),
                        x=class_names,
                        y=heat_feats,
                        colorscale="RdBu_r",
                        zmid=0,
                        text=[[f"{v:.3f}" for v in row] for row in heat_data],
                        texttemplate="%{text}",
                    ))
                    heat_fig.update_layout(
                        title="SHAP Values: Top Features × All Attack Classes",
                        height=400,
                        xaxis_tickangle=-35,
                        template="plotly_white",
                    )
                    st.plotly_chart(heat_fig, use_container_width=True)

            st.caption(f"Model: {xgb_explanation['model']}")
        else:
            st.warning(
                "XGBoost SHAP explanation could not be generated for this record. "
                "The binary prediction (ATTACK) is still valid."
            )

    elif binary_label == "ATTACK" and not svc.status.xgb_ok:
        st.info(
            "XGBoost model not loaded — attack-category SHAP explanation unavailable. "
            "Binary prediction is still valid.",
            icon="ℹ️",
        )
    elif binary_label == "BENIGN":
        st.info(
            "Record is predicted **BENIGN** — Stage 2 (XGBoost) was not invoked. "
            "Only the Random Forest explanation above is applicable.",
            icon="✅",
        )

    st.divider()
    st.caption(
        "**Interpretation note:** SHAP values reflect the model's learned patterns, not causal relationships. "
        "A high SHAP value for a feature means the model gave that feature significant weight for this prediction. "
        "It does not prove that feature is the actual cause of the network attack."
    )
