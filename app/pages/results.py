"""Page 3 — Results: prediction summary, filterable table, download, record selection."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render(svc, nav_to):
    st.markdown("## 📊  Detection Results")

    results = st.session_state.get("results_df")
    if results is None or not st.session_state.get("analysis_ready"):
        st.info(
            "No analysis available yet. Go to **Analyze** to upload a CSV and run detection.",
            icon="📂",
        )
        if st.button("📂  Go to Analyze", type="primary", key="results_goto_analyze"):
            nav_to("Analyze")
        return

    st.caption(
        f"Results from the most recent analysis · "
        f"{len(results):,} records · "
        f"Click any row to explain that prediction."
    )

    n_total = len(results)
    n_attack = (results["binary_label"] == "ATTACK").sum()
    n_benign = n_total - n_attack
    has_cats = "attack_category" in results.columns
    n_classified = results["attack_category"].notna().sum() if has_cats else 0

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{n_total:,}")
    m2.metric("BENIGN",  f"{n_benign:,}", delta=f"{n_benign/n_total*100:.1f}%")
    m3.metric("ATTACK",  f"{n_attack:,}",
              delta=f"{n_attack/n_total*100:.1f}%", delta_color="inverse")
    if svc.status.xgb_ok and n_attack > 0:
        m4.metric("Attack Categories Found", f"{results['attack_category'].nunique():,}")
    elif not svc.status.xgb_ok:
        m4.info("XGBoost not loaded")
    else:
        m4.metric("Attack Categories Found", "0")

    # ── Charts ────────────────────────────────────────────────────────────────
    chart1, chart2 = st.columns(2)

    with chart1:
        pie = go.Figure(go.Pie(
            labels=["BENIGN", "ATTACK"],
            values=[n_benign, n_attack],
            marker_colors=["#16a34a", "#dc2626"],
            hole=0.45,
            textinfo="label+percent",
        ))
        pie.update_layout(title="Binary Detection Split", height=280,
                          margin=dict(t=40, b=10, l=10, r=10),
                          showlegend=False)
        st.plotly_chart(pie, width='stretch')

    with chart2:
        if n_attack > 0 and has_cats and n_classified > 0:
            cat_counts = (
                results[results["binary_label"] == "ATTACK"]["attack_category"]
                .value_counts()
            )
            bar = px.bar(
                x=cat_counts.values, y=cat_counts.index, orientation="h",
                color=cat_counts.values, color_continuous_scale="Reds",
                labels={"x": "Count", "y": ""},
                title="Attack Category Distribution",
            )
            bar.update_layout(height=280, showlegend=False,
                              yaxis=dict(autorange="reversed"),
                              margin=dict(t=40, b=10, l=10, r=10),
                              coloraxis_showscale=False)
            st.plotly_chart(bar, width='stretch')
        else:
            st.info(
                "No attack categories to display." if n_attack == 0
                else "XGBoost not loaded — attack categories unavailable.",
                icon="ℹ️",
            )

    # ── Filter + table ────────────────────────────────────────────────────────
    st.markdown("---")
    f_col, d_col = st.columns([3, 1])
    with f_col:
        filter_sel = st.selectbox(
            "Filter:", ["All", "ATTACK only", "BENIGN only"],
            key="results_filter", label_visibility="visible"
        )
    with d_col:
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download CSV",
            data=csv_bytes,
            file_name="threat_detection_results.csv",
            mime="text/csv",
            width='stretch',
        )

    display = results.copy()
    if filter_sel == "ATTACK only":
        display = display[display["binary_label"] == "ATTACK"]
    elif filter_sel == "BENIGN only":
        display = display[display["binary_label"] == "BENIGN"]

    # Nice column labels
    col_rename = {
        "row_id": "Row ID",
        "binary_label": "Prediction",
        "attack_proba": "P(ATTACK)",
        "attack_category": "Attack Category",
        "category_proba": "Category Prob.",
        "dt_label": "DT Prediction",
        "dt_attack_proba": "DT P(ATTACK)",
        "status": "Status",
    }
    show_cols = [c for c in col_rename if c in display.columns]
    table_df = display[show_cols].rename(columns=col_rename).reset_index(drop=True)

    st.dataframe(
        table_df.style.apply(
            lambda row: [
                "background-color:#fef2f2" if row.get("Prediction") == "ATTACK"
                else "background-color:#f0fdf4" if row.get("Prediction") == "BENIGN"
                else "" for _ in row
            ],
            axis=1,
        ),
        width='stretch',
        height=420,
    )

    st.caption(
        "P(ATTACK) = Random Forest model score (not a calibrated probability). "
        "Category Prob. = XGBoost max class score."
    )

    # ── Record selector for Explain ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍  Explain a Record")

    sel_col, btn_col = st.columns([3, 1])
    with sel_col:
        # Build human-readable labels
        labels = []
        for _, row in results.iterrows():
            tag = row["binary_label"]
            cat = row.get("attack_category") or ""
            lbl = f"Row {row['row_id']} — {tag}"
            if cat and cat != "XGBoost Error":
                lbl += f" ({cat})"
            labels.append(lbl)

        # Keep selection within bounds after a new upload
        max_idx = len(labels) - 1
        cur_idx = min(st.session_state.get("selected_row_idx", 0), max_idx)
        chosen = st.selectbox("Select a record to explain:", labels,
                              index=cur_idx, key="results_record_sel")
        st.session_state["selected_row_idx"] = labels.index(chosen)

    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍  Open Explain →", type="primary",
                     width='stretch', key="goto_explain"):
            nav_to("Explain")

    st.info(
        "💡  **Tip:** Select any record above and click **Open Explain** to "
        "view its SHAP feature contributions.",
        icon="💡",
    )
