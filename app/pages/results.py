"""Page 3 — Results: Prediction summary, filterable results table, download, and explain trigger."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render(svc, nav_to):
    st.markdown("## 📊  Detection Results")

    results = st.session_state.get("results_df")
    if results is None or not st.session_state.get("analysis_ready"):
        st.info(
            "No traffic analysis has been run yet. Please go to **Analyze** to enter network details or upload a CSV file and check for threats.",
            icon="📂",
        )
        if st.button("📂  Go to Analyze", type="primary", key="results_goto_analyze"):
            nav_to("Analyze")
        return

    n_total = len(results)
    n_attack = int((results["binary_label"] == "ATTACK").sum())
    n_benign = n_total - n_attack
    has_cats = "attack_category" in results.columns
    n_classified = int(results["attack_category"].notna().sum()) if has_cats else 0

    st.caption(
        f"Displaying results for {n_total:,} record(s). "
        "Select any record below and click **Why Was This Result Given?** to explore the model's reasoning."
    )

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{n_total:,}")
    m2.metric(
        "No Threat Detected",
        f"{n_benign:,}",
        delta=f"{(n_benign / n_total) * 100:.1f}%" if n_total > 0 else "0%",
    )
    m3.metric(
        "Threat Detected",
        f"{n_attack:,}",
        delta=f"{(n_attack / n_total) * 100:.1f}%" if n_total > 0 else "0%",
        delta_color="inverse",
    )
    if svc.status.xgb_ok and n_attack > 0:
        n_unique_cats = results[results["binary_label"] == "ATTACK"]["attack_category"].dropna().nunique()
        m4.metric("Possible Attack Types", f"{n_unique_cats:,}")
    elif not svc.status.xgb_ok:
        m4.info("XGBoost classifier unavailable")
    else:
        m4.metric("Possible Attack Types", "0")

    # ── Charts (for batch results) ────────────────────────────────────────────
    if n_total > 1:
        chart1, chart2 = st.columns(2)

        with chart1:
            pie = go.Figure(go.Pie(
                labels=["No Threat Detected", "Threat Detected"],
                values=[n_benign, n_attack],
                marker_colors=["#16a34a", "#dc2626"],
                hole=0.45,
                textinfo="label+percent",
            ))
            pie.update_layout(
                title="Overall Threat Detection Split",
                height=280,
                margin=dict(t=40, b=10, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(pie, width='stretch')

        with chart2:
            if n_attack > 0 and has_cats and n_classified > 0:
                cat_counts = (
                    results[results["binary_label"] == "ATTACK"]["attack_category"]
                    .dropna()
                    .value_counts()
                )
                bar = px.bar(
                    x=cat_counts.values,
                    y=cat_counts.index,
                    orientation="h",
                    color=cat_counts.values,
                    color_continuous_scale="Reds",
                    labels={"x": "Count", "y": ""},
                    title="Possible Attack Type Distribution",
                )
                bar.update_layout(
                    height=280,
                    showlegend=False,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=40, b=10, l=10, r=10),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(bar, width='stretch')
            else:
                st.info(
                    "No attack categories to display — all analyzed flows are safe." if n_attack == 0
                    else "Attack categorization model unavailable.",
                    icon="ℹ️",
                )

    # ── Results Table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Detailed Records")

    f_col, d_col = st.columns([3, 1])
    with f_col:
        filter_sel = st.selectbox(
            "Filter records:",
            ["All Records", "Threat Detected only", "No Threat Detected only"],
            key="results_filter",
        )
    with d_col:
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download Results CSV",
            data=csv_bytes,
            file_name="network_threat_detection_results.csv",
            mime="text/csv",
            width='stretch',
        )

    display = results.copy()
    if filter_sel == "Threat Detected only":
        display = display[display["binary_label"] == "ATTACK"]
    elif filter_sel == "No Threat Detected only":
        display = display[display["binary_label"] == "BENIGN"]

    # Convert binary label to user-friendly label for display
    display["display_label"] = display["binary_label"].map({
        "ATTACK": "Threat Detected",
        "BENIGN": "No Threat Detected",
    }).fillna(display["binary_label"])

    col_rename = {
        "row_id": "Record ID",
        "display_label": "Prediction",
        "attack_proba": "Threat Probability",
        "attack_category": "Possible Attack Type",
        "category_proba": "Attack Type Confidence",
        "dt_label": "Baseline Prediction",
        "dt_attack_proba": "Baseline Threat Score",
        "status": "Status",
    }
    show_cols = [c for c in col_rename if c in display.columns]
    table_df = display[show_cols].rename(columns=col_rename).reset_index(drop=True)

    # Render formatted table
    st.dataframe(
        table_df.style.apply(
            lambda row: [
                "background-color:#fef2f2; font-weight:600; color:#991b1b" if row.get("Prediction") == "Threat Detected"
                else "background-color:#f0fdf4; font-weight:600; color:#166534" if row.get("Prediction") == "No Threat Detected"
                else "" for _ in row
            ],
            axis=1,
        ),
        width='stretch',
        height=min(420, max(180, (len(table_df) + 1) * 38)),
    )

    st.caption(
        "Threat Probability: Model confidence score indicating likelihood of an attack. "
        "Possible Attack Type: Specific attack category identified when a threat is detected."
    )

    # ── Explain Record Section ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍  Why Was This Result Given?")
    st.caption("Select any record above to see the specific network features that drove the prediction.")

    sel_col, btn_col = st.columns([3, 1.5])
    with sel_col:
        # Build plain-English labels for selector
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
        chosen = st.selectbox(
            "Select record to explain:",
            labels,
            index=cur_idx,
            key="results_record_sel",
        )
        st.session_state["selected_row_idx"] = labels.index(chosen)

    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍  Why Was This Result Given?", type="primary", width='stretch', key="goto_explain"):
            nav_to("Explain")

    st.info(
        "💡  Click **Why Was This Result Given?** above to view the SHAP feature contributions for the selected record.",
        icon="💡",
    )
