"""Page B — Predict from CSV upload."""

import io
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def _file_hash(uploaded_file) -> str:
    """Stable hash of file content for cache key."""
    uploaded_file.seek(0)
    h = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h


def render(svc):
    st.title("📂 Predict from CSV")
    st.markdown(
        "Upload a CSV file containing raw CIC-IDS2017 network flow records. "
        "The application will run both detection stages and return per-row predictions."
    )

    if not svc.status.can_predict_binary:
        st.error(
            "**Prediction unavailable** — required artifacts are missing:\n" +
            "\n".join(f"• {e}" for e in svc.status.errors)
        )
        return

    # ── Upload widget ─────────────────────────────────────────────────────────
    st.markdown("---")
    col_up, col_cfg = st.columns([2, 1])

    with col_cfg:
        st.markdown("#### Options")
        id_col_input = st.text_input(
            "Row identifier column (optional)",
            value="",
            help="Column name to use as row identifier (e.g. 'Flow ID'). Leave blank to use row number.",
            placeholder="e.g. Flow ID",
        )
        max_rows = st.number_input("Max rows to process", min_value=10, max_value=10000, value=2000, step=100)
        show_raw = st.checkbox("Show raw uploaded data preview", value=False)
        run_dt_baseline = st.checkbox(
            "Also run Decision Tree baseline",
            value=svc.status.dt_ok,
            disabled=not svc.status.dt_ok,
            help="Requires binary_decision_tree.joblib to be loaded.",
        )

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload CIC-IDS2017 format CSV",
            type=["csv"],
            help=(
                f"Required columns: {len(svc.get_canonical_columns())} canonical CIC-IDS2017 features. "
                f"Max rows processed: {max_rows:,}."
            ),
        )

    if uploaded_file is None:
        st.info("👆 Upload a CSV file to get started.", icon="📂")

        # Show required column schema
        with st.expander("📋 Required input columns"):
            cols = svc.get_canonical_columns()
            if cols:
                col_df = pd.DataFrame({"Required Column": cols, "#": range(1, len(cols)+1)}).set_index("#")
                st.dataframe(col_df, width='stretch')
            else:
                st.warning("Column schema unavailable (preprocessor not loaded).")
        return

    # ── Parse uploaded file ───────────────────────────────────────────────────
    file_hash = _file_hash(uploaded_file)

    try:
        raw_df = pd.read_csv(uploaded_file, low_memory=False)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        return

    if len(raw_df) > max_rows:
        st.warning(f"File has {len(raw_df):,} rows. Processing first {max_rows:,} only.")
        raw_df = raw_df.head(max_rows)

    st.success(f"File parsed: **{len(raw_df):,} rows × {len(raw_df.columns)} columns**")

    if show_raw:
        st.dataframe(raw_df.head(20), width='stretch')

    # ── Validate columns ──────────────────────────────────────────────────────
    is_valid, val_errors, val_warnings = svc.validate_uploaded_csv(raw_df)

    for w in val_warnings:
        st.warning(w)

    if not is_valid:
        for e in val_errors:
            st.error(e)
        return

    # ── Determine ID column ───────────────────────────────────────────────────
    id_col = id_col_input.strip() if id_col_input.strip() else None
    if id_col and id_col not in raw_df.columns:
        st.warning(f"Identifier column '{id_col}' not found. Using row number instead.")
        id_col = None

    # ── Run prediction ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.spinner("Running two-stage detection pipeline…"):
        try:
            results = svc.run_full_pipeline(raw_df, id_col=id_col)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return

    # Optionally run DT baseline
    if run_dt_baseline and svc.status.dt_ok:
        with st.spinner("Running Decision Tree baseline…"):
            try:
                X_trans = svc.preprocess(raw_df)
                dt_preds, dt_proba = svc.predict_binary_dt(X_trans)
                results["dt_label"] = ["ATTACK" if p == 1 else "BENIGN" for p in dt_preds]
                results["dt_attack_proba"] = [round(float(p), 4) for p in dt_proba]
            except Exception as e:
                st.warning(f"Decision Tree baseline failed: {e}")

    # Store in session state for Explain page
    st.session_state["last_results"] = results
    st.session_state["last_raw_df"] = raw_df
    st.session_state["last_file_hash"] = file_hash

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("📊 Prediction Summary")

    n_total = len(results)
    n_attack = (results["binary_label"] == "ATTACK").sum()
    n_benign = n_total - n_attack
    attack_pct = n_attack / n_total * 100

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Records", f"{n_total:,}")
    s2.metric("BENIGN", f"{n_benign:,}", delta=f"{100-attack_pct:.1f}%")
    s3.metric("ATTACK", f"{n_attack:,}", delta=f"{attack_pct:.1f}%", delta_color="inverse")

    if svc.status.can_predict_attack and n_attack > 0:
        n_classified = results["attack_category"].notna().sum()
        s4.metric("Attack Categories Identified", f"{n_classified:,}")
    else:
        s4.metric("XGBoost Classification", "Unavailable" if not svc.status.xgb_ok else "No attacks detected")

    # Charts
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        pie = go.Figure(go.Pie(
            labels=["BENIGN", "ATTACK"],
            values=[n_benign, n_attack],
            marker_colors=["#16A34A", "#DC2626"],
            hole=0.4,
        ))
        pie.update_layout(title="Binary Detection Distribution", height=300, showlegend=True)
        st.plotly_chart(pie, width='stretch')

    with col_c2:
        if n_attack > 0 and svc.status.can_predict_attack:
            cat_counts = results[results["binary_label"] == "ATTACK"]["attack_category"].value_counts()
            bar = px.bar(
                x=cat_counts.values, y=cat_counts.index, orientation="h",
                color=cat_counts.values, color_continuous_scale="Reds",
                labels={"x": "Count", "y": "Attack Category"},
                title="Attack Category Distribution",
            )
            bar.update_layout(height=300, showlegend=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(bar, width='stretch')
        elif n_attack > 0:
            st.info("XGBoost model not loaded — attack categories unavailable.")
        else:
            st.info("No ATTACK records detected in this batch.")

    # ── Results table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Per-Record Predictions")

    # Colour ATTACK rows
    display_cols = ["row_id", "binary_label", "attack_proba"]
    if "attack_category" in results.columns:
        display_cols.append("attack_category")
    if "category_proba" in results.columns:
        display_cols.append("category_proba")
    if "dt_label" in results.columns:
        display_cols += ["dt_label", "dt_attack_proba"]

    display_df = results[display_cols].copy()
    display_df.columns = [c.replace("_", " ").title() for c in display_cols]

    # Filter controls
    filter_opts = ["All", "ATTACK only", "BENIGN only"]
    filter_sel = st.selectbox("Filter results:", filter_opts)
    if filter_sel == "ATTACK only":
        display_df = display_df[results["binary_label"] == "ATTACK"]
    elif filter_sel == "BENIGN only":
        display_df = display_df[results["binary_label"] == "BENIGN"]

    st.dataframe(
        display_df.style.apply(
            lambda row: [
                "background-color: #FEE2E2" if row.get("Binary Label") == "ATTACK"
                else "background-color: #D1FAE5" if row.get("Binary Label") == "BENIGN"
                else "" for _ in row
            ],
            axis=1,
        ),
        width='stretch',
        height=400,
    )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=csv_bytes,
        file_name="threat_detection_predictions.csv",
        mime="text/csv",
    )

    st.caption(
        "**Note:** 'Attack Proba' is the model's P(ATTACK) score. "
        "'Category Proba' is XGBoost's max class probability. "
        "These are model scores, not calibrated confidence intervals."
    )
    st.info(
        "💡 **Tip:** Go to **C · Explain a Prediction** to inspect SHAP explanations "
        "for any individual record from this upload.",
        icon="💡",
    )
