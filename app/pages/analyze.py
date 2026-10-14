"""Page 2 — Analyze: CSV upload, validation, and Run Detection."""

import hashlib
import streamlit as st
import pandas as pd
import numpy as np


def _file_hash(f) -> str:
    f.seek(0)
    h = hashlib.md5(f.read()).hexdigest()
    f.seek(0)
    return h


def render(svc, nav_to):
    st.markdown("## 📂  Analyze Network Traffic")
    st.caption("Upload a CIC-IDS2017 network flow CSV, validate, then run the two-stage detection pipeline.")

    if not svc.status.can_predict_binary:
        st.error(
            "**Prediction unavailable** — required artifacts (preprocessor / RF model) are missing. "
            "See **System Info → Artifacts & Setup** for details."
        )
        return

    st.markdown("---")

    # ── Side-by-side: upload + options ───────────────────────────────────────
    upload_col, opt_col = st.columns([3, 1])

    with opt_col:
        st.markdown("#### ⚙️  Options")
        id_col_input = st.text_input(
            "Row identifier column",
            value="",
            placeholder="e.g. Flow ID",
            help="Column used as row label in results. Leave blank for row number.",
        )
        max_rows = st.number_input(
            "Max rows to process", min_value=10, max_value=10000, value=2000, step=100,
        )
        run_dt = st.checkbox(
            "Also run Decision Tree baseline",
            value=False,
            disabled=not svc.status.dt_ok,
            help="Requires binary_decision_tree.joblib.",
        )

    with upload_col:
        canonical = svc.get_canonical_columns()
        st.markdown(
            f"**Required format:** CIC-IDS2017 raw flow CSV &nbsp;·&nbsp; "
            f"{len(canonical)} canonical feature columns &nbsp;·&nbsp; "
            f"Max {max_rows:,} rows processed"
        )

        uploaded = st.file_uploader(
            "Drop your CSV here or click to browse",
            type=["csv"],
            label_visibility="collapsed",
        )

        with st.expander(f"📋  Required column names ({len(canonical)} columns)"):
            col_df = pd.DataFrame(
                {"Column Name": canonical[i] for i in range(len(canonical))
                 }.items() if False else
                [{"#": i + 1, "Column": c} for i, c in enumerate(canonical)]
            )
            st.dataframe(col_df.set_index("#"), width='stretch', height=280)

    if uploaded is None:
        st.info("👆  Upload a CSV file to begin.", icon="📂")
        return

    # ── Parse ─────────────────────────────────────────────────────────────────
    fhash = _file_hash(uploaded)

    # Detect new upload — clear previous results
    if fhash != st.session_state.get("file_hash"):
        st.session_state["results_df"] = None
        st.session_state["raw_df"] = None
        st.session_state["analysis_ready"] = False
        st.session_state["selected_row_idx"] = 0

    try:
        raw_df = pd.read_csv(uploaded, low_memory=False)
    except Exception as e:
        st.error(f"❌  Failed to parse CSV: {e}")
        return

    if len(raw_df) > max_rows:
        st.warning(f"File has {len(raw_df):,} rows — processing first {max_rows:,}.")
        raw_df = raw_df.head(max_rows)

    st.success(f"✅  Parsed: **{len(raw_df):,} rows × {len(raw_df.columns)} columns**")

    # ── Validate ──────────────────────────────────────────────────────────────
    is_valid, val_errors, val_warnings = svc.validate_uploaded_csv(raw_df)
    st.session_state["val_errors"] = val_errors
    st.session_state["val_warnings"] = val_warnings

    for w in val_warnings:
        st.warning(w)

    if not is_valid:
        for e in val_errors:
            st.error(e)
        st.info(
            "**Tip:** Make sure your CSV contains raw CIC-IDS2017 network flow statistics "
            "(79 columns). Do not upload the cleaned/preprocessed dataset."
        )
        return

    st.session_state["file_hash"] = fhash
    st.session_state["raw_df"] = raw_df

    id_col = id_col_input.strip() or None
    if id_col and id_col not in raw_df.columns:
        st.warning(f"Column '{id_col}' not found — using row number.")
        id_col = None

    # ── Run Detection ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Ready to detect threats")

    c_info, c_btn = st.columns([3, 1])
    with c_info:
        st.markdown(
            f"- **{len(raw_df):,}** records validated  \n"
            f"- Stage 1 → Random Forest binary detection  \n"
            f"{'- Stage 2 → XGBoost attack classification  ' if svc.status.xgb_ok else '- Stage 2 → XGBoost **unavailable** (model not loaded)  '}"
        )
    with c_btn:
        run_clicked = st.button("🚀  Run Detection", type="primary",
                                width='stretch', key="btn_run")

    if run_clicked:
        with st.spinner("Running two-stage detection pipeline…"):
            try:
                results = svc.run_full_pipeline(raw_df, id_col=id_col)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                return

        if run_dt and svc.status.dt_ok:
            with st.spinner("Running Decision Tree baseline…"):
                try:
                    X_trans = svc.preprocess(raw_df)
                    dt_preds, dt_proba = svc.predict_binary_dt(X_trans)
                    results["dt_label"] = ["ATTACK" if p == 1 else "BENIGN" for p in dt_preds]
                    results["dt_attack_proba"] = [round(float(p), 4) for p in dt_proba]
                except Exception as e:
                    st.warning(f"Decision Tree baseline failed: {e}")

        # Store results
        st.session_state["results_df"] = results
        st.session_state["analysis_ready"] = True
        st.session_state["selected_row_idx"] = 0

        # Quick summary
        n_total = len(results)
        n_attack = (results["binary_label"] == "ATTACK").sum()
        n_benign = n_total - n_attack

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", f"{n_total:,}")
        m2.metric("BENIGN",  f"{n_benign:,}", delta=f"{100-n_attack/n_total*100:.1f}%")
        m3.metric("ATTACK",  f"{n_attack:,}",
                  delta=f"{n_attack/n_total*100:.1f}%", delta_color="inverse")

        st.success(
            f"Detection complete — **{n_attack:,} ATTACK** and **{n_benign:,} BENIGN** "
            f"records found. Click below to view the full results."
        )
        if st.button("📊  View Results →", type="primary", key="goto_results"):
            nav_to("Results")
