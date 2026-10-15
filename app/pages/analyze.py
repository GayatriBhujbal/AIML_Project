"""Page 2 — Analyze: Manual data entry, CSV upload, and threat detection."""

import hashlib
import io
import streamlit as st
import pandas as pd
import numpy as np


# Realistic baseline presets for student demo
BENIGN_PRESET = {
    "Destination Port": 443.0,
    "Flow Bytes/s": 1540.0,
    "Flow IAT Min": 4.0,
    "Fwd Packet Length Max": 482.0,
    "Fwd Packet Length Std": 128.5,
    "Avg Fwd Segment Size": 142.0,
    "Bwd Packet Length Mean": 320.0,
    "Bwd Packet Length Std": 240.0,
    "Min Packet Length": 0.0,
    "Packet Length Std": 182.0,
    "Packet Length Variance": 33124.0,
    "Average Packet Size": 195.0,
    "Subflow Fwd Bytes": 1420.0,
    "Subflow Bwd Bytes": 3200.0,
    "Init_Win_bytes_forward": 29200.0,
    "Init_Win_bytes_backward": 28960.0,
    "Fwd Packets/s": 42.0,
    "Fwd IAT Total": 48500.0,
    "Fwd IAT Std": 1150.0,
    "Fwd IAT Max": 24000.0,
    "Fwd IAT Min": 5.0,
    "Bwd IAT Mean": 8200.0,
    "Bwd IAT Std": 2100.0,
    "Bwd IAT Max": 17500.0,
    "Active Mean": 0.0,
}

ATTACK_PRESET = {
    "Destination Port": 80.0,
    "Flow Bytes/s": 850000.0,
    "Flow IAT Min": 1.0,
    "Fwd Packet Length Max": 20.0,
    "Fwd Packet Length Std": 0.0,
    "Avg Fwd Segment Size": 20.0,
    "Bwd Packet Length Mean": 0.0,
    "Bwd Packet Length Std": 0.0,
    "Min Packet Length": 20.0,
    "Packet Length Std": 0.0,
    "Packet Length Variance": 0.0,
    "Average Packet Size": 20.0,
    "Subflow Fwd Bytes": 50000.0,
    "Subflow Bwd Bytes": 0.0,
    "Init_Win_bytes_forward": 256.0,
    "Init_Win_bytes_backward": -1.0,
    "Fwd Packets/s": 12500.0,
    "Fwd IAT Total": 2000.0,
    "Fwd IAT Std": 15.0,
    "Fwd IAT Max": 120.0,
    "Fwd IAT Min": 1.0,
    "Bwd IAT Mean": 0.0,
    "Bwd IAT Std": 0.0,
    "Bwd IAT Max": 0.0,
    "Active Mean": 0.0,
}


def _file_hash(f) -> str:
    f.seek(0)
    h = hashlib.md5(f.read()).hexdigest()
    f.seek(0)
    return h


def render(svc, nav_to):
    st.markdown("## 📂  Check for Threats")
    st.caption("Enter network traffic details manually or upload a CSV file to test for network intrusions.")

    if not svc.status.can_predict_binary:
        st.error(
            "Threat detection is currently unavailable — model artifacts are missing. "
            "Please check **System Info** for setup details."
        )
        return

    st.markdown("---")

    # ── Input mode tabs: CSV Upload vs Manual Entry ───────────────────────────
    tab_csv, tab_manual = st.tabs(["📁  Upload a CSV File", "✍️  Enter Network Details"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: CSV Upload
    # ══════════════════════════════════════════════════════════════════════════
    with tab_csv:
        upload_col, opt_col = st.columns([3, 1])

        with opt_col:
            st.markdown("#### ⚙️  Options")
            id_col_input = st.text_input(
                "Row identifier column",
                value="",
                placeholder="e.g. Flow ID",
                help="Column used as label for records in results. Leave blank to use row numbers.",
            )
            max_rows = st.number_input(
                "Max records to check", min_value=10, max_value=10000, value=2000, step=100,
            )

        with upload_col:
            st.markdown("#### Upload a CSV File")
            st.caption("Upload network traffic records in CSV format. The required features will be automatically identified.")

            uploaded = st.file_uploader(
                "Select a CSV file to check",
                type=["csv"],
                key="csv_file_uploader",
            )

            req_features = svc.get_required_features()
            with st.expander(f"📋  View Required Features ({len(req_features)} features)"):
                st.caption(
                    "Your CSV should include the following network measurements. Any extra columns will be safely ignored."
                )
                feat_table = pd.DataFrame([{"#": i + 1, "Feature Name": f} for i, f in enumerate(req_features)])
                st.dataframe(feat_table.set_index("#"), width='stretch', height=240)

                # Provide a quick download of a template CSV
                sample_template = pd.DataFrame([BENIGN_PRESET])
                csv_template_bytes = sample_template.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️  Download Sample CSV Template",
                    data=csv_template_bytes,
                    file_name="network_traffic_sample_template.csv",
                    mime="text/csv",
                )

        if uploaded is not None:
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
                st.error(f"Failed to read CSV file: {e}")
                raw_df = None

            if raw_df is not None:
                if len(raw_df) > max_rows:
                    st.warning(f"File contains {len(raw_df):,} records. Processing the first {max_rows:,} records.")
                    raw_df = raw_df.head(max_rows)

                # Validate
                is_valid, val_errors, val_warnings = svc.validate_uploaded_csv(raw_df)
                st.session_state["val_errors"] = val_errors
                st.session_state["val_warnings"] = val_warnings

                for w in val_warnings:
                    st.info(w)

                if not is_valid:
                    for e in val_errors:
                        st.error(e)
                    st.info(
                        "Please verify that your CSV contains the required network traffic features. "
                        "You can download the sample template above to check the required columns."
                    )
                else:
                    st.success(f"✅  Validated {len(raw_df):,} record(s) ready for analysis.")
                    st.session_state["file_hash"] = fhash

                    id_col = id_col_input.strip() or None
                    if id_col and id_col not in raw_df.columns:
                        st.warning(f"Column '{id_col}' not found — using row numbers as identifiers.")
                        id_col = None

                    c_act, _ = st.columns([1, 2])
                    with c_act:
                        if st.button("🔍  Check for Threats", type="primary", key="btn_run_csv", use_container_width=True):
                            with st.spinner("Checking network traffic for threats…"):
                                try:
                                    results = svc.run_full_pipeline(raw_df, id_col=id_col)
                                    st.session_state["results_df"] = results
                                    st.session_state["raw_df"] = raw_df
                                    st.session_state["analysis_ready"] = True
                                    st.session_state["selected_row_idx"] = 0
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Threat check error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: Enter Network Details (Manual Data Entry)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_manual:
        st.markdown("#### Enter Network Details")
        st.caption("Provide network measurements directly below to evaluate a single connection for threats.")

        # Preset action buttons for quick testing
        st.markdown("**Quick Fill Presets:**")
        p_col1, p_col2, p_col3, _ = st.columns([1.5, 1.5, 1.2, 2])

        with p_col1:
            if st.button("🟢  Load Normal Traffic Sample", key="btn_preset_benign", use_container_width=True):
                for k, v in BENIGN_PRESET.items():
                    st.session_state[f"manual_{k}"] = float(v)
                st.rerun()

        with p_col2:
            if st.button("🔴  Load Suspicious Attack Sample", key="btn_preset_attack", use_container_width=True):
                for k, v in ATTACK_PRESET.items():
                    st.session_state[f"manual_{k}"] = float(v)
                st.rerun()

        with p_col3:
            if st.button("🔄  Reset Values", key="btn_preset_reset", use_container_width=True):
                for k in BENIGN_PRESET.keys():
                    st.session_state[f"manual_{k}"] = 0.0
                st.rerun()

        # Helper to get current value with default
        def _get_val(feat_key: str, default_val: float) -> float:
            sk = f"manual_{feat_key}"
            if sk not in st.session_state:
                st.session_state[sk] = float(default_val)
            return float(st.session_state[sk])

        # ── Group 1: Connection & Destination Port ──
        st.markdown("---")
        st.markdown("##### 1. Network Connection")
        c1, c2, c3 = st.columns(3)
        with c1:
            dest_port = st.number_input(
                "Destination Port",
                min_value=0, max_value=65535,
                value=int(_get_val("Destination Port", 443.0)),
                help="Network service port (e.g., 80 for HTTP, 443 for HTTPS, 21 for FTP, 22 for SSH)",
                key="input_Destination Port"
            )
            st.session_state["manual_Destination Port"] = float(dest_port)

        # ── Group 2: Traffic Volume & Speed ──
        st.markdown("##### 2. Traffic Volume & Speed")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            f_bytes_s = st.number_input(
                "Flow Bytes/s",
                value=_get_val("Flow Bytes/s", 1540.0),
                help="Rate of byte transmission per second",
                key="input_Flow Bytes/s"
            )
            st.session_state["manual_Flow Bytes/s"] = float(f_bytes_s)
        with v2:
            f_pkts_s = st.number_input(
                "Forward Packets/s",
                value=_get_val("Fwd Packets/s", 42.0),
                help="Rate of forward packet transmission per second",
                key="input_Fwd Packets/s"
            )
            st.session_state["manual_Fwd Packets/s"] = float(f_pkts_s)
        with v3:
            sub_fwd = st.number_input(
                "Subflow Forward Bytes",
                value=_get_val("Subflow Fwd Bytes", 1420.0),
                help="Average bytes transferred in forward subflows",
                key="input_Subflow Fwd Bytes"
            )
            st.session_state["manual_Subflow Fwd Bytes"] = float(sub_fwd)
        with v4:
            sub_bwd = st.number_input(
                "Subflow Backward Bytes",
                value=_get_val("Subflow Bwd Bytes", 3200.0),
                help="Average bytes transferred in backward subflows",
                key="input_Subflow Bwd Bytes"
            )
            st.session_state["manual_Subflow Bwd Bytes"] = float(sub_bwd)

        # ── Group 3: Packet Length Characteristics ──
        st.markdown("##### 3. Packet Length Characteristics")
        p1, p2, p3 = st.columns(3)
        with p1:
            min_len = st.number_input(
                "Min Packet Length (bytes)",
                value=_get_val("Min Packet Length", 0.0),
                help="Smallest packet observed in bytes",
                key="input_Min Packet Length"
            )
            st.session_state["manual_Min Packet Length"] = float(min_len)
            avg_size = st.number_input(
                "Average Packet Size (bytes)",
                value=_get_val("Average Packet Size", 195.0),
                help="Mean size of all packets in the flow",
                key="input_Average Packet Size"
            )
            st.session_state["manual_Average Packet Size"] = float(avg_size)
            fwd_max = st.number_input(
                "Forward Packet Length Max",
                value=_get_val("Fwd Packet Length Max", 482.0),
                help="Maximum forward packet size in bytes",
                key="input_Fwd Packet Length Max"
            )
            st.session_state["manual_Fwd Packet Length Max"] = float(fwd_max)
        with p2:
            pkt_std = st.number_input(
                "Packet Length Std Dev",
                value=_get_val("Packet Length Std", 182.0),
                help="Standard deviation of packet lengths",
                key="input_Packet Length Std"
            )
            st.session_state["manual_Packet Length Std"] = float(pkt_std)
            pkt_var = st.number_input(
                "Packet Length Variance",
                value=_get_val("Packet Length Variance", 33124.0),
                help="Variance of packet lengths across the flow",
                key="input_Packet Length Variance"
            )
            st.session_state["manual_Packet Length Variance"] = float(pkt_var)
            fwd_std = st.number_input(
                "Forward Packet Length Std Dev",
                value=_get_val("Fwd Packet Length Std", 128.5),
                help="Standard deviation of forward packet sizes",
                key="input_Fwd Packet Length Std"
            )
            st.session_state["manual_Fwd Packet Length Std"] = float(fwd_std)
        with p3:
            bwd_mean = st.number_input(
                "Backward Packet Length Mean",
                value=_get_val("Bwd Packet Length Mean", 320.0),
                help="Average backward packet size in bytes",
                key="input_Bwd Packet Length Mean"
            )
            st.session_state["manual_Bwd Packet Length Mean"] = float(bwd_mean)
            bwd_std = st.number_input(
                "Backward Packet Length Std Dev",
                value=_get_val("Bwd Packet Length Std", 240.0),
                help="Standard deviation of backward packet sizes",
                key="input_Bwd Packet Length Std"
            )
            st.session_state["manual_Bwd Packet Length Std"] = float(bwd_std)
            fwd_seg = st.number_input(
                "Average Forward Segment Size",
                value=_get_val("Avg Fwd Segment Size", 142.0),
                help="Average segment size in the forward direction",
                key="input_Avg Fwd Segment Size"
            )
            st.session_state["manual_Avg Fwd Segment Size"] = float(fwd_seg)

        # ── Group 4: Timing & Inter-Arrival Times (IAT) ──
        st.markdown("##### 4. Timing & Inter-Arrival Times")
        t1, t2, t3 = st.columns(3)
        with t1:
            flow_iat_min = st.number_input(
                "Flow IAT Min (µs)",
                value=_get_val("Flow IAT Min", 4.0),
                help="Minimum time between consecutive packets",
                key="input_Flow IAT Min"
            )
            st.session_state["manual_Flow IAT Min"] = float(flow_iat_min)
            fwd_iat_tot = st.number_input(
                "Forward IAT Total (µs)",
                value=_get_val("Fwd IAT Total", 48500.0),
                help="Total time between packets in forward direction",
                key="input_Fwd IAT Total"
            )
            st.session_state["manual_Fwd IAT Total"] = float(fwd_iat_tot)
            fwd_iat_max = st.number_input(
                "Forward IAT Max (µs)",
                value=_get_val("Fwd IAT Max", 24000.0),
                help="Maximum time between forward packets",
                key="input_Fwd IAT Max"
            )
            st.session_state["manual_Fwd IAT Max"] = float(fwd_iat_max)
        with t2:
            fwd_iat_min = st.number_input(
                "Forward IAT Min (µs)",
                value=_get_val("Fwd IAT Min", 5.0),
                help="Minimum time between forward packets",
                key="input_Fwd IAT Min"
            )
            st.session_state["manual_Fwd IAT Min"] = float(fwd_iat_min)
            fwd_iat_std = st.number_input(
                "Forward IAT Std Dev",
                value=_get_val("Fwd IAT Std", 1150.0),
                help="Standard deviation of forward packet arrival times",
                key="input_Fwd IAT Std"
            )
            st.session_state["manual_Fwd IAT Std"] = float(fwd_iat_std)
            active_mean = st.number_input(
                "Active Mean (µs)",
                value=_get_val("Active Mean", 0.0),
                help="Mean time flow was active before becoming idle",
                key="input_Active Mean"
            )
            st.session_state["manual_Active Mean"] = float(active_mean)
        with t3:
            bwd_iat_mean = st.number_input(
                "Backward IAT Mean (µs)",
                value=_get_val("Bwd IAT Mean", 8200.0),
                help="Average time between backward packets",
                key="input_Bwd IAT Mean"
            )
            st.session_state["manual_Bwd IAT Mean"] = float(bwd_iat_mean)
            bwd_iat_max = st.number_input(
                "Backward IAT Max (µs)",
                value=_get_val("Bwd IAT Max", 17500.0),
                help="Maximum time between backward packets",
                key="input_Bwd IAT Max"
            )
            st.session_state["manual_Bwd IAT Max"] = float(bwd_iat_max)
            bwd_iat_std = st.number_input(
                "Backward IAT Std Dev",
                value=_get_val("Bwd IAT Std", 2100.0),
                help="Standard deviation of backward packet arrival times",
                key="input_Bwd IAT Std"
            )
            st.session_state["manual_Bwd IAT Std"] = float(bwd_iat_std)

        # ── Group 5: Connection Windows ──
        st.markdown("##### 5. Connection Windows")
        w1, w2 = st.columns(2)
        with w1:
            win_fwd = st.number_input(
                "Initial Forward Window Bytes",
                value=_get_val("Init_Win_bytes_forward", 29200.0),
                help="Number of bytes sent in initial forward window",
                key="input_Init_Win_bytes_forward"
            )
            st.session_state["manual_Init_Win_bytes_forward"] = float(win_fwd)
        with w2:
            win_bwd = st.number_input(
                "Initial Backward Window Bytes",
                value=_get_val("Init_Win_bytes_backward", 28960.0),
                help="Number of bytes sent in initial backward window",
                key="input_Init_Win_bytes_backward"
            )
            st.session_state["manual_Init_Win_bytes_backward"] = float(win_bwd)

        st.markdown("<br>", unsafe_allow_html=True)
        m_act, _ = st.columns([1, 2])
        with m_act:
            if st.button("🔍  Check for Threats", type="primary", key="btn_run_manual", use_container_width=True):
                # Build single-record DataFrame with exact feature names
                record = {k: float(st.session_state[f"manual_{k}"]) for k in BENIGN_PRESET.keys()}
                manual_df = pd.DataFrame([record])

                with st.spinner("Checking network flow for threats…"):
                    try:
                        results = svc.run_full_pipeline(manual_df, id_col=None)
                        st.session_state["results_df"] = results
                        st.session_state["raw_df"] = manual_df
                        st.session_state["analysis_ready"] = True
                        st.session_state["selected_row_idx"] = 0
                        st.rerun()
                    except Exception as e:
                        st.error(f"Threat check error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Outcome Summary & View Results Navigation
    # Rendered at the top level when analysis has completed
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.get("analysis_ready") and st.session_state.get("results_df") is not None:
        results = st.session_state["results_df"]
        st.markdown("---")
        st.markdown("### 📋  Detection Summary")

        n_total = len(results)
        n_attack = int((results["binary_label"] == "ATTACK").sum())
        n_benign = n_total - n_attack

        if n_total == 1:
            row = results.iloc[0]
            is_attack = row["binary_label"] == "ATTACK"
            attack_cat = row.get("attack_category") or ""

            if is_attack:
                st.error("#### 🚨  Threat Detected")
                c1, c2, c3 = st.columns(3)
                c1.metric("Threat Status", "Threat Detected")
                display_cat = attack_cat if attack_cat and attack_cat != "XGBoost Error" else "Under Evaluation"
                c2.metric("Possible Attack Type", display_cat)
                c3.metric("Threat Confidence", f"{row['attack_proba']:.1%}")
                st.caption("This connection shows characteristics matching known network attack traffic.")
            else:
                st.success("#### ✅  No Threat Detected")
                c1, c2 = st.columns(2)
                c1.metric("Threat Status", "No Threat Detected")
                c2.metric("Safety Confidence", f"{(1.0 - row['attack_proba']):.1%}")
                st.caption("This network traffic pattern is normal and safe.")
        else:
            if n_attack > 0:
                st.warning(f"#### ⚠️  Analysis Complete: {n_attack:,} threat(s) detected out of {n_total:,} records.")
            else:
                st.success(f"#### ✅  Analysis Complete: All {n_total:,} records are normal and safe.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", f"{n_total:,}")
            c2.metric("No Threat Detected", f"{n_benign:,}", delta=f"{(n_benign / n_total) * 100:.1f}%")
            c3.metric("Threat Detected", f"{n_attack:,}", delta=f"{(n_attack / n_total) * 100:.1f}%", delta_color="inverse")

            if n_attack > 0 and "attack_category" in results.columns:
                top_cats = results[results["binary_label"] == "ATTACK"]["attack_category"].dropna().value_counts()
                if not top_cats.empty:
                    top_list = ", ".join([f"{cat} ({cnt})" for cat, cnt in top_cats.head(3).items()])
                    st.info(f"**Possible Attack Type(s) identified:** {top_list}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_view, _ = st.columns([1.5, 3])
        with col_view:
            if st.button("📊  View Results", type="primary", key="btn_view_results_action", use_container_width=True):
                nav_to("Results")
