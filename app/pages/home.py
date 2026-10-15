"""Page 1 — Home: landing page with function cards and Check for Threats CTA."""

import streamlit as st


def render(svc, nav_to):
    # ── Hero section ──────────────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem;">
  <div style="font-size: 3.2rem; margin-bottom: 0.5rem;">🛡️</div>
  <h1 style="font-size: 2.2rem; font-weight: 800; color: #e2e8f0; margin: 0;">
    Network Threat Detection &amp; Classification
  </h1>
  <p style="font-size: 1.1rem; color: #60a5fa; margin: 0.4rem 0 0.2rem; font-weight: 500;">
    Explainable AI-Based Network Threat Detection and Classification
  </p>
  <p style="color: #94a3b8; font-size: 0.95rem; max-width: 650px; margin: 0.6rem auto 0;">
    Identify suspicious network activity and understand the results in simple language.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Function cards ────────────────────────────────────────────────────────
    st.markdown("""
<div class="card-grid">
  <div class="feature-card">
    <div class="icon">📁</div>
    <h4>Enter Details or Upload CSV</h4>
    <p>Enter network measurements manually or upload a CSV file with network flow records.</p>
  </div>
  <div class="feature-card">
    <div class="icon">🔍</div>
    <h4>Check for Threats</h4>
    <p>Analyze network traffic in real time to identify safe activity versus potential threats.</p>
  </div>
  <div class="feature-card">
    <div class="icon">🎯</div>
    <h4>Possible Attack Type</h4>
    <p>Identifies the specific cyberattack category when a malicious threat is detected.</p>
  </div>
  <div class="feature-card">
    <div class="icon">💡</div>
    <h4>Why Was This Result Given?</h4>
    <p>Clear explanations show what factors and measurements contributed to the prediction.</p>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── CTA button ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([2, 1.4, 2])
    with col_c:
        if st.button("🔍  Check for Threats", type="primary", width='stretch', key="cta_start"):
            nav_to("Analyze")

    # ── System status strip ───────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    def _status(col, label, ok):
        with col:
            if ok:
                st.success(f"**{label}**  \nReady", icon=None)
            else:
                st.error(f"**{label}**  \nUnavailable", icon=None)

    data_prep_ok = bool(svc.status.preprocessor_ok and svc.status.selected_features_ok)
    threat_det_ok = bool(svc.status.rf_ok and svc.status.can_predict_binary)
    attack_id_ok = bool(svc.status.xgb_ok)
    results_exp_ok = bool(
        svc.status.can_predict_binary and (svc.status.binary_report_ok or svc.status.attack_report_ok)
    )

    _status(col1, "Data Preparation", data_prep_ok)
    _status(col2, "Threat Detection", threat_det_ok)
    _status(col3, "Attack Identification", attack_id_ok)
    _status(col4, "Results and Explanations", results_exp_ok)

    if not svc.status.can_predict_binary:
        st.error(
            "System components are missing — threat detection is currently unavailable. "
            "See **System Information** for setup details.",
        )

    # ── Description of the two-stage pipeline in plain English ────────────────
    st.markdown("---")
    st.markdown("""
<div style="max-width:800px; margin: 0 auto; color: #94a3b8; font-size: 0.9rem; line-height: 1.8; text-align: center;">
  <b style="color:#e2e8f0;">How it works:</b>&nbsp;
  Each network flow is first evaluated to determine whether a threat is detected.
  If a threat is detected, the system identifies the <b>Possible Attack Type</b> from known attack categories.
  Detailed explanations show <b>Why Was This Result Given?</b> for every inspected flow.
</div>
<br>
<div style="max-width:800px; margin: 0 auto; background:#1e293b; border-left: 3px solid #f59e0b;
     border-radius: 6px; padding: 0.8rem 1.2rem; font-size: 0.82rem; color: #94a3b8;">
  ⚠️ <b style="color:#fbbf24;">Student Project Demonstration:</b> This application demonstrates explainable AI
  for cybersecurity education and network intrusion analysis.
</div>
""", unsafe_allow_html=True)

    # ── Quick links ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        if st.button("🔍  Check for Threats", width='stretch', key="home_analyze"):
            nav_to("Analyze")
    with lc2:
        if st.button("📊  View Results", width='stretch', key="home_results"):
            nav_to("Results")
    with lc3:
        if st.button("⚙️  System Information", width='stretch', key="home_sysinfo"):
            nav_to("System Info")
