"""Page 1 — Home: landing page with function cards and Start Analysis CTA."""

import streamlit as st


def render(svc, nav_to):
    # ── Hero section ──────────────────────────────────────────────────────────
    st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem;">
  <div style="font-size: 3.2rem; margin-bottom: 0.5rem;">🛡️</div>
  <h1 style="font-size: 2.2rem; font-weight: 800; color: #e2e8f0; margin: 0;">
    Network Threat Detection
  </h1>
  <p style="font-size: 1.1rem; color: #60a5fa; margin: 0.4rem 0 0.2rem; font-weight: 500;">
    Explainable AI-Based Classification of Network Intrusions
  </p>
  <p style="color: #64748b; font-size: 0.9rem; max-width: 600px; margin: 0.6rem auto 0;">
    CIC-IDS2017 Dataset &nbsp;·&nbsp; Random Forest &amp; XGBoost &nbsp;·&nbsp; SHAP Explainability
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Function cards ────────────────────────────────────────────────────────
    st.markdown("""
<div class="card-grid">
  <div class="feature-card">
    <div class="icon">📂</div>
    <h4>Upload Traffic Data</h4>
    <p>Upload a CIC-IDS2017 format CSV containing network flow records for analysis.</p>
  </div>
  <div class="feature-card">
    <div class="icon">🔍</div>
    <h4>Detect Threats</h4>
    <p>Random Forest binary classifier identifies BENIGN and ATTACK flows in real time.</p>
  </div>
  <div class="feature-card">
    <div class="icon">🎯</div>
    <h4>Classify Attacks</h4>
    <p>XGBoost multiclass classifier identifies the specific attack category across 13 types.</p>
  </div>
  <div class="feature-card">
    <div class="icon">💡</div>
    <h4>Explain Predictions</h4>
    <p>SHAP TreeExplainer shows which network features drove each detection decision.</p>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── CTA button ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([2, 1.2, 2])
    with col_c:
        if st.button("🚀  Start Analysis", type="primary", width='stretch', key="cta_start"):
            nav_to("Analyze")

    # ── System status strip ───────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    def _status(col, label, ok):
        with col:
            if ok:
                st.success(f"✅ {label}", icon=None)
            else:
                st.error(f"❌ {label}", icon=None)

    _status(col1, "Preprocessor", svc.status.preprocessor_ok)
    _status(col2, "RF Detector",  svc.status.rf_ok)
    _status(col3, "XGBoost Classifier", svc.status.xgb_ok)
    _status(col4, "Evaluation Reports",
            svc.status.binary_report_ok and svc.status.attack_report_ok)

    if not svc.status.can_predict_binary:
        st.error(
            "**Critical artifacts missing — prediction will be unavailable.** "
            "See **System Info** → **Artifacts & Setup** for details.",
        )

    # ── Two-sentence description of the two-stage pipeline ────────────────────
    st.markdown("---")
    st.markdown("""
<div style="max-width:800px; margin: 0 auto; color: #94a3b8; font-size: 0.9rem; line-height: 1.8; text-align: center;">
  <b style="color:#e2e8f0;">How it works:</b>&nbsp;
  Each uploaded network flow record is first classified as <b>BENIGN</b> or <b>ATTACK</b>
  by the Random Forest model (Stage 1).  Records predicted as ATTACK are then passed to
  the XGBoost classifier which identifies one of 13 specific attack categories (Stage 2).
  SHAP explanations are available for any individual record.
</div>
<br>
<div style="max-width:800px; margin: 0 auto; background:#1e293b; border-left: 3px solid #f59e0b;
     border-radius: 6px; padding: 0.8rem 1.2rem; font-size: 0.82rem; color: #94a3b8;">
  ⚠️ <b style="color:#fbbf24;">Disclaimer:</b> This is an undergraduate research project.
  Model outputs are probabilistic predictions and should not be treated as guaranteed
  security decisions or used as the sole basis for production network security responses.
</div>
""", unsafe_allow_html=True)

    # ── Quick links ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        if st.button("📊  View Evaluation Results", width='stretch', key="home_eval"):
            nav_to("System Info")
    with lc2:
        if st.button("📂  Upload & Analyze", width='stretch', key="home_analyze"):
            nav_to("Analyze")
    with lc3:
        if st.button("⚙️  System Information", width='stretch', key="home_sysinfo"):
            nav_to("System Info")
