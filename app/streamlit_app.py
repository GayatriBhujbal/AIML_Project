"""
Streamlit multi-page application entry point.

Run with:
    streamlit run app/streamlit_app.py

Pages
-----
A  Overview / Dashboard
B  Predict from CSV
C  Explain a Prediction
D  Evaluation Results
E  Artifacts & System Info
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Network Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared inference service (cached once per session) ────────────────────────
@st.cache_resource(show_spinner="Loading ML models and artifacts…")
def get_service():
    from app.inference_service import InferenceService
    return InferenceService()


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ XAI Threat Detection")
    st.markdown("*Explainable AI-Based Network Threat Detection and Classification*")
    st.divider()

    page = st.radio(
        "Navigate to:",
        options=[
            "A · Overview",
            "B · Predict from CSV",
            "C · Explain a Prediction",
            "D · Evaluation Results",
            "E · Artifacts & System Info",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("CIC-IDS2017 Dataset · Python 3.x")
    st.caption("Random Forest · XGBoost · SHAP")

# ── Route to selected page ────────────────────────────────────────────────────
svc = get_service()

if page == "A · Overview":
    from app.pages.page_overview import render
    render(svc)

elif page == "B · Predict from CSV":
    from app.pages.page_predict import render
    render(svc)

elif page == "C · Explain a Prediction":
    from app.pages.page_explain import render
    render(svc)

elif page == "D · Evaluation Results":
    from app.pages.page_evaluation import render
    render(svc)

elif page == "E · Artifacts & System Info":
    from app.pages.page_artifacts import render
    render(svc)
