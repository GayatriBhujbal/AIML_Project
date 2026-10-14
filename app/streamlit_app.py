"""
Streamlit app entry point — redesigned with horizontal top navigation.

Pages:
  Home          — Landing page with function cards and Start Analysis CTA
  Analyze       — CSV upload, validation, Run Detection
  Results       — Prediction results table, summary, download
  Explain       — SHAP explanations for a selected record
  System Info   — All technical/evaluation content (tabs)

Navigation is driven by st.session_state["page"]; no URL routing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "page": "Home",
    "raw_df": None,
    "results_df": None,
    "file_hash": None,
    "selected_row_idx": 0,
    "val_errors": [],
    "val_warnings": [],
    "analysis_ready": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Navigation helper ─────────────────────────────────────────────────────────
def nav_to(page: str):
    st.session_state["page"] = page
    st.rerun()


# ── Inference service (cached once per session) ───────────────────────────────
@st.cache_resource(show_spinner="Loading ML models…")
def get_service():
    from app.inference_service import InferenceService
    return InferenceService()






# ── Top navigation bar (real clickable Streamlit buttons) ────────────────────
PAGES = ["🏠  Home", "📂  Analyze", "📊  Results", "🔍  Explain", "⚙️  System Info"]
PAGE_KEYS = ["Home", "Analyze", "Results", "Explain", "System Info"]

cur = st.session_state["page"]

# Style the nav bar container and override button appearance inside it
st.markdown("""
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stSidebarCollapseButton"] { display: none; }

/* Nav container — brand label */
.nav-brand {
    font-size: 1.05rem; font-weight: 700; color: #60a5fa;
    white-space: nowrap; padding: 0.5rem 0.8rem 0.5rem 0;
}

/* Style ALL nav buttons uniformly */
div[data-testid="stHorizontalBlock"]:first-of-type button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.4rem 0.9rem !important;
    border: none !important;
    transition: background 0.2s !important;
    width: 100% !important;
}

/* Feature cards */
.card-grid { display: flex; gap: 1.2rem; margin: 1.5rem 0; flex-wrap: wrap; }
.feature-card {
    flex: 1; min-width: 200px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155; border-radius: 14px;
    padding: 1.4rem 1.2rem; text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.feature-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
.feature-card .icon { font-size: 2.2rem; margin-bottom: 0.6rem; }
.feature-card h4 { color: #e2e8f0; margin: 0.4rem 0 0.3rem; font-size: 1rem; }
.feature-card p { color: #94a3b8; font-size: 0.82rem; margin: 0; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

# Brand + nav buttons in one row
brand_col, *nav_cols, _ = st.columns([1.6, 1, 1, 1, 1, 1.4])
with brand_col:
    st.markdown('<p class="nav-brand">🛡️ XAI Threat Detection</p>', unsafe_allow_html=True)

for col, label, key in zip(nav_cols, PAGES, PAGE_KEYS):
    with col:
        btn_type = "primary" if key == cur else "secondary"
        if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
            nav_to(key)

st.divider()


# ── Route to page ─────────────────────────────────────────────────────────────
svc = get_service()

if cur == "Home":
    from app.pages.home import render
    render(svc, nav_to)

elif cur == "Analyze":
    from app.pages.analyze import render
    render(svc, nav_to)

elif cur == "Results":
    from app.pages.results import render
    render(svc, nav_to)

elif cur == "Explain":
    from app.pages.explain import render
    render(svc, nav_to)

elif cur == "System Info":
    from app.pages.sysinfo import render
    render(svc, nav_to)
