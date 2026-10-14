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


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit sidebar toggle and top-right menu */
[data-testid="stSidebarCollapseButton"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Top navigation bar */
.topnav-wrapper {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    padding: 0.6rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.topnav-brand {
    font-size: 1.1rem;
    font-weight: 700;
    color: #60a5fa;
    margin-right: 1.5rem;
    white-space: nowrap;
    letter-spacing: 0.02em;
}
.topnav-btn {
    background: transparent;
    border: none;
    color: #94a3b8;
    padding: 0.45rem 1.1rem;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}
.topnav-btn:hover { background: rgba(255,255,255,0.08); color: #e2e8f0; }
.topnav-btn.active {
    background: #2563eb;
    color: #fff;
    box-shadow: 0 2px 8px rgba(37,99,235,0.4);
}

/* Card grid */
.card-grid { display: flex; gap: 1.2rem; margin: 1.5rem 0; flex-wrap: wrap; }
.feature-card {
    flex: 1; min-width: 200px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.feature-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
.feature-card .icon { font-size: 2.2rem; margin-bottom: 0.6rem; }
.feature-card h4 { color: #e2e8f0; margin: 0.4rem 0 0.3rem; font-size: 1rem; }
.feature-card p { color: #94a3b8; font-size: 0.82rem; margin: 0; line-height: 1.4; }

/* CTA button */
.cta-wrap { text-align: center; margin: 2rem 0 1rem; }

/* Status badge */
.badge-attack {
    background: #fef2f2; color: #dc2626;
    padding: 2px 10px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600;
}
.badge-benign {
    background: #f0fdf4; color: #16a34a;
    padding: 2px 10px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600;
}

/* Metric tile */
.metric-tile {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 1rem 1.2rem;
    text-align: center;
}
.metric-tile .val { font-size: 2rem; font-weight: 700; color: #60a5fa; }
.metric-tile .lbl { font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ── Top navigation bar ────────────────────────────────────────────────────────
PAGES = ["Home", "Analyze", "Results", "Explain", "System Info"]
PAGE_ICONS = {"Home": "🏠", "Analyze": "📂", "Results": "📊", "Explain": "🔍", "System Info": "⚙️"}

cur = st.session_state["page"]

# Build nav HTML + use Streamlit columns for click targets
nav_html = f"""
<div class="topnav-wrapper">
  <span class="topnav-brand">🛡️ XAI Threat Detection</span>
"""
for p in PAGES:
    active = "active" if p == cur else ""
    nav_html += f'<span class="topnav-btn {active}">{PAGE_ICONS[p]} {p}</span>'
nav_html += "</div>"
st.markdown(nav_html, unsafe_allow_html=True)

# Invisible Streamlit buttons for actual navigation (overlaid via columns)
nav_cols = st.columns(len(PAGES) + 2)  # +2 for brand spacer
for i, page in enumerate(PAGES):
    with nav_cols[i + 1]:
        if st.button(page, key=f"nav_{page}", width='stretch',
                     type="primary" if page == cur else "secondary"):
            nav_to(page)

st.markdown("<style>[data-testid='stHorizontalBlock']:nth-of-type(1) button { opacity: 0; height: 0; padding: 0; margin: 0; border: none; }</style>", unsafe_allow_html=True)

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
