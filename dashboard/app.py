import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

SHIELD = "\U0001F6E1\uFE0F"

st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon=SHIELD,
    layout="wide",
    initial_sidebar_state="expanded"
)

from dashboard.real_ml_pages import (
    render_overview,
    render_user_behaviour,
    render_threat_detection,
    render_explainable_ai,
    render_global_evaluation,
)

from dashboard.evaluation_pages import (
    render_ground_truth_validation,
    render_attack_scenarios,
    render_model_reliability,
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title(
    f"{SHIELD} Threat Detection"
)

st.sidebar.caption(
    "Explainable Insider Threat Detection "
    "using User Behavior Analytics"
)

st.sidebar.markdown("---")

pages = {
    "Overview":
        render_overview,

    "User Behaviour Analytics":
        render_user_behaviour,

    "Threat Detection":
        render_threat_detection,

    "Explainable AI":
        render_explainable_ai,

    "Ground Truth Validation":
        render_ground_truth_validation,

    "Global ML Evaluation":
        render_global_evaluation,

    "Attack Scenario Analysis":
        render_attack_scenarios,

    "Model Reliability & Explainability":
        render_model_reliability
}

page = st.sidebar.radio(
    "Navigation",
    list(pages.keys())
)

st.sidebar.markdown("---")

st.sidebar.caption("Dataset: CERT r4.2")

st.sidebar.caption("Primary Model: EBM")

st.sidebar.caption(
    "Comparative Model: XGBoost"
)

st.sidebar.caption(
    "Historical Research Prototype"
)


# ==========================================
# PAGE ROUTING
# ==========================================

st.title(
    f"{SHIELD} {page}"
)

pages[page]()