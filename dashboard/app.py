import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🛡️ Threat Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "User Behaviour Analytics",
        "Threat Detection",
        "Explainable AI",
        "Ground Truth Validation",
        "Global ML Evaluation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Explainable Insider Threat Detection System")

st.title(f"🛡️ {page}")

import pandas as pd
import plotly.express as px

from dashboard.real_ml_pages import (
    render_overview,
    render_user_behaviour,
    render_threat_detection,
    render_explainable_ai,
    render_global_evaluation,
)

if page == "Overview":
    render_overview()

elif page == "User Behaviour Analytics":
    render_user_behaviour()

elif page == "Threat Detection":
    render_threat_detection()

elif page == "Explainable AI":
    render_explainable_ai()

elif page == "Ground Truth Validation":
    st.info("Ground-truth validation is being integrated.")

elif page == "Global ML Evaluation":
    render_global_evaluation()