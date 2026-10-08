"""
Main Streamlit Application Controller for ChurnGuard AI
Author: ChurnGuard AI Team
Description:
    Core entrypoint managing global session state, dark SaaS styling,
    always-available unified sidebar navigation, and modular page orchestration.
"""

import streamlit as st
import sys
import os

# Set page configuration FIRST before any other streamlit commands
st.set_page_config(
    page_title="ChurnGuard AI | Customer Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup search paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
APP_DIR = os.path.join(BASE_DIR, "app")
for path in [BASE_DIR, SRC_DIR, APP_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from auth import init_db, authenticate_user
from components.ui import apply_custom_css, render_brand_header
from pages.landing import render_landing_page
from pages.dashboard import render_dashboard_page
from pages.analytics import render_analytics_page
from pages.prediction import render_prediction_page
from pages.retention import render_retention_page
from pages.ml_insights import render_ml_insights_page
from pages.profile import render_profile_page

# Initialize SQLite database on startup
init_db()

# Apply global dark SaaS CSS
apply_custom_css()

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Landing"

user = st.session_state.get("user", {})

# ----------------------------------------------------
# Always-Available Sidebar Navigation
# ----------------------------------------------------
with st.sidebar:
    render_brand_header()
    
    pages = {
        "Landing": "🏠 Home & Overview",
        "Dashboard": "📊 Executive Dashboard",
        "Customer Analytics": "📈 Customer Analytics",
        "Predict Churn": "🔮 Predict Churn",
        "Retention Insights": "💡 Retention Playbooks",
        "ML Insights": "🧠 ML Model Insights",
        "Profile": "👤 Profile & Settings" if st.session_state["authenticated"] else "👤 Sign In / Register"
    }
    
    # Determine current selection index
    current = st.session_state.get("current_page", "Landing")
    page_keys = list(pages.keys())
    default_idx = page_keys.index(current) if current in page_keys else 0
    
    selected_page = st.radio(
        "Navigation Menu",
        options=page_keys,
        format_func=lambda x: pages[x],
        index=default_idx,
        key="global_nav_radio",
        label_visibility="collapsed"
    )
    st.session_state["current_page"] = selected_page
    
    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
    
    # User session badge in sidebar bottom
    if st.session_state["authenticated"]:
        st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px 14px; margin-bottom: 12px;">
                <div style="font-size: 0.72rem; color: #34d399; font-weight:700; text-transform: uppercase; letter-spacing: 0.05em;">● AUTHENTICATED USER</div>
                <div style="font-size: 0.92rem; font-weight: 700; color: #ffffff; margin-top: 2px;">{user.get('full_name', 'Workspace User')}</div>
                <div style="font-size: 0.78rem; color: #94a3b8;">{user.get('email', '')}</div>
                <div style="font-size: 0.72rem; color: #818cf8; margin-top: 4px;">{user.get('role', 'Strategist')}</div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Sign Out", use_container_width=True, key="sidebar_logout_btn"):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.session_state["current_page"] = "Landing"
            st.rerun()
    else:
        st.markdown("""
            <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 12px; padding: 12px 14px; margin-bottom: 12px;">
                <div style="font-size: 0.72rem; color: #818cf8; font-weight:700; text-transform: uppercase; letter-spacing: 0.05em;">⚡ GUEST / REVIEWER MODE</div>
                <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 4px; line-height:1.4;">All analytics, ML diagnostics, and prediction tools are unlocked for review.</div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("⚡ Quick Sign-In as Demo User", use_container_width=True, key="sidebar_demo_login_btn"):
            demo_user, _ = authenticate_user("demo@churnguard.ai", "Admin@123")
            if demo_user:
                st.session_state["authenticated"] = True
                st.session_state["user"] = demo_user
                st.session_state["current_page"] = "Dashboard"
                st.rerun()

# ----------------------------------------------------
# Route Page Dispatch
# ----------------------------------------------------
active_page = st.session_state.get("current_page", "Landing")

if active_page == "Landing":
    render_landing_page()
elif active_page == "Dashboard":
    render_dashboard_page()
elif active_page == "Customer Analytics":
    render_analytics_page()
elif active_page == "Predict Churn":
    render_prediction_page()
elif active_page == "Retention Insights":
    render_retention_page()
elif active_page == "ML Insights":
    render_ml_insights_page()
elif active_page == "Profile":
    if st.session_state["authenticated"]:
        render_profile_page()
    else:
        render_landing_page()
else:
    render_landing_page()
