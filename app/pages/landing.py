"""
Landing Page & Authentication View for ChurnGuard AI
Author: ChurnGuard AI Team
Description:
    Renders high-converting SaaS landing page with hero banner,
    analytics previews, 3-step workflow, and SQLite-backed Login/Registration.
"""

import streamlit as st
import sys
import os

# Ensure src is in sys.path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from auth import authenticate_user, register_user
from components.ui import render_footer, apply_custom_css

def render_landing_page():
    """Renders the landing and authentication screen."""
    # Hero Section
    st.markdown("""
        <div class="hero-box">
            <div class="hero-badge">⚡ ENTERPRISE CUSTOMER RETENTION PLATFORM</div>
            <div class="hero-title">CUSTOMER CHURN PREDICTION SYSTEM</div>
            <div class="hero-tagline">Predict. Prevent. Retain.</div>
            <div class="hero-desc">
                Identify customers at risk of leaving before it's too late. Turn machine learning insights into smarter retention decisions with calibrated risk modeling and automated playbooks.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 3 Metrics preview cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-title">HISTORICAL PROFILES</div>
                <div class="kpi-value">7,043</div>
                <div class="kpi-sub">IBM Telco Real-World Dataset</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="kpi-card retained">
                <div class="kpi-title">MODEL ROC-AUC</div>
                <div class="kpi-value">84.8%</div>
                <div class="kpi-sub">Cross-Validated XGBoost Engine</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="kpi-card churn">
                <div class="kpi-title">CHURN REDUCTION</div>
                <div class="kpi-value">~38%</div>
                <div class="kpi-sub">Targeted Retention Impact</div>
            </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
            <div class="kpi-card">
                <div class="kpi-title">EVALUATION METRICS</div>
                <div class="kpi-value">5+ Models</div>
                <div class="kpi-sub">GridSearchCV Hyper-Tuning</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    # Auth Container
    col_left, col_right = st.columns([1.1, 0.9])
    
    with col_left:
        st.markdown("""
            <div class="glass-card">
                <h3 style="color:#ffffff; margin-top:0;">How It Works</h3>
                <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 16px;">
                    <div class="step-card">
                        <div class="step-num">01</div>
                        <div class="step-title">Analyze Customer Data</div>
                        <div class="step-desc">Ingest multidimensional telemetry including contract tenure, usage tiers, internet services, payment methods, and billing records.</div>
                    </div>
                    <div class="step-card">
                        <div class="step-num">02</div>
                        <div class="step-title">Predict Churn Risk</div>
                        <div class="step-desc">Advanced ensemble algorithms compute real-time churn probability and stratify customers into Low, Medium, and High Risk tiers.</div>
                    </div>
                    <div class="step-card">
                        <div class="step-num">03</div>
                        <div class="step-title">Take Retention Action</div>
                        <div class="step-desc">Generate tailored, rule-based retention interventions, contract incentives, and technical support bundles to preserve ARR.</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        st.markdown("""
            <div class="glass-card">
                <h3 style="color:#ffffff; margin-top:0; margin-bottom: 4px;">Platform Access</h3>
                <p style="color:#94a3b8; font-size:0.85rem; margin-bottom: 16px;">Sign in with your enterprise credentials or register a new workspace account.</p>
            </div>
        """, unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["🔑 Sign In", "📝 Create Account"])
        
        with tab_login:
            login_email = st.text_input("Work Email", value="", placeholder="name@company.com", key="in_login_email")
            login_pwd = st.text_input("Password", value="", type="password", placeholder="Enter your password", key="in_login_pwd")
            
            if st.button("Sign In to ChurnGuard", use_container_width=True, key="btn_do_login"):
                if not login_email or not login_pwd:
                    st.error("Please enter both email and password.")
                else:
                    user, msg = authenticate_user(login_email, login_pwd)
                    if user:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user
                        st.session_state["current_page"] = "Dashboard"
                        st.success("Authentication successful! Loading workspace...")
                        st.rerun()
                    else:
                        st.error(msg)
                        
        with tab_register:
            reg_name = st.text_input("Full Name", placeholder="e.g. Sarah Connor", key="reg_name")
            reg_email = st.text_input("Email Address", placeholder="name@company.com", key="reg_email")
            reg_org = st.text_input("Organization", placeholder="e.g. Metro Telecom", key="reg_org")
            reg_role = st.text_input("Role", placeholder="e.g. Retention Director", key="reg_role")
            reg_p1 = st.text_input("Password", type="password", key="reg_p1")
            reg_p2 = st.text_input("Confirm Password", type="password", key="reg_p2")
            
            if st.button("Create Account", use_container_width=True, key="btn_do_register"):
                success, msg = register_user(reg_name, reg_email, reg_org, reg_role, reg_p1, reg_p2)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

    render_footer()

if __name__ == "__main__":
    apply_custom_css()
    render_landing_page()
