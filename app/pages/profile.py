"""
User Profile and Account Settings View for ChurnGuard AI
Author: ChurnGuard AI Team
Description:
    Displays current user credentials, organizational role, workspace diagnostics,
    allows live profile updates stored in SQLite, and provides session logout.
"""

import streamlit as st
import sys
import os

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from auth import update_user_profile, get_user_by_id
from components.ui import render_page_header, render_footer, apply_custom_css

def render_profile_page():
    render_page_header("User Profile & Workspace Settings", "Manage enterprise credentials, organizational identity, and platform preferences.")
    
    user = st.session_state.get("user", {})
    if not user:
        st.warning("No active session found. Please sign in.")
        return

    # Profile Summary Banner
    st.markdown(f"""
        <div class="glass-card" style="display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
            <div style="width:68px; height:68px; border-radius:50%; background:linear-gradient(135deg, #6366f1, #06b6d4); display:flex; align-items:center; justify-content:center; font-size:28px; font-weight:700; color:#ffffff; box-shadow:0 4px 20px rgba(99, 102, 241, 0.4);">
                {user.get('full_name', 'U')[0].upper()}
            </div>
            <div>
                <h3 style="margin:0; color:#ffffff; font-size:1.4rem;">{user.get('full_name', 'Workspace User')}</h3>
                <p style="margin:2px 0 0 0; color:#94a3b8; font-size:0.88rem;">{user.get('role', 'Strategist')} • {user.get('organization', 'Enterprise')}</p>
                <p style="margin:2px 0 0 0; color:#64748b; font-size:0.78rem;">Email: {user.get('email', '')} • Member Since: {user.get('created_at', '2026')}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_edit, col_diag = st.columns([1.2, 0.8])
    
    with col_edit:
        st.markdown("""<div class="glass-card"><h4 style="margin-top:0; color:#fff;">Edit Profile Information</h4></div>""", unsafe_allow_html=True)
        
        with st.form("edit_profile_form"):
            new_name = st.text_input("Full Name", value=user.get("full_name", ""))
            new_org = st.text_input("Organization", value=user.get("organization", ""))
            new_role = st.text_input("Role / Job Title", value=user.get("role", ""))
            
            st.text_input("Email (Primary Identifier)", value=user.get("email", ""), disabled=True)
            
            st.write("")
            save_btn = st.form_submit_button("Save Profile Changes", use_container_width=True)
            
        if save_btn:
            ok, msg = update_user_profile(user["id"], new_name, new_org, new_role)
            if ok:
                updated_user = get_user_by_id(user["id"])
                st.session_state["user"] = updated_user
                st.success("Profile updated successfully!")
                st.rerun()
            else:
                st.error(msg)

    with col_diag:
        st.markdown("""
            <div class="glass-card">
                <h4 style="margin-top:0; color:#fff;">System & Security Diagnostics</h4>
                <div style="display:flex; flex-direction:column; gap:10px; font-size:0.85rem; color:#cbd5e1;">
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;">
                        <span>Authentication Engine:</span>
                        <strong style="color:#10b981;">PBKDF2-HMAC-SHA256</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;">
                        <span>Database Provider:</span>
                        <strong style="color:#ffffff;">SQLite3 WAL</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;">
                        <span>Inference Model:</span>
                        <strong style="color:#818cf8;">XGBoost (Tuned)</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;">
                        <span>Model Holdout ROC-AUC:</span>
                        <strong style="color:#06b6d4;">84.79%</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span>Training Corpus:</span>
                        <strong style="color:#ffffff;">7,043 Records</strong>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        if st.button("🚪 Logout of Account", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            st.session_state["current_page"] = "Landing"
            st.rerun()

    render_footer()

if __name__ == "__main__":
    apply_custom_css()
    if not st.session_state.get("authenticated", False):
        st.warning("🔒 Authentication Required: Please sign in on the main portal to access this page.")
        st.stop()
    render_profile_page()
