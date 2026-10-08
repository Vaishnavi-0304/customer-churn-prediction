"""
Customer Churn Prediction View for ChurnGuard AI
Author: ChurnGuard AI Team
Description:
    Interactive prediction workspace with grouped inputs (Profile, Services, Billing),
    instant demo profile presets, probability gauge meter, risk tier classification
    (LOW, MEDIUM, HIGH), customer-specific explainability drivers, and retention guidance.
"""

import streamlit as st
import sys
import os

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from predict import ChurnPredictor
from components.ui import render_page_header, create_risk_meter_chart, render_footer, apply_custom_css

@st.cache_resource
def get_predictor():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    return ChurnPredictor(models_dir)

def load_preset(preset_type: str):
    """Loads realistic presets into session state."""
    if preset_type == "high_risk":
        st.session_state["p_gender"] = "Female"
        st.session_state["p_senior"] = "Yes"
        st.session_state["p_partner"] = "No"
        st.session_state["p_dependents"] = "No"
        st.session_state["p_tenure"] = 3
        st.session_state["p_phone"] = "Yes"
        st.session_state["p_multiple"] = "Yes"
        st.session_state["p_internet"] = "Fiber optic"
        st.session_state["p_security"] = "No"
        st.session_state["p_backup"] = "No"
        st.session_state["p_protection"] = "No"
        st.session_state["p_tech"] = "No"
        st.session_state["p_tv"] = "Yes"
        st.session_state["p_movies"] = "Yes"
        st.session_state["p_contract"] = "Month-to-month"
        st.session_state["p_paperless"] = "Yes"
        st.session_state["p_payment"] = "Electronic check"
        st.session_state["p_monthly"] = 98.50
        st.session_state["p_total"] = 295.50
    elif preset_type == "loyal":
        st.session_state["p_gender"] = "Male"
        st.session_state["p_senior"] = "No"
        st.session_state["p_partner"] = "Yes"
        st.session_state["p_dependents"] = "Yes"
        st.session_state["p_tenure"] = 62
        st.session_state["p_phone"] = "Yes"
        st.session_state["p_multiple"] = "Yes"
        st.session_state["p_internet"] = "DSL"
        st.session_state["p_security"] = "Yes"
        st.session_state["p_backup"] = "Yes"
        st.session_state["p_protection"] = "Yes"
        st.session_state["p_tech"] = "Yes"
        st.session_state["p_tv"] = "No"
        st.session_state["p_movies"] = "No"
        st.session_state["p_contract"] = "Two year"
        st.session_state["p_paperless"] = "No"
        st.session_state["p_payment"] = "Bank transfer (automatic)"
        st.session_state["p_monthly"] = 58.20
        st.session_state["p_total"] = 3608.40
    elif preset_type == "medium_risk":
        st.session_state["p_gender"] = "Female"
        st.session_state["p_senior"] = "No"
        st.session_state["p_partner"] = "Yes"
        st.session_state["p_dependents"] = "No"
        st.session_state["p_tenure"] = 16
        st.session_state["p_phone"] = "Yes"
        st.session_state["p_multiple"] = "No"
        st.session_state["p_internet"] = "Fiber optic"
        st.session_state["p_security"] = "Yes"
        st.session_state["p_backup"] = "No"
        st.session_state["p_protection"] = "No"
        st.session_state["p_tech"] = "No"
        st.session_state["p_tv"] = "No"
        st.session_state["p_movies"] = "No"
        st.session_state["p_contract"] = "One year"
        st.session_state["p_paperless"] = "Yes"
        st.session_state["p_payment"] = "Credit card (automatic)"
        st.session_state["p_monthly"] = 74.50
        st.session_state["p_total"] = 1192.00

def render_prediction_page():
    render_page_header("Predictive Churn Risk Engine", "Simulate individual subscriber profiles to generate instant ML churn probability, risk tiering, and actionable mitigation playbooks.")
    
    predictor = get_predictor()
    
    # Preset Toolbar
    st.markdown("""<div class="glass-card" style="padding: 14px 20px; margin-bottom: 20px;"><div style="display:flex; justify-content:space-between; align-items:center;"><strong style="color:#ffffff;">Quick Test Presets:</strong><span style="font-size:0.8rem; color:#94a3b8;">Click to load real-world customer profiles instantly</span></div></div>""", unsafe_allow_html=True)
    
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🔴 Load High-Risk Customer Example", use_container_width=True):
            load_preset("high_risk")
            st.rerun()
    with b2:
        if st.button("🟡 Load Moderate-Risk Customer Example", use_container_width=True):
            load_preset("medium_risk")
            st.rerun()
    with b3:
        if st.button("🟢 Load Loyal Retained Customer Example", use_container_width=True):
            load_preset("loyal")
            st.rerun()

    # Form Container
    st.markdown("""<div class="glass-card"><h4 style="margin-top:0; color:#fff;">Customer Attributes</h4></div>""", unsafe_allow_html=True)
    
    with st.form("churn_prediction_form"):
        # SECTION 1: CUSTOMER PROFILE
        st.markdown("""<h5 style="color:#6366f1; margin-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;">SECTION 1: CUSTOMER PROFILE</h5>""", unsafe_allow_html=True)
        col1_1, col1_2, col1_3, col1_4, col1_5 = st.columns(5)
        with col1_1:
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if st.session_state.get("p_gender", "Female") == "Female" else 1)
        with col1_2:
            senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=1 if st.session_state.get("p_senior", "No") == "Yes" else 0)
        with col1_3:
            partner = st.selectbox("Partner", ["No", "Yes"], index=1 if st.session_state.get("p_partner", "No") == "Yes" else 0)
        with col1_4:
            dependents = st.selectbox("Dependents", ["No", "Yes"], index=1 if st.session_state.get("p_dependents", "No") == "Yes" else 0)
        with col1_5:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=int(st.session_state.get("p_tenure", 12)), step=1)

        # SECTION 2: SERVICES
        st.markdown("""<h5 style="color:#6366f1; margin-top:20px; margin-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;">SECTION 2: SERVICES</h5>""", unsafe_allow_html=True)
        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if st.session_state.get("p_phone", "Yes") == "Yes" else 1)
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=1 if st.session_state.get("p_multiple", "No") == "Yes" else (2 if st.session_state.get("p_multiple", "No") == "No phone service" else 0))
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0 if st.session_state.get("p_internet", "Fiber optic") == "Fiber optic" else (1 if st.session_state.get("p_internet", "Fiber optic") == "DSL" else 2))
        with col2_2:
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=1 if st.session_state.get("p_security", "No") == "Yes" else (2 if st.session_state.get("p_security", "No") == "No internet service" else 0))
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=1 if st.session_state.get("p_backup", "No") == "Yes" else (2 if st.session_state.get("p_backup", "No") == "No internet service" else 0))
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=1 if st.session_state.get("p_protection", "No") == "Yes" else (2 if st.session_state.get("p_protection", "No") == "No internet service" else 0))
        with col2_3:
            tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=1 if st.session_state.get("p_tech", "No") == "Yes" else (2 if st.session_state.get("p_tech", "No") == "No internet service" else 0))
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=1 if st.session_state.get("p_tv", "No") == "Yes" else (2 if st.session_state.get("p_tv", "No") == "No internet service" else 0))
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=1 if st.session_state.get("p_movies", "No") == "Yes" else (2 if st.session_state.get("p_movies", "No") == "No internet service" else 0))

        # SECTION 3: ACCOUNT & BILLING
        st.markdown("""<h5 style="color:#6366f1; margin-top:20px; margin-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:6px;">SECTION 3: ACCOUNT & BILLING</h5>""", unsafe_allow_html=True)
        col3_1, col3_2, col3_3, col3_4, col3_5 = st.columns(5)
        with col3_1:
            contract_idx = 0
            if st.session_state.get("p_contract") == "One year":
                contract_idx = 1
            elif st.session_state.get("p_contract") == "Two year":
                contract_idx = 2
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=contract_idx)
        with col3_2:
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if st.session_state.get("p_paperless", "Yes") == "Yes" else 1)
        with col3_3:
            pay_methods = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            cur_pay = st.session_state.get("p_payment", "Electronic check")
            pay_idx = pay_methods.index(cur_pay) if cur_pay in pay_methods else 0
            payment = st.selectbox("Payment Method", pay_methods, index=pay_idx)
        with col3_4:
            monthly = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=float(st.session_state.get("p_monthly", 70.0)), step=1.0)
        with col3_5:
            # Auto-calculate reasonable total charges if not manually specified
            default_total = float(st.session_state.get("p_total", max(monthly, tenure * monthly)))
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=default_total, step=10.0)

        st.write("")
        submit_btn = st.form_submit_button("🚀 Analyze Churn Risk", use_container_width=True)

    # Prediction Execution & Results Card
    if submit_btn:
        customer_payload = {
            'gender': gender,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': protection,
            'TechSupport': tech,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total
        }
        
        result = predictor.predict_single(customer_payload)
        st.session_state["last_prediction_result"] = result

    if "last_prediction_result" in st.session_state:
        res = st.session_state["last_prediction_result"]
        
        st.markdown(f"""
            <div class="glass-card" style="border: 2px solid {res['risk_color']}; box-shadow: 0 0 35px rgba(99, 102, 241, 0.25); margin-top:28px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:16px;">
                    <div>
                        <span style="font-size:0.8rem; letter-spacing:0.08em; color:#94a3b8; text-transform:uppercase;">CUSTOMER RISK ASSESSMENT</span>
                        <h2 style="margin:4px 0 0 0; color:#ffffff; font-size:1.8rem;">
                            Prediction: <span style="color:{res['risk_color']}">{res['prediction_label']}</span>
                        </h2>
                    </div>
                    <div>
                        <span class="risk-badge {res['risk_badge']}" style="font-size:1rem; padding:8px 18px;">
                            {res['risk_level']}
                        </span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Risk Meter & Probability Breakdown
        r_col1, r_col2 = st.columns([1, 1.3])
        
        with r_col1:
            st.markdown("""<div class="glass-card" style="margin-bottom:0;"><h4 style="margin:0; color:#fff;">Calibrated Churn Probability</h4></div>""", unsafe_allow_html=True)
            gauge_fig = create_risk_meter_chart(res['churn_probability_pct'], res['risk_level'], res['risk_color'])
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            st.markdown(f"""
                <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; text-align:center; margin-top:4px;">
                    <div style="background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.25); border-radius:10px; padding:10px;">
                        <div style="font-size:0.75rem; color:#f87171;">CHURN PROBABILITY</div>
                        <div style="font-size:1.35rem; font-weight:700; color:#ffffff;">{res['churn_probability_pct']}%</div>
                    </div>
                    <div style="background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.25); border-radius:10px; padding:10px;">
                        <div style="font-size:0.75rem; color:#34d399;">RETENTION PROBABILITY</div>
                        <div style="font-size:1.35rem; font-weight:700; color:#ffffff;">{res['stay_probability_pct']}%</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with r_col2:
            st.markdown("""<div class="glass-card" style="margin-bottom:0;"><h4 style="margin:0; color:#fff;">Why This Customer May Churn (Explainability)</h4></div>""", unsafe_allow_html=True)
            
            if res['risk_factors']:
                st.markdown("""<strong style="color:#f87171; font-size:0.9rem;">⚠️ Primary Risk Drivers:</strong>""", unsafe_allow_html=True)
                for rf in res['risk_factors']:
                    st.markdown(f"""
                        <div style="background:rgba(239, 68, 68, 0.08); border-left:3px solid #ef4444; border-radius:0 8px 8px 0; padding:8px 12px; margin-top:6px; margin-bottom:8px;">
                            <div style="color:#ffffff; font-weight:600; font-size:0.87rem;">{rf['factor']} <span style="font-weight:400; color:#fca5a5; font-size:0.75rem;">({rf['impact']})</span></div>
                            <div style="color:#cbd5e1; font-size:0.8rem; margin-top:2px;">{rf['detail']}</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No critical risk drivers identified for this subscriber.")
                
            if res['positive_factors']:
                st.markdown("""<strong style="color:#34d399; font-size:0.9rem; margin-top:10px; display:inline-block;">🛡️ Positive Retention Anchors:</strong>""", unsafe_allow_html=True)
                for pf in res['positive_factors']:
                    st.markdown(f"""
                        <div style="background:rgba(16, 185, 129, 0.08); border-left:3px solid #10b981; border-radius:0 8px 8px 0; padding:8px 12px; margin-top:6px; margin-bottom:8px;">
                            <div style="color:#ffffff; font-weight:600; font-size:0.87rem;">{pf['factor']} <span style="font-weight:400; color:#86efac; font-size:0.75rem;">({pf['impact']})</span></div>
                            <div style="color:#cbd5e1; font-size:0.8rem; margin-top:2px;">{pf['detail']}</div>
                        </div>
                    """, unsafe_allow_html=True)

        # Actionable Retention Playbook Row
        st.markdown("""
            <div class="glass-card" style="margin-top:20px;">
                <h4 style="margin-top:0; color:#ffffff;">🎯 Recommended Retention Actions</h4>
                <p style="color:#94a3b8; font-size:0.85rem; margin-bottom:12px;">Automated business recommendations tailored to this customer's profile:</p>
            </div>
        """, unsafe_allow_html=True)
        
        r_cols = st.columns(len(res['retention_recommendations']))
        for i, (col, action) in enumerate(zip(r_cols, res['retention_recommendations'])):
            with col:
                st.markdown(f"""
                    <div style="background:rgba(99, 102, 241, 0.08); border:1px solid rgba(99, 102, 241, 0.2); border-radius:12px; padding:14px; height:100%;">
                        <div style="color:#818cf8; font-weight:700; font-size:0.8rem; margin-bottom:6px;">ACTION #{i+1}</div>
                        <div style="color:#e2e8f0; font-size:0.85rem; line-height:1.4;">{action}</div>
                    </div>
                """, unsafe_allow_html=True)

    render_footer()

if __name__ == "__main__":
    apply_custom_css()
    render_prediction_page()
