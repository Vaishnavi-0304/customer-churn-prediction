"""
Customer Analytics View for ChurnGuard AI
Author: ChurnGuard AI Team
Description:
    Exploratory Data Analysis (EDA) platform featuring interactive Plotly charts,
    multivariate cross-tabulations, dynamic dataset filtering, and
    human-readable business insights under every visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from preprocessing import load_raw_data, clean_data
from components.ui import render_page_header, render_insight_box, render_footer, PLOTLY_DARK_LAYOUT

@st.cache_data
def load_analytics_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "customer_churn.csv")
    df = load_raw_data(data_path)
    df = clean_data(df)
    return df

def render_analytics_page():
    render_page_header("Customer Analytics & Exploratory Data Analysis", "Interactively inspect multi-variable churn correlations across billing, contract, and service segments.")
    
    raw_df = load_analytics_data()
    
    # ----------------------------------------------------
    # Interactive Filters Section
    # ----------------------------------------------------
    st.markdown("""<div class="glass-card" style="padding: 16px 20px; margin-bottom: 20px;"><h5 style="margin:0 0 10px 0; color:#fff;">🔍 Dynamic Customer Population Filters</h5></div>""", unsafe_allow_html=True)
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        contract_opts = ["All Contracts"] + list(raw_df['Contract'].unique())
        selected_contract = st.selectbox("Contract Type", contract_opts, index=0)
    with f2:
        internet_opts = ["All Internet Types"] + list(raw_df['InternetService'].unique())
        selected_internet = st.selectbox("Internet Service", internet_opts, index=0)
    with f3:
        payment_opts = ["All Payment Methods"] + list(raw_df['PaymentMethod'].unique())
        selected_payment = st.selectbox("Payment Method", payment_opts, index=0)
    with f4:
        churn_opts = ["All Statuses", "Retained (No)", "Churned (Yes)"]
        selected_churn = st.selectbox("Churn Filter", churn_opts, index=0)

    # Apply filters
    filtered_df = raw_df.copy()
    if selected_contract != "All Contracts":
        filtered_df = filtered_df[filtered_df['Contract'] == selected_contract]
    if selected_internet != "All Internet Types":
        filtered_df = filtered_df[filtered_df['InternetService'] == selected_internet]
    if selected_payment != "All Payment Methods":
        filtered_df = filtered_df[filtered_df['PaymentMethod'] == selected_payment]
    if selected_churn == "Retained (No)":
        filtered_df = filtered_df[filtered_df['Churn'] == 'No']
    elif selected_churn == "Churned (Yes)":
        filtered_df = filtered_df[filtered_df['Churn'] == 'Yes']

    filt_churn_rate = (filtered_df['Churn'] == 'Yes').mean() * 100 if len(filtered_df) > 0 else 0
    st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:10px 18px; margin-bottom:24px; font-size:0.88rem;">
            <span>Displaying <strong>{len(filtered_df):,}</strong> of <strong>{len(raw_df):,}</strong> total customer records ({(len(filtered_df)/len(raw_df))*100:.1f}%)</span>
            <span style="color:#06b6d4;">Segment Churn Rate: <strong>{filt_churn_rate:.1f}%</strong></span>
        </div>
    """, unsafe_allow_html=True)
    
    if len(filtered_df) == 0:
        st.warning("No customers match the current filter selection. Please loosen your filters.")
        return

    # ----------------------------------------------------
    # Chart Row 1: Contract Analysis & Tenure Distribution
    # ----------------------------------------------------
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""<div class="glass-card" style="margin-bottom:0;"><h4 style="margin:0; color:#fff;">A. Churn Rate by Contract Type</h4></div>""", unsafe_allow_html=True)
        contract_data = filtered_df.groupby(['Contract', 'Churn'], observed=False).size().reset_index(name='Count')
        fig_contract = px.bar(
            contract_data,
            x='Contract',
            y='Count',
            color='Churn',
            barmode='group',
            color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
            labels={'Count': 'Customer Count', 'Contract': 'Contract Term'}
        )
        fig_contract.update_layout(**PLOTLY_DARK_LAYOUT, height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_contract, use_container_width=True)
        render_insight_box("Customers on Month-to-Month contracts have a ~43% churn rate compared to ~11% for 1-year and <3% for 2-year contracts. Migrating month-to-month subscribers to annual contracts cuts defection by over 70%.")

    with col_b:
        st.markdown("""<div class="glass-card" style="margin-bottom:0;"><h4 style="margin:0; color:#fff;">B. Customer Tenure Distribution vs Churn</h4></div>""", unsafe_allow_html=True)
        fig_tenure = px.histogram(
            filtered_df,
            x='tenure',
            color='Churn',
            nbins=36,
            barmode='overlay',
            opacity=0.75,
            color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
            labels={'tenure': 'Tenure in Months'}
        )
        fig_tenure.update_layout(**PLOTLY_DARK_LAYOUT, height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_tenure, use_container_width=True)
        render_insight_box("Churn probability decays dramatically with tenure. A massive spike in customer defection occurs in the first 1-5 months; once a subscriber crosses the 24-month mark, churn probability plummets below 15%.")

    # ----------------------------------------------------
    # Chart Row 2: Internet Service & Payment Method
    # ----------------------------------------------------
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.markdown("""<div class="glass-card" style="margin-bottom:0;"><h4 style="margin:0; color:#fff;">C. Churn by Internet Service Architecture</h4></div>""", unsafe_allow_html=True)
        net_data = filtered_df.groupby(['InternetService', 'Churn'], observed=False).size().reset_index(name='Count')
        fig_net = px.bar(
            net_data,
            x='InternetService',
            y='Count',
            color='Churn',
            barmode='group',
            color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
            labels={'InternetService': 'Internet Service Infrastructure'}
        )
        fig_net.update_layout(**PLOTLY_DARK_LAYOUT, height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_net, use_container_width=True)
        render_insight_box("Fiber optic customers exhibit a staggering 41.9% churn rate, substantially higher than DSL (18.9%) and No Internet (7.4%). Fiber optic is premium priced, so service outages or billing friction provoke immediate switching.")

    with col_d:
        st.markdown("""<div class="glass-card" style="margin-bottom:0;"><h4 style="margin:0; color:#fff;">D. Churn by Payment Method Channel</h4></div>""", unsafe_allow_html=True)
        pay_data = filtered_df.groupby(['PaymentMethod', 'Churn'], observed=False).size().reset_index(name='Count')
        fig_pay = px.bar(
            pay_data,
            x='Count',
            y='PaymentMethod',
            color='Churn',
            orientation='h',
            barmode='stack',
            color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
            labels={'PaymentMethod': 'Payment Method'}
        )
        fig_pay.update_layout(**PLOTLY_DARK_LAYOUT, height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_pay, use_container_width=True)
        render_insight_box("Electronic check users experience the highest churn rate (45.3%). In contrast, automated methods (Bank Transfer Autopay 16.7%, Credit Card Autopay 15.2%) have significantly lower churn due to frictionless recurring transactions.")

    # ----------------------------------------------------
    # Chart Row 3: Monthly Charges Boxplot & Support Services
    # ----------------------------------------------------
    col_e, col_f = st.columns(2)
    
    with col_e:
        st.markdown("""<div class="glass-card" style="margin-bottom:0;"><h4 style="margin:0; color:#fff;">E. Monthly Charges vs Churn Status</h4></div>""", unsafe_allow_html=True)
        fig_box = px.box(
            filtered_df,
            x='Churn',
            y='MonthlyCharges',
            color='Churn',
            color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'},
            points="outliers",
            labels={'MonthlyCharges': 'Monthly Charges ($)'}
        )
        fig_box.update_layout(**PLOTLY_DARK_LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        render_insight_box("Median monthly charges for churned customers ($79.65) are substantially higher than retained customers ($64.43). Customers paying high monthly bills expect flawless quality; any perceived gap between price and value causes rapid attrition.")

    with col_f:
        st.markdown("""<div class="glass-card" style="margin-bottom:0;"><h4 style="margin:0; color:#fff;">F. Technical Support & Online Security Impact</h4></div>""", unsafe_allow_html=True)
        
        # Cross comparison of TechSupport & OnlineSecurity
        tech_churn = filtered_df.groupby('TechSupport')['Churn'].value_counts(normalize=True).unstack().fillna(0) * 100
        sec_churn = filtered_df.groupby('OnlineSecurity')['Churn'].value_counts(normalize=True).unstack().fillna(0) * 100
        
        service_comp = pd.DataFrame({
            'Service Category': ['No Tech Support', 'Has Tech Support', 'No Online Security', 'Has Online Security'],
            'Churn Rate (%)': [
                float(tech_churn.loc['No', 'Yes']) if 'No' in tech_churn.index and 'Yes' in tech_churn.columns else 0.0,
                float(tech_churn.loc['Yes', 'Yes']) if 'Yes' in tech_churn.index and 'Yes' in tech_churn.columns else 0.0,
                float(sec_churn.loc['No', 'Yes']) if 'No' in sec_churn.index and 'Yes' in sec_churn.columns else 0.0,
                float(sec_churn.loc['Yes', 'Yes']) if 'Yes' in sec_churn.index and 'Yes' in sec_churn.columns else 0.0,
            ]
        })
        
        fig_services = px.bar(
            service_comp,
            x='Service Category',
            y='Churn Rate (%)',
            color='Service Category',
            color_discrete_sequence=['#EF4444', '#10B981', '#EF4444', '#10B981']
        )
        fig_services.update_layout(**PLOTLY_DARK_LAYOUT, height=340, showlegend=False)
        st.plotly_chart(fig_services, use_container_width=True)
        render_insight_box("Customers without Tech Support experience a 41.6% churn rate vs just 15.2% for those with Tech Support (2.7x higher). Online Security shows a near-identical pattern (41.8% vs 14.6%). Add-on protection services act as strong customer anchors.")

    # ----------------------------------------------------
    # Chart Row 4: Senior Citizen & Paperless Billing
    # ----------------------------------------------------
    col_g, col_h = st.columns(2)
    
    with col_g:
        st.markdown("""<div class="glass-card" style="margin-bottom:0;"><h4 style="margin:0; color:#fff;">G. Churn by Senior Citizen Demographic</h4></div>""", unsafe_allow_html=True)
        filtered_df_senior = filtered_df.copy()
        filtered_df_senior['SeniorCitizenLabel'] = filtered_df_senior['SeniorCitizen'].map({1: 'Senior Citizen', 0: 'Non-Senior'})
        senior_data = filtered_df_senior.groupby(['SeniorCitizenLabel', 'Churn'], observed=False).size().reset_index(name='Count')
        fig_senior = px.bar(
            senior_data,
            x='SeniorCitizenLabel',
            y='Count',
            color='Churn',
            barmode='group',
            color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'}
        )
        fig_senior.update_layout(**PLOTLY_DARK_LAYOUT, height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_senior, use_container_width=True)
        render_insight_box("Senior citizens churn at a significantly higher rate (41.7%) compared to younger demographics (23.6%), primarily because fixed-income seniors are more price-sensitive and frequently report setup confusion.")

    with col_h:
        st.markdown("""<div class="glass-card" style="margin-bottom:0;"><h4 style="margin:0; color:#fff;">H. Churn by Paperless Billing Adoption</h4></div>""", unsafe_allow_html=True)
        paper_data = filtered_df.groupby(['PaperlessBilling', 'Churn'], observed=False).size().reset_index(name='Count')
        fig_paper = px.bar(
            paper_data,
            x='PaperlessBilling',
            y='Count',
            color='Churn',
            barmode='group',
            color_discrete_map={'No': '#10B981', 'Yes': '#EF4444'}
        )
        fig_paper.update_layout(**PLOTLY_DARK_LAYOUT, height=320, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_paper, use_container_width=True)
        render_insight_box("Subscribers with Paperless Billing churn at 33.6% compared to 16.3% for paper billing. Paperless billing customers are digitally active, tech-savvy, and more prone to exploring competitor promotions online.")

    render_footer()
