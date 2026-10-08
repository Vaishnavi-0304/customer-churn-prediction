"""
Retention Insights and Business Strategy View for ChurnGuard AI
Author: ChurnGuard AI Team
Description:
    Translates empirical ML models and behavioral drivers into executive retention playbooks,
    risk-tiered mitigation matrices, and an interactive Financial ROI calculator.
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

from components.ui import render_page_header, render_insight_box, render_footer, PLOTLY_DARK_LAYOUT, apply_custom_css

def render_retention_page():
    render_page_header("Customer Retention Playbooks & Strategy", "Operationalize predictive churn scores into proactive customer interventions and financial ROI preservation.")
    
    # ----------------------------------------------------
    # Risk-Tier Strategy Playbooks
    # ----------------------------------------------------
    st.markdown("""<div class="glass-card"><h4 style="margin-top:0; color:#fff;">🛡️ Risk-Tier Retention Framework</h4><p style="color:#94a3b8; font-size:0.85rem; margin-bottom:0;">Automated triage protocols based on model-derived churn probability scores:</p></div>""", unsafe_allow_html=True)
    
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
            <div class="glass-card" style="border-top: 3px solid #ef4444; height: 100%;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <strong style="color:#f87171; font-size:1.05rem;">HIGH RISK TIER</strong>
                    <span class="risk-badge high" style="padding:4px 10px; font-size:0.75rem;">65% – 100%</span>
                </div>
                <p style="color:#94a3b8; font-size:0.82rem; margin-bottom:12px;">Immediate defection danger within 30-60 days. Requires proactive outreach.</p>
                <div style="display:flex; flex-direction:column; gap:8px;">
                    <div style="background:rgba(239, 68, 68, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        🎯 <strong>Contract Migration Discount:</strong> Offer a 15–20% discount voucher conditioned on upgrading to a 12-month contract.
                    </div>
                    <div style="background:rgba(239, 68, 68, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        📞 <strong>Direct Retention Desk Call:</strong> Trigger proactive outreach from customer success within 48 hours.
                    </div>
                    <div style="background:rgba(239, 68, 68, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        🛡️ <strong>Tech Support Package:</strong> Include 6 months of complimentary Tech Support & Cyber Security.
                    </div>
                    <div style="background:rgba(239, 68, 68, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        💳 <strong>Auto-Pay Credit:</strong> $5/month statement credit upon enrolling in automatic bank/card billing.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
            <div class="glass-card" style="border-top: 3px solid #f59e0b; height: 100%;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <strong style="color:#fbbf24; font-size:1.05rem;">MEDIUM RISK TIER</strong>
                    <span class="risk-badge medium" style="padding:4px 10px; font-size:0.75rem;">35% – 65%</span>
                </div>
                <p style="color:#94a3b8; font-size:0.82rem; margin-bottom:12px;">Emerging dissatisfaction or competitor cross-shopping behavior.</p>
                <div style="display:flex; flex-direction:column; gap:8px;">
                    <div style="background:rgba(245, 158, 11, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        🎁 <strong>Surprise Loyalty Perks:</strong> Grant complimentary bandwidth speed boost or streaming credits.
                    </div>
                    <div style="background:rgba(245, 158, 11, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        📊 <strong>Plan Right-Sizing Audit:</strong> Proactively identify underutilized features and optimize their bill.
                    </div>
                    <div style="background:rgba(245, 158, 11, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        💬 <strong>Pulse Engagement Survey:</strong> Send targeted 1-click in-app satisfaction feedback questions.
                    </div>
                    <div style="background:rgba(245, 158, 11, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        🔔 <strong>Value Showcase Digests:</strong> Send monthly summaries of data saved, speed reliability, and uptime.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
            <div class="glass-card" style="border-top: 3px solid #10b981; height: 100%;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <strong style="color:#34d399; font-size:1.05rem;">LOW RISK TIER</strong>
                    <span class="risk-badge low" style="padding:4px 10px; font-size:0.75rem;">0% – 35%</span>
                </div>
                <p style="color:#94a3b8; font-size:0.82rem; margin-bottom:12px;">High satisfaction and deep service integration. Focus on advocacy & upsell.</p>
                <div style="display:flex; flex-direction:column; gap:8px;">
                    <div style="background:rgba(16, 185, 129, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        🤝 <strong>Advocacy & Referral Rewards:</strong> Offer double referral credits ($50 for customer and peer).
                    </div>
                    <div style="background:rgba(16, 185, 129, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        ⭐ <strong>Milestone Anniversary Gifts:</strong> Celebrate 1, 2, and 5-year subscriber anniversaries with gifts.
                    </div>
                    <div style="background:rgba(16, 185, 129, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        🚀 <strong>Beta Program Previews:</strong> Early access to next-gen hardware and streaming bundle releases.
                    </div>
                    <div style="background:rgba(16, 185, 129, 0.08); padding:8px 10px; border-radius:8px; font-size:0.83rem; color:#cbd5e1;">
                        📦 <strong>Cross-Sell Bundling:</strong> Introduce multi-device protection or home smart hub accessories.
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    # ----------------------------------------------------
    # Root Cause Matrix
    # ----------------------------------------------------
    st.markdown("""
        <div class="glass-card">
            <h4 style="margin-top:0; color:#fff;">🚨 Root Cause Matrix: Top Churn Drivers vs Mitigation Blueprint</h4>
            <p style="color:#94a3b8; font-size:0.85rem; margin-bottom:16px;">Direct mapping between data-validated churn triggers and high-impact corporate responses:</p>
            <div style="overflow-x:auto;">
                <table style="width:100%; border-collapse:collapse; font-size:0.88rem; color:#cbd5e1;">
                    <thead>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.1); text-align:left; color:#94a3b8;">
                            <th style="padding:10px 12px;">Churn Driver</th>
                            <th style="padding:10px 12px;">Cohort Churn Rate</th>
                            <th style="padding:10px 12px;">Underlying Behavioral Cause</th>
                            <th style="padding:10px 12px;">Recommended Corrective Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:10px 12px; font-weight:600; color:#ffffff;">Month-to-Month Contract</td>
                            <td style="padding:10px 12px; color:#f87171; font-weight:700;">42.7%</td>
                            <td style="padding:10px 12px;">Zero switching friction; exposed to competitor marketing cycle.</td>
                            <td style="padding:10px 12px;">Implement 1-Year lock-in incentive with 2 free billing cycles or free router upgrade.</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:10px 12px; font-weight:600; color:#ffffff;">Electronic Check Payment</td>
                            <td style="padding:10px 12px; color:#f87171; font-weight:700;">45.3%</td>
                            <td style="padding:10px 12px;">Active monthly manual payment prompts price reassessment & friction.</td>
                            <td style="padding:10px 12px;">Deploy targeted in-app wizard promoting ACH/Credit Card Autopay with immediate $10 credit.</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:10px 12px; font-weight:600; color:#ffffff;">Fiber Optic without Tech Support</td>
                            <td style="padding:10px 12px; color:#f87171; font-weight:700;">41.9%</td>
                            <td style="padding:10px 12px;">High price sensitivity ($70-110/mo) without troubleshooting assistance.</td>
                            <td style="padding:10px 12px;">Bundle 24/7 dedicated fiber technical support directly into premium broadband tiers.</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:10px 12px; font-weight:600; color:#ffffff;">Tenure Under 12 Months</td>
                            <td style="padding:10px 12px; color:#f87171; font-weight:700;">47.4%</td>
                            <td style="padding:10px 12px;">Poor onboarding experience; initial expectations mismatch.</td>
                            <td style="padding:10px 12px;">Establish a 90-day white-glove onboarding nurture program with dedicated welcome reps.</td>
                        </tr>
                        <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                            <td style="padding:10px 12px; font-weight:600; color:#ffffff;">Monthly Charges > $85</td>
                            <td style="padding:10px 12px; color:#f87171; font-weight:700;">38.5%</td>
                            <td style="padding:10px 12px;">Bill shock; perceived pricing creep over promotional period.</td>
                            <td style="padding:10px 12px;">Transparent tiered discounting; replace expired promotions with family bundle discounts.</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    # ----------------------------------------------------
    # Financial Impact & Retention ROI Calculator
    # ----------------------------------------------------
    st.markdown("""<div class="glass-card"><h4 style="margin-top:0; color:#fff;">💰 Financial Impact & Retention ROI Calculator</h4><p style="color:#94a3b8; font-size:0.85rem; margin-bottom:12px;">Model the monetary annual recurring revenue (ARR) preserved by deploying targeted ML retention playbooks:</p></div>""", unsafe_allow_html=True)
    
    calc_c1, calc_c2, calc_c3 = st.columns(3)
    with calc_c1:
        at_risk_pool = st.slider("Targeted At-Risk Subscribers", min_value=100, max_value=5000, value=1200, step=50)
    with calc_c2:
        avg_monthly_rev = st.slider("Average Monthly Revenue per User (ARPU)", min_value=30.0, max_value=150.0, value=75.0, step=2.5)
    with calc_c3:
        retention_success = st.slider("Targeted Retention Success Rate (%)", min_value=10, max_value=60, value=35, step=5)
        
    saved_customers = int(at_risk_pool * (retention_success / 100))
    monthly_saved = saved_customers * avg_monthly_rev
    annual_saved = monthly_saved * 12
    
    res_c1, res_c2, res_c3 = st.columns(3)
    with res_c1:
        st.markdown(f"""
            <div class="kpi-card retained">
                <div class="kpi-title">PRESERVED SUBSCRIBERS</div>
                <div class="kpi-value">{saved_customers:,}</div>
                <div class="kpi-sub">{retention_success}% Intervention Recovery</div>
            </div>
        """, unsafe_allow_html=True)
    with res_c2:
        st.markdown(f"""
            <div class="kpi-card retained">
                <div class="kpi-title">MONTHLY RECURRING REVENUE (MRR)</div>
                <div class="kpi-value">${monthly_saved:,.0f}</div>
                <div class="kpi-sub">Preserved Monthly Cashflow</div>
            </div>
        """, unsafe_allow_html=True)
    with res_c3:
        st.markdown(f"""
            <div class="kpi-card retained">
                <div class="kpi-title">ANNUAL RECURRING REVENUE (ARR)</div>
                <div class="kpi-value">${annual_saved:,.0f}</div>
                <div class="kpi-sub">Total Bottom-Line Value</div>
            </div>
        """, unsafe_allow_html=True)

    render_insight_box(f"By successfully intervening with {retention_success}% of {at_risk_pool:,} high-risk customers paying ${avg_monthly_rev:.2f}/mo, the enterprise secures ${annual_saved:,.0f} in annual revenue that would otherwise be permanently lost to competitor defection.")

    render_footer()

if __name__ == "__main__":
    apply_custom_css()
    render_retention_page()
