"""
UI Component Helpers for ChurnGuard AI
Author: ChurnGuard AI Team
Description:
    Reusable UI building blocks for cards, KPI metrics, Plotly charts,
    risk gauges, headers, and navigation elements.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import os

PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
)

def apply_custom_css():
    """Injects custom SaaS dark stylesheet."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def render_brand_header():
    """Renders sleek sidebar brand banner."""
    st.markdown("""
        <div class="brand-header">
            <div class="brand-logo">🛡️</div>
            <div class="brand-text">
                <h2>ChurnGuard AI</h2>
                <p>Customer Intelligence Platform</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_page_header(title: str, subtitle: str):
    """Renders styled page title and subtitle."""
    st.markdown(f"""
        <div>
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
    """, unsafe_allow_html=True)

def render_kpi_card(title: str, value: str, sub: str, category: str = "normal"):
    """Renders glassmorphic KPI summary card."""
    card_class = f"kpi-card {category}"
    return f"""
        <div class="{card_class}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
    """

def render_insight_box(text: str):
    """Renders human-readable analytics insight callout."""
    st.markdown(f"""
        <div class="insight-box">
            💡 <strong>Key Business Insight:</strong> {text}
        </div>
    """, unsafe_allow_html=True)

def create_risk_meter_chart(probability_pct: float, risk_level: str, risk_color: str):
    """Creates a high-precision Plotly gauge chart for churn probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability_pct,
        number={'suffix': "%", 'font': {'size': 44, 'color': '#ffffff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#64748b', 'tickfont': {'color': '#94a3b8'}},
            'bar': {'color': risk_color, 'thickness': 0.3},
            'bgcolor': 'rgba(15, 23, 42, 0.6)',
            'borderwidth': 1,
            'bordercolor': 'rgba(255, 255, 255, 0.1)',
            'steps': [
                {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 3},
                'thickness': 0.8,
                'value': probability_pct
            }
        }
    ))
    
    layout_args = {
        **PLOTLY_DARK_LAYOUT,
        "height": 240,
        "margin": dict(l=25, r=25, t=25, b=15)
    }
    fig.update_layout(**layout_args)
    return fig

def render_footer():
    """Renders standard application footer."""
    st.markdown("""
        <div class="footer-container">
            <p><strong>ChurnGuard AI</strong> • Customer Intelligence & Churn Prediction System</p>
            <p>Built with Python, Scikit-learn, XGBoost, Pandas, Plotly & Streamlit • Powered by Real-World IBM Telco Dataset</p>
        </div>
    """, unsafe_allow_html=True)
