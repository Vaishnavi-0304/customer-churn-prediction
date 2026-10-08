"""
Machine Learning Insights & Academic Workflow Demonstration View
Author: ChurnGuard AI Team
Description:
    Exhaustive documentation of the complete 12-stage ML lifecycle:
    dataset telemetry, preprocessing pipeline, feature selection analysis,
    model comparison table, ROC curves, confusion matrices, GridSearchCV tuning results,
    and deep theoretical explanations of why model performances differ.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from components.ui import render_page_header, render_insight_box, render_footer, PLOTLY_DARK_LAYOUT, apply_custom_css

@st.cache_data
def load_ml_summary():
    """Load training summary and metrics from serialized JSON."""
    summary_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "training_summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def render_ml_insights_page():
    render_page_header("Machine Learning Workflow & Insights", "Academic project verification covering end-to-end data pipeline, feature selection, 5 algorithms, GridSearchCV tuning, and comparative diagnostics.")
    
    summary = load_ml_summary()
    if not summary:
        st.error("Model artifacts not found. Please verify that `src/train.py` was executed.")
        return
        
    ds = summary["dataset_stats"]
    fs = summary["feature_selection"]
    evals = summary["model_evaluations"]
    best_name = summary["best_model_name"]
    tuning = summary["tuning_details"]
    reasons = summary["performance_explanations"]

    # Top banner highlighting best model
    st.markdown(f"""
        <div class="glass-card" style="border:1px solid #10b981; background:rgba(16, 185, 129, 0.05); padding:16px 22px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                <div>
                    <span style="font-size:0.78rem; letter-spacing:0.06em; color:#34d399; font-weight:700; text-transform:uppercase;">CHAMPION MODEL SELECTED</span>
                    <h3 style="margin:4px 0 0 0; color:#ffffff;">{best_name}</h3>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:0.8rem; color:#94a3b8;">Primary Evaluation Metric:</span>
                    <div style="font-size:1.4rem; font-weight:800; color:#10b981;">ROC-AUC: 84.79% | Acc: 80.84%</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Workflow Navigation Tabs
    tab_overview, tab_fs, tab_models, tab_tuning, tab_eval, tab_theory = st.tabs([
        "1. Dataset & Pipeline",
        "2. Feature Selection",
        "3. Model Comparison",
        "4. Hyperparameter Tuning",
        "5. ROC & Confusion Matrices",
        "6. Why Performances Differ"
    ])

    # ----------------------------------------------------
    # TAB 1: DATASET & PREPROCESSING PIPELINE
    # ----------------------------------------------------
    with tab_overview:
        st.markdown("""<div class="glass-card"><h4 style="margin-top:0; color:#fff;">📊 Dataset Overview & Stratified Split</h4></div>""", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{ds['total_records']:,}")
        c2.metric("Raw Features", f"{ds['total_features']}")
        c3.metric("Training Split (80%)", f"{ds['train_records']:,}")
        c4.metric("Holdout Test Split (20%)", f"{ds['test_records']:,}")

        st.markdown(f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:16px;">
                <div style="background:rgba(15, 23, 42, 0.7); border:1px solid rgba(255, 255, 255, 0.06); border-radius:12px; padding:16px;">
                    <strong style="color:#ffffff;">Dataset Source & Metadata:</strong>
                    <ul style="color:#cbd5e1; font-size:0.85rem; margin-top:8px; line-height:1.6;">
                        <li><strong>Origin:</strong> Real-world IBM Telco Customer Churn dataset</li>
                        <li><strong>Target Variable:</strong> <code>Churn</code> (Binary: 0 = Retained, 1 = Churned)</li>
                        <li><strong>Class Imbalance:</strong> {ds['target_distribution']['retained']:,} Retained ({100 - ds['target_distribution']['churn_rate_pct']}%) vs {ds['target_distribution']['churned']:,} Churned ({ds['target_distribution']['churn_rate_pct']}%)</li>
                        <li><strong>Data Leakage Prevention:</strong> Stratified split executed prior to any scaling or encoding. Preprocessor fitted exclusively on training set.</li>
                    </ul>
                </div>
                <div style="background:rgba(15, 23, 42, 0.7); border:1px solid rgba(255, 255, 255, 0.06); border-radius:12px; padding:16px;">
                    <strong style="color:#ffffff;">Scikit-learn Pipeline Transformations:</strong>
                    <ul style="color:#cbd5e1; font-size:0.85rem; margin-top:8px; line-height:1.6;">
                        <li><strong>TotalCharges Handling:</strong> Blank strings from 11 zero-tenure rows coerced to 0.0 float.</li>
                        <li><strong>Numerical Scaler:</strong> <code>StandardScaler()</code> applied to tenure, MonthlyCharges, TotalCharges (Z-score normalization).</li>
                        <li><strong>Categorical Encoder:</strong> <code>OneHotEncoder(sparse_output=False, handle_unknown='ignore')</code> across 16 multi-class attributes.</li>
                        <li><strong>Engineered Dimension:</strong> Expanded from 19 raw features to 46 one-hot encoded regressors.</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 2: FEATURE SELECTION
    # ----------------------------------------------------
    with tab_fs:
        st.markdown("""<div class="glass-card"><h4 style="margin-top:0; color:#fff;">🔬 Feature Selection Methodology & Importance</h4></div>""", unsafe_allow_html=True)
        
        fs_c1, fs_c2, fs_c3 = st.columns(3)
        fs_c1.metric("Original Feature Dimension", fs["total_original_features"])
        fs_c2.metric("Selected Key Features", fs["total_selected_features"])
        fs_c3.metric("Low-Signal Features Filtered", fs["total_removed_features"])

        # Top 10 features bar chart
        top10_df = pd.DataFrame(fs["top_10_features"]).sort_values(by="composite_score", ascending=True)
        fig_feat = px.bar(
            top10_df,
            x="composite_score",
            y="feature",
            orientation="h",
            color="composite_score",
            color_continuous_scale="Purples",
            labels={"composite_score": "Tri-Method Composite Score (MI + ANOVA + RF)", "feature": "Feature"}
        )
        fig_feat.update_layout(**PLOTLY_DARK_LAYOUT, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_feat, use_container_width=True)
        render_insight_box("Contract type (Month-to-Month), tenure, TotalCharges, MonthlyCharges, and InternetService_Fiber optic dominate predictive influence across Mutual Information, ANOVA F-scores, and Random Forest Gini impurity reduction.")

        # Removed features documentation table
        st.markdown("""<h5 style="color:#ffffff; margin-top:20px;">Documented Rationale for Feature Pruning / Low Variance:</h5>""", unsafe_allow_html=True)
        removed_df = pd.DataFrame(fs["removed_features"])[["feature", "composite_score", "reason"]]
        st.dataframe(removed_df, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: MODEL COMPARISON
    # ----------------------------------------------------
    with tab_models:
        st.markdown("""<div class="glass-card"><h4 style="margin-top:0; color:#fff;">🏆 Algorithm Comparison Table</h4></div>""", unsafe_allow_html=True)
        
        comp_df = pd.DataFrame(evals)[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].copy()
        comp_df.columns = ["Model Name", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
        comp_df = comp_df.sort_values(by=["ROC-AUC", "F1 Score"], ascending=False).reset_index(drop=True)
        
        # Highlight best model
        st.dataframe(
            comp_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], color="#1e3a8a"),
            use_container_width=True
        )

        # Visual comparison bar chart
        melted_comp = comp_df.melt(id_vars=["Model Name"], value_vars=["Accuracy", "F1 Score", "ROC-AUC"], var_name="Metric", value_name="Score")
        fig_comp = px.bar(
            melted_comp,
            x="Model Name",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_sequence=["#06B6D4", "#F59E0B", "#10B981"]
        )
        fig_comp.update_layout(**PLOTLY_DARK_LAYOUT, height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_comp, use_container_width=True)
        render_insight_box("While all models achieve ~78-81% accuracy, accuracy alone masks false negatives due to the 26.5% minority class. ROC-AUC and F1 Score clearly isolate XGBoost (Tuned) and Logistic Regression as superior at separating churners.")

    # ----------------------------------------------------
    # TAB 4: HYPERPARAMETER TUNING
    # ----------------------------------------------------
    with tab_tuning:
        st.markdown("""<div class="glass-card"><h4 style="margin-top:0; color:#fff;">⚙️ Hyperparameter Tuning Results (5-Fold Stratified CV)</h4></div>""", unsafe_allow_html=True)
        
        tune_col1, tune_col2 = st.columns(2)
        
        with tune_col1:
            st.markdown("""
                <div class="glass-card" style="border-left:3px solid #8b5cf6;">
                    <h4 style="margin-top:0; color:#fff;">Random Forest Tuning</h4>
                    <p style="color:#94a3b8; font-size:0.85rem;">GridSearchCV over <code>n_estimators</code>, <code>max_depth</code>, <code>min_samples_split</code>, <code>min_samples_leaf</code>, and <code>class_weight</code>.</p>
                </div>
            """, unsafe_allow_html=True)
            
            rf_t = tuning["random_forest"]
            st.json(rf_t["best_params"])
            
            rf_comp_data = {
                "Metric": ["Accuracy", "F1 Score", "ROC-AUC"],
                "Default (Before)": [rf_t["before_tuning"]["accuracy"], rf_t["before_tuning"]["f1_score"], rf_t["before_tuning"]["roc_auc"]],
                "Tuned (After)": [rf_t["after_tuning"]["accuracy"], rf_t["after_tuning"]["f1_score"], rf_t["after_tuning"]["roc_auc"]]
            }
            st.table(pd.DataFrame(rf_comp_data))

        with tune_col2:
            st.markdown("""
                <div class="glass-card" style="border-left:3px solid #06b6d4;">
                    <h4 style="margin-top:0; color:#fff;">XGBoost Classifier Tuning</h4>
                    <p style="color:#94a3b8; font-size:0.85rem;">GridSearchCV over <code>learning_rate</code>, <code>max_depth</code>, <code>n_estimators</code>, <code>subsample</code>, and <code>colsample_bytree</code>.</p>
                </div>
            """, unsafe_allow_html=True)
            
            xgb_t = tuning["xgboost"]
            st.json(xgb_t["best_params"])
            
            xgb_comp_data = {
                "Metric": ["Accuracy", "F1 Score", "ROC-AUC"],
                "Default (Before)": [xgb_t["before_tuning"]["accuracy"], xgb_t["before_tuning"]["f1_score"], xgb_t["before_tuning"]["roc_auc"]],
                "Tuned (After)": [xgb_t["after_tuning"]["accuracy"], xgb_t["after_tuning"]["f1_score"], xgb_t["after_tuning"]["roc_auc"]]
            }
            st.table(pd.DataFrame(xgb_comp_data))

        render_insight_box("Hyperparameter tuning regularized tree growth (limiting max_depth to 3-5 and dialing learning_rate to 0.03-0.10). This eliminated training split overfitting and elevated holdout ROC-AUC from 0.8152 to 0.8479.")

    # ----------------------------------------------------
    # TAB 5: ROC CURVES & CONFUSION MATRICES
    # ----------------------------------------------------
    with tab_eval:
        st.markdown("""<div class="glass-card"><h4 style="margin-top:0; color:#fff;">📈 ROC Curves & Confusion Matrix Inspector</h4></div>""", unsafe_allow_html=True)
        
        # ROC Curves
        fig_roc = go.Figure()
        # Add diagonal reference line
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#64748b'), name='Random Chance (AUC = 0.50)'))
        
        colors = ['#6366F1', '#EC4899', '#F59E0B', '#10B981', '#3B82F6', '#8B5CF6', '#06B6D4']
        for idx, mod in enumerate(evals):
            if "roc_curve" in mod:
                pts = mod["roc_curve"]
                fpr = [p["fpr"] for p in pts]
                tpr = [p["tpr"] for p in pts]
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    name=f"{mod['model_name']} (AUC = {mod['roc_auc']:.3f})",
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
                
        fig_roc.update_layout(
            **PLOTLY_DARK_LAYOUT,
            height=400,
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall)",
            legend=dict(orientation="v", yanchor="bottom", y=0.05, xanchor="right", x=0.98)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Confusion Matrix Selector
        st.markdown("""<h5 style="color:#ffffff; margin-top:20px;">Confusion Matrix Breakdown (Holdout Test Set: 1,409 Records):</h5>""", unsafe_allow_html=True)
        sel_model_cm = st.selectbox("Select Model for Detailed Matrix", [m["model_name"] for m in evals], index=evals.index(next(m for m in evals if m["model_name"] == best_name)))
        
        target_eval = next(m for m in evals if m["model_name"] == sel_model_cm)
        cm = target_eval["confusion_matrix"]
        
        cm_col1, cm_col2 = st.columns([1, 1.2])
        with cm_col1:
            cm_fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
                x=['Retained (0)', 'Churned (1)'],
                y=['Retained (0)', 'Churned (1)'],
                color_continuous_scale="Purples"
            )
            cm_fig.update_layout(**PLOTLY_DARK_LAYOUT, height=280, coloraxis_showscale=False)
            st.plotly_chart(cm_fig, use_container_width=True)
            
        with cm_col2:
            st.markdown(f"""
                <div style="background:rgba(15, 23, 42, 0.7); border:1px solid rgba(255, 255, 255, 0.08); border-radius:12px; padding:18px;">
                    <h5 style="margin-top:0; color:#ffffff;">{sel_model_cm} Diagnostic Metrics:</h5>
                    <ul style="color:#cbd5e1; font-size:0.88rem; line-height:1.7;">
                        <li><strong>True Negatives (TN):</strong> {target_eval['true_negatives']} correctly predicted loyal accounts</li>
                        <li><strong>False Positives (FP):</strong> {target_eval['false_positives']} false alarms (retained classified as churn)</li>
                        <li><strong>False Negatives (FN):</strong> {target_eval['false_negatives']} missed churners (critical business risk)</li>
                        <li><strong>True Positives (TP):</strong> {target_eval['true_positives']} correctly flagged at-risk accounts</li>
                        <li><strong>Recall / Sensitivity:</strong> {target_eval['recall'] * 100:.1f}%</li>
                        <li><strong>Precision:</strong> {target_eval['precision'] * 100:.1f}%</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 6: WHY MODEL PERFORMANCES DIFFER
    # ----------------------------------------------------
    with tab_theory:
        st.markdown("""<div class="glass-card"><h4 style="margin-top:0; color:#fff;">🧠 Theoretical Analysis: Why Algorithm Performances Differ</h4></div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="insight-box" style="margin-bottom:20px; font-size:0.92rem; line-height:1.6;">
                <strong>Synthesis:</strong> {reasons['synthesis']}
            </div>
        """, unsafe_allow_html=True)

        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown(f"""
                <div class="glass-card" style="border-left:3px solid #6366f1;">
                    <h5 style="color:#ffffff; margin-top:0;">1. Logistic Regression (Baseline Linear Model)</h5>
                    <p style="color:#a5b4fc; font-size:0.83rem;"><strong>Strengths:</strong> {reasons['logistic_regression']['strengths']}</p>
                    <p style="color:#fca5a5; font-size:0.83rem;"><strong>Limitations:</strong> {reasons['logistic_regression']['limitations']}</p>
                    <p style="color:#cbd5e1; font-size:0.83rem;"><strong>Verdict:</strong> {reasons['logistic_regression']['verdict']}</p>
                </div>
                <div class="glass-card" style="border-left:3px solid #10b981;">
                    <h5 style="color:#ffffff; margin-top:0;">2. Random Forest (Bagging Ensemble)</h5>
                    <p style="color:#a5b4fc; font-size:0.83rem;"><strong>Strengths:</strong> {reasons['random_forest']['strengths']}</p>
                    <p style="color:#fca5a5; font-size:0.83rem;"><strong>Limitations:</strong> {reasons['random_forest']['limitations']}</p>
                    <p style="color:#cbd5e1; font-size:0.83rem;"><strong>Verdict:</strong> {reasons['random_forest']['verdict']}</p>
                </div>
            """, unsafe_allow_html=True)

        with col_t2:
            st.markdown(f"""
                <div class="glass-card" style="border-left:3px solid #06b6d4;">
                    <h5 style="color:#ffffff; margin-top:0;">3. XGBoost (Extreme Gradient Boosting)</h5>
                    <p style="color:#a5b4fc; font-size:0.83rem;"><strong>Strengths:</strong> {reasons['xgboost']['strengths']}</p>
                    <p style="color:#fca5a5; font-size:0.83rem;"><strong>Limitations:</strong> {reasons['xgboost']['limitations']}</p>
                    <p style="color:#cbd5e1; font-size:0.83rem;"><strong>Verdict:</strong> {reasons['xgboost']['verdict']}</p>
                </div>
                <div class="glass-card" style="border-left:3px solid #f59e0b;">
                    <h5 style="color:#ffffff; margin-top:0;">4. Decision Tree & Gradient Boosting</h5>
                    <p style="color:#cbd5e1; font-size:0.83rem;">Decision trees overfit single split partitions (high variance), while Gradient Boosting sequentially remedies errors but requires XGBoost's column subsampling and L2 shrinkage to avoid marginal drift.</p>
                </div>
            """, unsafe_allow_html=True)

    render_footer()

if __name__ == "__main__":
    apply_custom_css()
    if not st.session_state.get("authenticated", False):
        st.warning("🔒 Authentication Required: Please sign in on the main portal to access this page.")
        st.stop()
    render_ml_insights_page()
