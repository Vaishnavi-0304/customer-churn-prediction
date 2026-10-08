"""
Model Training and Hyperparameter Tuning Module
Author: ChurnGuard AI Team
Description:
    Trains Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.
    Executes systematic Hyperparameter Tuning (GridSearchCV/RandomizedSearchCV),
    evaluates on stratified holdout test set, selects the best-performing model,
    and persists all models, pipelines, and evaluation metrics.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from preprocessing import prepare_train_test_data
from feature_selection import run_feature_selection
from evaluate import evaluate_model, get_performance_discrepancy_explanation

def train_and_tune_all(data_path: str, models_dir: str):
    """
    Executes full machine learning pipeline from raw data to model persistence.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n[1/6] Loading and Preprocessing Dataset...")
    data = prepare_train_test_data(data_path)
    X_train_df = data['X_train_df']
    X_test_df = data['X_test_df']
    y_train = data['y_train']
    y_test = data['y_test']
    preprocessor = data['preprocessor']
    feature_names = data['feature_names']
    clean_df = data['clean_full_df']
    
    print(f"Dataset shape: {clean_df.shape} | Training rows: {len(X_train_df)} | Test rows: {len(X_test_df)}")
    
    print("\n[2/6] Running Feature Selection & Importance Analysis...")
    fs_results, fs_df = run_feature_selection(X_train_df, y_train, top_k=35)
    
    # ----------------------------------------------------
    # Baseline Models Setup
    # ----------------------------------------------------
    print("\n[3/6] Training Baseline Candidate Models...")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest (Default)": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost (Default)": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    baseline_evals = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_df, y_train)
        baseline_evals[name] = evaluate_model(model, X_test_df, y_test, model_name=name)
        print(f"    -> Acc: {baseline_evals[name]['accuracy']:.4f} | F1: {baseline_evals[name]['f1_score']:.4f} | ROC-AUC: {baseline_evals[name]['roc_auc']:.4f}")
        
    # ----------------------------------------------------
    # Hyperparameter Tuning
    # ----------------------------------------------------
    print("\n[4/6] Conducting Hyperparameter Tuning with 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Random Forest Tuning
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    }
    print("  Tuning Random Forest...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    rf_grid.fit(X_train_df, y_train)
    rf_best = rf_grid.best_estimator_
    rf_tuned_eval = evaluate_model(rf_best, X_test_df, y_test, model_name="Random Forest (Tuned)")
    
    # 2. XGBoost Tuning
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.03, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    print("  Tuning XGBoost...")
    xgb_grid = GridSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_grid=xgb_param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    xgb_grid.fit(X_train_df, y_train)
    xgb_best = xgb_grid.best_estimator_
    xgb_tuned_eval = evaluate_model(xgb_best, X_test_df, y_test, model_name="XGBoost (Tuned)")
    
    # ----------------------------------------------------
    # Comprehensive Comparison
    # ----------------------------------------------------
    print("\n[5/6] Consolidating Model Comparison Table...")
    
    comparison_models = {
        "Logistic Regression": models["Logistic Regression"],
        "Decision Tree": models["Decision Tree"],
        "Random Forest (Default)": models["Random Forest (Default)"],
        "Random Forest (Tuned)": rf_best,
        "Gradient Boosting": models["Gradient Boosting"],
        "XGBoost (Default)": models["XGBoost (Default)"],
        "XGBoost (Tuned)": xgb_best
    }
    
    all_evaluations = [
        baseline_evals["Logistic Regression"],
        baseline_evals["Decision Tree"],
        baseline_evals["Random Forest (Default)"],
        rf_tuned_eval,
        baseline_evals["Gradient Boosting"],
        baseline_evals["XGBoost (Default)"],
        xgb_tuned_eval
    ]
    
    # Sort models by ROC-AUC and F1-score to pick overall best
    comparison_df = pd.DataFrame(all_evaluations).sort_values(by=['roc_auc', 'f1_score'], ascending=False)
    best_model_name = comparison_df.iloc[0]['model_name']
    best_model = comparison_models[best_model_name]
    
    print(f"\n>>> Best Performing Model Selected: {best_model_name}")
    print(f"    ROC-AUC: {comparison_df.iloc[0]['roc_auc']} | F1: {comparison_df.iloc[0]['f1_score']} | Accuracy: {comparison_df.iloc[0]['accuracy']}")
    
    tuning_comparison = {
        "random_forest": {
            "default_params": {"n_estimators": 100, "max_depth": "None", "min_samples_split": 2, "min_samples_leaf": 1},
            "best_params": {k: (str(v) if v is None else v) for k, v in rf_grid.best_params_.items()},
            "before_tuning": baseline_evals["Random Forest (Default)"],
            "after_tuning": rf_tuned_eval
        },
        "xgboost": {
            "default_params": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3, "subsample": 1.0},
            "best_params": {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in xgb_grid.best_params_.items()},
            "before_tuning": baseline_evals["XGBoost (Default)"],
            "after_tuning": xgb_tuned_eval
        }
    }
    
    # ----------------------------------------------------
    # Serialization
    # ----------------------------------------------------
    print("\n[6/6] Saving Trained Artifacts...")
    
    # 1. Best Model & Preprocessor
    joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessing_pipeline.pkl"))
    
    # 2. All Models dict for live exploration
    joblib.dump(comparison_models, os.path.join(models_dir, "all_models.pkl"))
    
    # 3. Training Summary JSON
    dataset_stats = {
        "source": "IBM Telco Customer Churn Dataset",
        "total_records": int(len(clean_df)),
        "total_features": int(clean_df.shape[1] - 1),
        "target_distribution": {
            "retained": int((clean_df['Churn'] == 'No').sum()),
            "churned": int((clean_df['Churn'] == 'Yes').sum()),
            "churn_rate_pct": round(float((clean_df['Churn'] == 'Yes').mean() * 100), 2)
        },
        "train_records": int(len(X_train_df)),
        "test_records": int(len(X_test_df)),
        "numerical_attributes": ['tenure', 'MonthlyCharges', 'TotalCharges'],
        "categorical_attributes": [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ]
    }
    
    summary_payload = {
        "dataset_stats": dataset_stats,
        "feature_selection": fs_results,
        "model_evaluations": all_evaluations,
        "best_model_name": best_model_name,
        "tuning_details": tuning_comparison,
        "performance_explanations": get_performance_discrepancy_explanation(),
        "feature_names": feature_names
    }
    
    with open(os.path.join(models_dir, "training_summary.json"), "w") as f:
        json.dump(summary_payload, f, indent=2)
        
    print("Training pipeline finished successfully!")
    print(f"Artifacts persisted to {models_dir}")
    return summary_payload

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "customer_churn.csv")
    models_dir = os.path.join(base_dir, "..", "models")
    train_and_tune_all(data_path, models_dir)
