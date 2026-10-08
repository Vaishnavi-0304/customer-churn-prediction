# CUSTOMER CHURN PREDICTION SYSTEM
> **Tagline:** *"Predict. Prevent. Retain."*  
> **Brand Name:** **ChurnGuard** — *Customer Intelligence Platform*

An enterprise-grade, full-lifecycle Machine Learning web application designed to forecast customer defection, isolate critical risk drivers, evaluate competing classification architectures, and generate automated retention playbooks for telecommunications service providers.

---

## 📌 Executive Summary & Objective

Customer acquisition in telecommunications costs approximately 5 to 7 times more than customer retention. The objective of **ChurnGuard** is to transition retention strategies from **reactive damage control** to **proactive algorithmic intervention**.

By analyzing historical behavioral, contractual, and billing telemetry, the system:
1. Identifies early defection indicators before cancellation occurs.
2. Quantifies customer-specific churn risk into calibrated probability percentages.
3. Classifies customers into **LOW (0–35%)**, **MEDIUM (35–65%)**, and **HIGH (65–100%)** risk tiers.
4. Explains individual risk factors using behavioral feature impact analysis.
5. Prescribes automated, risk-tiered retention playbooks to protect Monthly Recurring Revenue (MRR).

---

## 🏗️ 12-Stage Machine Learning Lifecycle Workflow

This project adheres to the complete end-to-end Machine Learning lifecycle:

```
[1. Problem Definition] 
         ↓
[2. Real-World Dataset Selection (IBM Telco 7,043 Records)]
         ↓
[3. Robust Data Cleaning & Preprocessing (StandardScaler + OneHotEncoder)]
         ↓
[4. Exploratory Data Analysis (EDA with Plotly Visualizations)]
         ↓
[5. Feature Selection (Mutual Information + ANOVA + Random Forest Importance)]
         ↓
[6. Model Development (5 Candidate Classification Algorithms)]
         ↓
[7. Hyperparameter Tuning (5-Fold Stratified GridSearchCV)]
         ↓
[8. Comprehensive Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, CM)]
         ↓
[9. Theoretical Model Comparison & Performance Discrepancy Analysis]
         ↓
[10. Champion Model & Pipeline Serialization (joblib)]
         ↓
[11. Interactive SaaS Web Application (Streamlit + SQLite Auth)]
         ↓
[12. Real-Time Inference, Factor Explainability & Retention Playbooks]
```

---

## 📊 Dataset Overview

The system is trained and validated on the real-world **IBM Telco Customer Churn Dataset**:

- **Total Records:** 7,043 customer accounts
- **Total Features:** 21 original attributes (demographic, service, and billing)
- **Target Variable:** `Churn` (Binary: `0 = Retained`, `1 = Churned`)
- **Target Distribution:**
  - Retained: 5,174 customers (73.46%)
  - Churned: 1,869 customers (26.54%)
- **Data Attributes:**
  - **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - **Account & Contract:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - **Subscribed Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## 🧹 Data Preprocessing & Leakage Prevention

- **Type Correction & Null Imputation:** The `TotalCharges` attribute contained 11 whitespace strings corresponding to new customers with `tenure = 0`. These were coerced to float and imputed to `0.0`.
- **Duplicate Removal:** Checked and confirmed zero duplicate account rows.
- **Stratified Holdout Split:** Split into **80% Training (5,634 rows)** and **20% Test (1,409 rows)** using stratified sampling on the imbalanced `Churn` label.
- **Leakage Prevention:** The `ColumnTransformer` pipeline (`StandardScaler` for continuous features and `OneHotEncoder(handle_unknown='ignore')` for categorical features) was fitted **strictly on the training split**, ensuring zero holdout leakage.

---

## 🔬 Feature Selection & Importance Ranking

A tri-method composite feature evaluation was conducted combining:
1. **Mutual Information (`mutual_info_classif`):** Captures non-linear dependencies.
2. **ANOVA F-Test (`f_classif`):** Measures linear between-group variance.
3. **Random Forest Gini Importance:** Evaluates decision tree impurity reduction.

### Top 5 Churn Drivers:
1. **Contract_Month-to-month:** Highest single predictor; 42.7% churn rate.
2. **Tenure:** Inversely correlated; >55% of churn occurs within months 0–12.
3. **TotalCharges & MonthlyCharges:** Price sensitivity; median bill of churners is $79.65 vs $64.43 for loyal customers.
4. **InternetService_Fiber optic:** 41.9% churn rate due to premium pricing without tech support.
5. **PaymentMethod_Electronic check:** 45.3% churn rate due to manual payment friction.

### Documented Feature Removal:
Features such as `gender_Male` and `PhoneService` were pruned or deprioritized due to negligible signal-to-noise ratios (churn rates across genders are virtually identical at ~26%).

---

## 🤖 Algorithms & Model Development

We developed and benchmarked 5 distinct classification models:

1. **Logistic Regression:** Linear baseline providing fast, interpretable log-odds estimation.
2. **Decision Tree Classifier:** Single white-box decision partition benchmark.
3. **Random Forest Classifier:** Bagging ensemble averaging decorrelated deep decision trees.
4. **Gradient Boosting Classifier:** Sequential boosting minimizing residual errors.
5. **XGBoost (Extreme Gradient Boosting):** Regularized gradient boosted tree system with second-order Taylor gradients.

---

## ⚙️ Hyperparameter Tuning (5-Fold Stratified GridSearchCV)

Grid search with 5-fold cross-validation was conducted on key ensemble models:

- **Random Forest:** Tuned across `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `class_weight`.
- **XGBoost:** Tuned across `n_estimators` [100, 200], `max_depth` [3, 5, 7], `learning_rate` [0.03, 0.1], `subsample` [0.8, 1.0], and `colsample_bytree` [0.8, 1.0].

### Tuning Performance Lift:
| Model | Pre-Tuning ROC-AUC | Post-Tuning ROC-AUC | Lift |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 0.8183 | 0.8415 | **+2.32%** |
| **XGBoost** | 0.8152 | **0.8479** | **+3.27%** |

---

## 🏆 Model Evaluation & Comparison Table

Evaluated on the independent **1,409 holdout test records**:

| Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **80.84%** | **68.22%** | **51.98%** | **0.5897** | **0.8479** | 🥇 **Best Model** |
| **Gradient Boosting** | 80.27% | 66.88% | 51.34% | 0.5813 | 0.8433 | Runner-Up |
| **Logistic Regression** | 80.55% | 66.21% | 55.45% | 0.6040 | 0.8420 | Strong Linear Baseline |
| **Random Forest (Tuned)** | 79.91% | 66.39% | 48.93% | 0.5630 | 0.8415 | Strong Bagging Ensemble |
| **Decision Tree** | 79.84% | 65.52% | 55.12% | 0.5989 | 0.8297 | Overfits Splits |
| **Random Forest (Default)**| 78.21% | 61.42% | 48.66% | 0.5425 | 0.8183 | Unregularized Trees |
| **XGBoost (Default)** | 77.29% | 58.74% | 50.80% | 0.5442 | 0.8152 | Unregularized Boosting |

---

## 🧠 Why Model Performances Differ (Academic Theory)

1. **Linear vs Non-Linear Boundaries:** Customer churn is governed by non-linear combinations (e.g., month-to-month contract + fiber optic + no tech support = 65%+ churn). Logistic Regression relies on monotonic additive log-odds, missing high-order feature cross-products.
2. **Variance Reduction in Tree Ensembles:** Single Decision Trees overfit localized splits. Random Forest reduces variance via feature bagging, while XGBoost builds shallow trees sequentially with L1/L2 shrinkage ($\gamma, \lambda$) to prevent over-specialization.
3. **Class Imbalance Effect:** With 26.5% churn prevalence, uncalibrated tree models disproportionately favor the majority class (retained customers). XGBoost with regularized learning rates balances minority class identification (Recall) without triggering excessive false alarms (Precision).

---

## 💻 Web Application Features

Designed as a modern dark analytics SaaS platform:
- **Authentication & Security:** SQLite3 database with PBKDF2-HMAC-SHA256 password hashing, validation, session management, and profile editing.
- **Executive Dashboard:** Real-time KPI cards, interactive Plotly donut charts, contract breakdowns, tenure distribution, and top churn drivers.
- **Customer Analytics (EDA):** Dynamic multi-parameter filtering across contract, internet type, and payment methods with 8 deep-dive charts and human-readable insights.
- **Real-Time Prediction Engine:** Multi-section input forms (Profile, Services, Billing), 1-click test presets, animated probability gauge, risk tier badges (LOW, MEDIUM, HIGH), and factor explainability.
- **Retention Playbooks & Strategy:** Risk-tiered mitigation frameworks, root cause matrix, and interactive Financial ROI & ARR preservation calculator.
- **Academic ML Insights:** Complete transparency view for classroom evaluations displaying pipeline schematics, feature selection rankings, comparison charts, ROC curves, and confusion matrices.

---

## 📂 Project Directory Structure

```
customer-churn-prediction/
│
├── app/
│   ├── main.py                     # Streamlit application entrypoint & routing
│   ├── assets/
│   │   └── style.css               # Modern dark SaaS theme stylesheet
│   ├── components/
│   │   └── ui.py                   # Reusable UI cards, gauges, and headers
│   └── pages/
│       ├── landing.py              # Landing page with hero banner & auth
│       ├── dashboard.py            # Executive KPI dashboard & macro charts
│       ├── analytics.py            # Exploratory Data Analysis & filterable EDA
│       ├── prediction.py           # Real-time churn predictor & explainability
│       ├── retention.py            # Business playbooks & financial ROI calculator
│       ├── ml_insights.py          # Academic ML workflow & diagnostic verification
│       └── profile.py              # User account settings & security status
│
├── data/
│   └── customer_churn.csv          # Real-world IBM Telco Churn Dataset (7,043 rows)
│
├── models/
│   ├── best_model.pkl              # Champion trained model (XGBoost Tuned)
│   ├── preprocessing_pipeline.pkl  # Fitted ColumnTransformer pipeline
│   ├── all_models.pkl              # Dictionary of all 5 candidate models
│   └── training_summary.json       # Serialized metrics, ROC curves, tuning records
│
├── notebooks/
│   └── churn_analysis.ipynb        # Step-by-step Jupyter Notebook workflow
│
├── src/
│   ├── auth.py                     # SQLite authentication & PBKDF2 hashing
│   ├── preprocessing.py            # Data cleaning & ColumnTransformer pipeline
│   ├── feature_selection.py        # Tri-method feature selection & ranking
│   ├── train.py                    # Model training & GridSearchCV execution
│   ├── evaluate.py                 # Evaluation metrics, ROC curve, & theory
│   └── predict.py                  # Real-time inference & rule-based explainability
│
├── database/
│   └── users.db                    # SQLite database for persistent user accounts
│
├── requirements.txt                # Python package dependencies
├── .gitignore                      # Git exclusion rules
└── README.md                       # Comprehensive project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10+ installed
- PowerShell / Command Prompt / Terminal

### 2. Clone or Navigate to the Project Directory
```powershell
cd C:\Users\vaishnavi\.gemini\antigravity\scratch\customer-churn-prediction
```

### 3. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 4. (Optional) Retrain Models & Pipeline
All models and preprocessors are already pre-trained and saved in `models/`. To execute the training and GridSearchCV pipeline from scratch:
```powershell
python src/train.py
```

### 5. Launch the Web Application
```powershell
streamlit run app/main.py
```
The application will launch automatically in your browser at `http://localhost:8501`.

### 6. Default Demo Credentials
You can register a new account on the landing page, or click **"⚡ Use Demo Account"**:
- **Email:** `demo@churnguard.ai`
- **Password:** `Admin@123`

---

## 🔮 Future Enhancements
- Integration with live customer CRM APIs (Salesforce, HubSpot) for automated webhook alerts.
- Incorporating SHAP (SHapley Additive exPlanations) force plots into the prediction UI.
- Expanding model retraining jobs via automated background cron schedules.

---
*Developed with ❤️ for Academic & Enterprise Machine Learning Excellence.*
