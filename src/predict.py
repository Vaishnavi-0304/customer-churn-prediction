"""
Inference and Explainability Module for Customer Churn Prediction System
Author: ChurnGuard AI Team
Description:
    Loads persisted model and preprocessing pipeline, runs real-time inference,
    categorizes risk level (LOW, MEDIUM, HIGH), and produces interpretable
    feature-level risk drivers and personalized retention recommendations.
"""

import os
import joblib
import numpy as np
import pandas as pd

class ChurnPredictor:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
            
        self.models_dir = models_dir
        self.model_path = os.path.join(models_dir, "best_model.pkl")
        self.pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Trained model or pipeline not found in {models_dir}")
            
        self.model = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)
        
        # Extract feature names from pipeline
        cat_encoder = self.pipeline.named_transformers_['cat'].named_steps['onehot']
        from preprocessing import CATEGORICAL_COLS, NUMERICAL_COLS
        self.feature_names = NUMERICAL_COLS + cat_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()

    def predict_single(self, customer_dict: dict) -> dict:
        """
        Takes raw customer dictionary, runs inference, and returns comprehensive
        risk assessment with business explanations and retention actions.
        """
        df = pd.DataFrame([customer_dict])
        
        # Ensure numerical types
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0.0)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
        df['SeniorCitizen'] = int(df['SeniorCitizen'].iloc[0]) if 'SeniorCitizen' in df.columns else 0
        
        # Transform through fitted pipeline
        X_transformed = self.pipeline.transform(df)
        X_df = pd.DataFrame(X_transformed, columns=self.feature_names)
        
        # Predict probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_df)[0]
            churn_prob = float(probs[1])
        else:
            pred = int(self.model.predict(X_df)[0])
            churn_prob = 1.0 if pred == 1 else 0.0
            
        churn_prob_pct = round(churn_prob * 100, 1)
        stay_prob_pct = round((1.0 - churn_prob) * 100, 1)
        
        # Categorize risk level
        if churn_prob_pct < 35.0:
            risk_level = "LOW RISK"
            prediction_label = "LIKELY TO STAY"
            risk_color = "#10B981"  # Emerald Green
            risk_badge = "success"
        elif churn_prob_pct <= 65.0:
            risk_level = "MEDIUM RISK"
            prediction_label = "MODERATE RISK / WATCHLIST"
            risk_color = "#F59E0B"  # Amber Orange
            risk_badge = "warning"
        else:
            risk_level = "HIGH RISK"
            prediction_label = "LIKELY TO CHURN"
            risk_color = "#EF4444"  # Rose Red
            risk_badge = "danger"
            
        # Analyze Customer Factors (Explainability)
        risk_factors = []
        positive_factors = []
        recommendations = []
        
        # 1. Contract Analysis
        contract = str(customer_dict.get('Contract', ''))
        if contract == 'Month-to-month':
            risk_factors.append({
                "factor": "Month-to-Month Contract",
                "impact": "High Risk Driver (+38% Churn Likelihood)",
                "detail": "Customer has zero contractual lock-in and can cancel without penalty anytime."
            })
            recommendations.append("Offer a 15% discount incentive to upgrade to a 1-year or 2-year commitment plan.")
        elif contract in ['One year', 'Two year']:
            positive_factors.append({
                "factor": f"{contract} Contract",
                "impact": "Strong Retention Anchor",
                "detail": "Committed agreement significantly reduces immediate switching behavior."
            })
            
        # 2. Tenure Analysis
        tenure = float(customer_dict.get('tenure', 0))
        if tenure < 12:
            risk_factors.append({
                "factor": f"Low Tenure ({int(tenure)} months)",
                "impact": "Early Lifecycle Vulnerability",
                "detail": "Customers in their first 12 months experience the highest churn rate (~47%)."
            })
            recommendations.append("Enroll in new-customer onboarding and schedule a 30-day proactive satisfaction check.")
        elif tenure >= 24:
            positive_factors.append({
                "factor": f"Established Loyalty ({int(tenure)} months tenure)",
                "impact": "High Brand Affinity",
                "detail": "Long-term relationships indicate deep service integration and high satisfaction."
            })
            
        # 3. Internet Service & Tech Support
        internet = str(customer_dict.get('InternetService', ''))
        tech_support = str(customer_dict.get('TechSupport', ''))
        online_sec = str(customer_dict.get('OnlineSecurity', ''))
        
        if internet == 'Fiber optic':
            if tech_support == 'No' or online_sec == 'No':
                risk_factors.append({
                    "factor": "Fiber Optic without Technical Support / Security",
                    "impact": "Frequent Service Dissatisfaction Risk",
                    "detail": "Fiber users pay premium rates; without security/support, technical glitches drive fast defection."
                })
                recommendations.append("Provide a complimentary 6-month Cyber Security & Priority Tech Support add-on.")
        
        if tech_support == 'Yes':
            positive_factors.append({
                "factor": "Active Tech Support Package",
                "impact": "High Operational Satisfaction",
                "detail": "Customers receiving rapid assistance are 2.8x less likely to churn."
            })
        if online_sec == 'Yes':
            positive_factors.append({
                "factor": "Online Security Enabled",
                "impact": "Protective Value Anchor",
                "detail": "Digital safety features build stickiness and perceived utility."
            })
            
        # 4. Payment Method
        pay_method = str(customer_dict.get('PaymentMethod', ''))
        if pay_method == 'Electronic check':
            risk_factors.append({
                "factor": "Electronic Check Billing",
                "impact": "High Churn Association (~45% Churn Rate)",
                "detail": "Manual payment friction often correlates with billing disputes and active price shopping."
            })
            recommendations.append("Promote Auto-pay enrollment via Credit Card or Bank Transfer with a $5 monthly bill credit.")
        elif 'automatic' in pay_method.lower():
            positive_factors.append({
                "factor": f"Automated Billing ({pay_method})",
                "impact": "Frictionless Payment Stream",
                "detail": "Autopay reduces active monthly bill scrutinization and unintended non-payment drops."
            })
            
        # 5. Pricing Burden Analysis
        monthly_charges = float(customer_dict.get('MonthlyCharges', 0))
        if monthly_charges > 80.0:
            risk_factors.append({
                "factor": f"High Monthly Charges (${monthly_charges:.2f}/mo)",
                "impact": "Premium Price Sensitivity",
                "detail": "Bills above $80/month fall into the top pricing tier where competitors actively target customers."
            })
            recommendations.append("Conduct a plan audit to bundle entertainment and streaming services into a cost-optimized package.")
        elif monthly_charges < 40.0:
            positive_factors.append({
                "factor": f"Economical Monthly Bill (${monthly_charges:.2f}/mo)",
                "impact": "Low Budget Friction",
                "detail": "Affordable baseline service reduces motivation to defect to competing carriers."
            })

        # 6. Default recommendations by risk category
        if risk_level == "HIGH RISK":
            recommendations.append("Dispatch high-priority retention ticket to Senior Account Management team.")
            recommendations.append("Prepare bespoke contract extension package with guaranteed price freeze.")
        elif risk_level == "MEDIUM RISK":
            recommendations.append("Trigger targeted in-app loyalty perks and engagement surveys.")
            recommendations.append("Highlight underutilized features (backup, cloud storage) to increase perceived value.")
        else:
            recommendations.append("Maintain standard service quality and invite customer to join customer advocacy/referral program.")
            recommendations.append("Offer seasonal upgrade previews and milestone anniversary gifts.")

        return {
            "prediction": int(churn_prob > 0.5),
            "prediction_label": prediction_label,
            "churn_probability": churn_prob,
            "churn_probability_pct": churn_prob_pct,
            "stay_probability_pct": stay_prob_pct,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_badge": risk_badge,
            "risk_factors": risk_factors,
            "positive_factors": positive_factors,
            "retention_recommendations": recommendations,
            "customer_summary": {
                "tenure_months": int(tenure),
                "contract": contract,
                "monthly_charges": monthly_charges,
                "internet_service": internet,
                "payment_method": pay_method
            }
        }

if __name__ == "__main__":
    predictor = ChurnPredictor()
    
    # Test High-risk customer
    high_risk_customer = {
        'gender': 'Male',
        'SeniorCitizen': 1,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 95.5,
        'TotalCharges': 191.0
    }
    
    res = predictor.predict_single(high_risk_customer)
    print("--- Test High Risk Customer ---")
    print(f"Prediction: {res['prediction_label']}")
    print(f"Probability: {res['churn_probability_pct']}% | Risk Level: {res['risk_level']}")
    print("Top Risk Factors:", [f['factor'] for f in res['risk_factors']])
    print("Recommendations:", res['retention_recommendations'][:2])
