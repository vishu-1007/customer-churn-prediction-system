"""
app/streamlit_app.py
Production-grade Streamlit dashboard.

Run: streamlit run app/streamlit_app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from src.utils import load_config
from src.predict import load_inference_artifacts, predict_single

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction | Virtusa",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center;
        border: 1px solid #e9ecef;
    }
    .risk-high   { color: #E24B4A; font-weight: bold; font-size: 1.3rem; }
    .risk-medium { color: #BA7517; font-weight: bold; font-size: 1.3rem; }
    .risk-low    { color: #1D9E75; font-weight: bold; font-size: 1.3rem; }
    .section-header { font-size: 1.1rem; font-weight: 600;
                      border-bottom: 2px solid #534AB7;
                      padding-bottom: 4px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ───────────────────────────────────────────
@st.cache_resource
def load_all():
    config = load_config()
    model, scaler, feature_names, explainer = load_inference_artifacts(config)
    return config, model, scaler, feature_names, explainer

config, model, scaler, feature_names, explainer = load_all()


# ── Header ───────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.title("📉 Customer Churn Prediction System")
    st.caption("XGBoost + SHAP Explainability | IBM Telco Dataset | Production Model")
with col_h2:
    st.metric("Model", "XGBoost (Tuned)")
    st.metric("Explainability", "SHAP")

st.divider()


# ── Sidebar — Customer inputs ─────────────────────────────────
st.sidebar.header("🧾 Customer Profile")
st.sidebar.caption("Fill in customer details to get churn prediction")

with st.sidebar.expander("📋 Account Info", expanded=True):
    tenure           = st.slider("Tenure (months)", 0, 72, 12)
    contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method   = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless        = st.checkbox("Paperless Billing", value=True)

with st.sidebar.expander("💰 Charges", expanded=True):
    monthly_charges  = st.slider("Monthly Charges ($)", 18, 120, 70)
    total_charges    = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                       float(tenure * monthly_charges))

with st.sidebar.expander("🌐 Services", expanded=False):
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security  = st.checkbox("Online Security")
    tech_support     = st.checkbox("Tech Support")
    streaming_tv     = st.checkbox("Streaming TV")
    streaming_movies = st.checkbox("Streaming Movies")
    online_backup    = st.checkbox("Online Backup")
    device_protect   = st.checkbox("Device Protection")

with st.sidebar.expander("👤 Demographics", expanded=False):
    senior_citizen   = st.checkbox("Senior Citizen")
    partner          = st.checkbox("Has Partner")
    dependents       = st.checkbox("Has Dependents")

predict_btn = st.sidebar.button("🔮 Predict Churn", type="primary", use_container_width=True)


# ── Prediction logic ─────────────────────────────────────────
if predict_btn:
    raw_input = {
        "tenure": tenure, "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": int(senior_citizen), "gender": 0,
        "Partner": int(partner), "Dependents": int(dependents),
        "PhoneService": 1, "MultipleLines": 0,
        "OnlineSecurity": int(online_security),
        "OnlineBackup": int(online_backup),
        "DeviceProtection": int(device_protect),
        "TechSupport": int(tech_support),
        "StreamingTV": int(streaming_tv), "StreamingMovies": int(streaming_movies),
        "PaperlessBilling": int(paperless),
        "Contract": contract, "InternetService": internet_service,
        "PaymentMethod": payment_method,
    }

    with st.spinner("Calculating churn probability..."):
        result = predict_single(raw_input, model, scaler, feature_names, explainer, config)

    prob  = result["churn_probability"]
    risk  = result["risk_level"]
    shap_vals = result["shap_values"]

    # ── Results row ──────────────────────────────────────────
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Churn Probability", f"{prob:.1f}%")
    r2.metric("Risk Level", risk)
    r3.metric("Contract", contract)
    r4.metric("Tenure", f"{tenure} months")

    # Risk banner
    risk_colors = {"HIGH": "#E24B4A", "MEDIUM": "#BA7517", "LOW": "#1D9E75"}
    risk_icons  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    st.markdown(
        f"<div style='background:{risk_colors[risk]}22; border-left: 4px solid {risk_colors[risk]}; "
        f"padding:12px 16px; border-radius:6px; margin:12px 0;'>"
        f"<strong>{risk_icons[risk]} {risk} CHURN RISK</strong> — "
        f"This customer has a <strong>{prob:.1f}%</strong> probability of churning.</div>",
        unsafe_allow_html=True
    )

    st.divider()

    # ── Two column layout ────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<p class="section-header">📊 Churn Probability Gauge</p>',
                    unsafe_allow_html=True)
        fig_g, ax_g = plt.subplots(figsize=(7, 1.8))
        color = risk_colors[risk]
        ax_g.barh(["Risk"], [prob], color=color, height=0.45)
        ax_g.barh(["Risk"], [100 - prob], left=[prob],
                  color="#F1EFE8", height=0.45)
        ax_g.axvline(50, color="#888780", linewidth=1.5, linestyle="--", alpha=0.7)
        ax_g.set_xlim(0, 100)
        ax_g.set_xlabel("Churn Probability (%)")
        ax_g.text(min(prob + 1, 95), 0, f" {prob:.1f}%",
                  va="center", fontweight="bold", color=color)
        ax_g.set_yticks([])
        ax_g.spines[["top", "right", "left"]].set_visible(False)
        fig_g.patch.set_alpha(0)
        ax_g.patch.set_alpha(0)
        st.pyplot(fig_g, use_container_width=True)
        plt.close()

    with right:
        st.markdown('<p class="section-header">🔍 SHAP Feature Impact</p>',
                    unsafe_allow_html=True)

        shap_series = pd.Series(shap_vals, index=feature_names)
        top10 = shap_series.abs().nlargest(10)
        top10_vals = shap_series[top10.index]

        fig_s, ax_s = plt.subplots(figsize=(7, 4))
        colors_bar = ["#E24B4A" if v > 0 else "#1D9E75" for v in top10_vals.values]
        ax_s.barh(range(len(top10_vals)), top10_vals.values,
                  color=colors_bar, edgecolor="white")
        ax_s.set_yticks(range(len(top10_vals)))
        ax_s.set_yticklabels(top10_vals.index, fontsize=9)
        ax_s.axvline(0, color="#444441", linewidth=0.8)
        ax_s.set_xlabel("SHAP Value")
        ax_s.set_title("Top 10 Features Driving This Prediction", fontsize=10)
        ax_s.invert_yaxis()
        fig_s.patch.set_alpha(0)
        ax_s.patch.set_alpha(0)
        st.pyplot(fig_s, use_container_width=True)
        plt.close()
        st.caption("🔴 Pushes toward churn   |   🟢 Pushes toward staying")

    st.divider()

    # ── Business recommendation ──────────────────────────────
    st.markdown('<p class="section-header">💡 Business Recommendation</p>',
                unsafe_allow_html=True)

    if risk == "HIGH":
        st.error(
            "**⚠️ Immediate Retention Action Required**\n\n"
            "- Offer contract upgrade incentive (month-to-month → annual)\n"
            "- Provide 20–30% loyalty discount on monthly charges\n"
            "- Assign dedicated account manager for personal outreach\n"
            "- Offer free service add-on (tech support / security) for 3 months"
        )
    elif risk == "MEDIUM":
        st.warning(
            "**📋 Proactive Engagement Recommended**\n\n"
            "- Send personalised retention email within 48 hours\n"
            "- Offer discounted service bundle upgrade\n"
            "- Schedule NPS survey to identify pain points"
        )
    else:
        st.success(
            "**✅ Low Risk — Standard Engagement**\n\n"
            "- No immediate retention action required\n"
            "- Include in quarterly NPS survey cycle\n"
            "- Flag for upsell opportunity (high satisfaction + long tenure)"
        )

    st.divider()
    st.caption(
        "Model: XGBoost (Optuna tuned) | Dataset: IBM Telco | "
        "Explainability: SHAP TreeExplainer | "
        f"Project: {config['project']['name']} v{config['project']['version']}"
    )

else:
    # ── Landing state ────────────────────────────────────────
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn** to get started.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🧠 Model")
        st.write("XGBoost tuned with Optuna — 50-trial Bayesian search over 7 hyperparameters")
    with c2:
        st.markdown("### 🔍 Explainability")
        st.write("SHAP TreeExplainer — shows exactly which features drive each individual prediction")
    with c3:
        st.markdown("### 📊 Performance")
        st.write("~87% F1-score on test set | ROC-AUC ~91% | Evaluated on original class distribution")
