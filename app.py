import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load Model and Scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(
    page_title="CreditWise Loan Approval",
    layout="wide",
    page_icon="🏦"
)

# Header
st.title("🏦 CreditWise Loan Approval System")
st.markdown("Machine Learning powered loan approval prediction system.")

st.divider()

# Sidebar Inputs
st.sidebar.header("Applicant Information")

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0.0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0.0)

age = st.sidebar.slider("Age", 18, 70, 30)
dependents = st.sidebar.slider("Dependents", 0, 5, 0)

savings = st.sidebar.number_input("Savings", min_value=0.0)
existing_loans = st.sidebar.slider("Existing Loans", 0, 5, 0)

loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0)
loan_term = st.sidebar.slider("Loan Term (months)", 6, 360, 120)

collateral_value = st.sidebar.number_input("Collateral Value", min_value=0.0)

employment_status = st.sidebar.selectbox(
    "Employment Status",
    ["Employed", "Self-Employed", "Unemployed"]
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

# Convert categorical variables
employment_map = {
    "Employed": 0,
    "Self-Employed": 1,
    "Unemployed": 2
}

marital_map = {
    "Single": 0,
    "Married": 1
}

employment_val = employment_map[employment_status]
marital_val = marital_map[marital_status]

# Dashboard Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Applicant Income", f"${applicant_income:,.0f}")
col2.metric("Loan Amount", f"${loan_amount:,.0f}")
col3.metric("Savings", f"${savings:,.0f}")

st.divider()

# Prediction Button
if st.button("🔍 Predict Loan Approval"):

    features = np.array([[
        applicant_income,
        coapplicant_income,
        age,
        dependents,
        savings,
        existing_loans,
        loan_amount,
        loan_term,
        collateral_value,
        employment_val,
        marital_val
    ]])

    # Scale features
    scaled_features = scaler.transform(features)

    # Prediction
    prediction = model.predict(scaled_features)

    # Probability if available
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_features)[0][1]
    else:
        prob = None

    st.divider()

    # Result
    if prediction[0] == 1:
        st.success("✅ Loan Approved")

        if prob is not None:
            st.progress(float(prob))
            st.write(f"Approval Confidence: {prob*100:.2f}%")

    else:
        st.error("❌ Loan Rejected")

        if prob is not None:
            st.progress(float(prob))
            st.write(f"Approval Confidence: {prob*100:.2f}%")

st.divider()
st.caption("CreditWise ML Loan Prediction • Built with Streamlit")


