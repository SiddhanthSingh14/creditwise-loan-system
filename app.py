import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("CreditWise Loan Approval System")

st.write("Enter applicant details")

income = st.number_input("Applicant Income")
loan_amount = st.number_input("Loan Amount")
credit_history = st.selectbox("Credit History", [0,1])

if st.button("Predict Loan Approval"):

    features = np.array([[income, loan_amount, credit_history]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Not Approved")
