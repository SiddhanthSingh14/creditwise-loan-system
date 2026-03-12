import streamlit as st
import pickle
import numpy as np

# -----------------------
# Load model and scaler
# -----------------------
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="CreditWise Loan System", layout="wide")

# -----------------------
# Custom Light + Cream UI
# -----------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #f7f3e9;
}

[data-testid="stHeader"] {
    background-color: #ffffff;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    height: 42px;
    width: 220px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Title
# -----------------------
st.title("🏦 CreditWise Loan Approval System")
st.write("Machine Learning powered loan approval prediction system.")

st.divider()

# -----------------------
# Detect model feature count
# -----------------------
try:
    feature_names = scaler.feature_names_in_
    num_features = len(feature_names)
except:
    num_features = scaler.n_features_in_
    feature_names = [f"Feature {i+1}" for i in range(num_features)]

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("Applicant Information")

inputs = []

for name in feature_names:
    value = st.sidebar.number_input(name, value=0.0)
    inputs.append(value)

# -----------------------
# Dashboard preview
# -----------------------
cols = st.columns(min(3, num_features))

for i in range(min(3, num_features)):
    cols[i].metric(feature_names[i], f"{inputs[i]}")

st.divider()

# -----------------------
# Prediction
# -----------------------
if st.button("🔍 Predict Loan Approval"):

    features = np.array(inputs).reshape(1, -1)

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_features)[0][1]
    else:
        prob = None

    st.divider()

    if prediction[0] == 1:
        st.success("✅ Loan Approved")

        if prob is not None:
            st.progress(float(prob))
            st.write(f"Approval Probability: {prob*100:.2f}%")

    else:
        st.error("❌ Loan Rejected")

        if prob is not None:
            st.progress(float(prob))
            st.write(f"Approval Probability: {prob*100:.2f}%")

st.divider()
st.caption("CreditWise ML Loan Prediction • Built with Streamlit")
