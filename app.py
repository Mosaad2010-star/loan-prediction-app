import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("Loan.pkl")

# Encoders for categorical inputs
gender_map = {
    "Male": 1,
    "Female": 0
}

marital_map = {
    "Married": 1,
    "Single": 0
}

employment_map = {
    "Employed": 0,
    "Self-employed": 1,
    "Unemployed": 2
}

loan_purpose_map = {
    "Car": 0,
    "Education": 1,
    "Business": 2,
    "Personal": 3,
    "Home Improvement": 4
}

# App title
st.title("Loan Approval Prediction")

# User inputs
age = st.slider("Age", 21, 65, 30)

gender = st.selectbox("Gender", ["Male", "Female"])

marital_status = st.selectbox("Marital Status", ["Single", "Married"])

income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000, step=500)

loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=50000, value=10000, step=500)

credit_score = st.slider("Credit Score (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed"])

loan_purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Business", "Personal", "Home Improvement"])

# Prepare input data
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_map[gender]],
    "Marital_Status": [marital_map[marital_status]],
    "Income": [income],
    "Loan_Amount": [loan_amount],
    "Credit_Score": [credit_score],
    "Employment_Status": [employment_map[employment_status]],
    "Loan_Purpose": [loan_purpose_map[loan_purpose]]
})

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"Loan Approved! (Confidence: {probability:.2f})")
    else:
        st.error(f"Loan Rejected. (Confidence: {probability:.2f})")

# Footer
st.markdown("---")
st.markdown("### Developed by **Mosaad Hendam**")
