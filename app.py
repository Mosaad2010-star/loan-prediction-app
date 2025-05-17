import streamlit as st
import pandas as pd
import joblib

model = joblib.load("Loan.pkl")

st.title("Loan Approval Prediction")

age = st.slider("Age", 21, 65)
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Single", "Married"])
income = st.number_input("Monthly Income", 1000, 50000, step=500)
loan = st.number_input("Loan Amount", 1000, 50000, step=500)
score = st.slider("Credit Score", 0.0, 1.0, step=0.01)
job = st.selectbox("Employment", ["Employed", "Self-employed", "Unemployed"])
purpose = st.selectbox("Loan Purpose", ["Car", "Education", "Business", "Personal", "Home Improvement"])

data = pd.DataFrame({
    "Age": [age],
    "Gender": [1 if gender == "Male" else 0],
    "Marital_Status": [1 if married == "Married" else 0],
    "Income": [income],
    "Loan_Amount": [loan],
    "Credit_Score": [score],
    "Employment_Status": [0 if job == "Employed" else 1 if job == "Self-employed" else 2],
    "Loan_Purpose": [ ["Car", "Education", "Business", "Personal", "Home Improvement"].index(purpose) ]
})

if st.button("Predict"):
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][pred]
    if pred == 1:
        st.success(f"Loan Approved! (Confidence: {prob:.2f})")
    else:
        st.error(f"Loan Rejected. (Confidence: {prob:.2f})")

st.markdown("---")
st.markdown("### Developed by **Mosaad Hendam**")
