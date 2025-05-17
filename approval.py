import streamlit as st
import pandas as pd
import joblib

# تحميل النموذج ومحولات التصنيف
model = joblib.load("loan_approval_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# عنوان التطبيق
st.title("🏦 Loan Approval Prediction App")

st.markdown("""
This app predicts whether a loan application will be **Approved** or **Rejected** based on user input.
""")

# واجهة إدخال البيانات
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_history = st.selectbox("Credit History", ["Good", "Bad"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

# زر التنبؤ
if st.button("Predict Loan Status"):
    # تجهيز البيانات
    input_data = pd.DataFrame({
        "ApplicantIncome": [income],
        "LoanAmount": [loan_amount],
        "Credit_History": [1 if credit_history == "Good" else 0],
        "Education": [education],
        "Married": [married],
        "Dependents": [dependents],
        "Self_Employed": [self_employed],
    })

    # تحويل البيانات التصنيفية
    for col in ["Education", "Married", "Self_Employed"]:
        encoder = label_encoders[col]
        input_data[col] = encoder.transform(input_data[col])

    # التنبؤ
    prediction = model.predict(input_data)[0]
    status = label_encoders["Loan_Status"].inverse_transform([prediction])[0]

    # النتيجة
    if status == "Approved":
        st.success("✅ Loan will likely be Approved!")
    else:
        st.error("❌ Loan will likely be Rejected.")
