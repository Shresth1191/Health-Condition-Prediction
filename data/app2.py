import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================== Load Model and Preprocessing Objects ==============================
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')

# ============================== Streamlit Page Config ==============================
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🌼 Diabetes Risk Prediction")

st.markdown("""
### Input your health information:
Fill out the form below to predict whether you may have diabetes risk.
""")

# ============================== Input Form ==============================
with st.form("prediction_form"):
    age = st.slider("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    smoking_history = st.selectbox("Smoking History", ["never", "former", "current", "ever", "not current", "No Info"])
    bmi = st.number_input("BMI", 10.0, 60.0, 22.0)
    hba1c_level = st.number_input("HbA1c Level", 3.0, 20.0, 5.5)  # NEW FIELD
    blood_glucose_level = st.number_input("Blood Glucose Level", 50.0, 300.0, 90.0)
    submitted = st.form_submit_button("Predict")

# ============================== Prediction ==============================
if submitted:
    # Prepare input data as DataFrame
    input_df = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c_level],  # ADDED HERE
        "blood_glucose_level": [blood_glucose_level]
    })

    # Encode categorical features
    le_gender = encoders['gender']
    le_smoking = encoders['smoking_history']

    input_df['gender'] = le_gender.transform(input_df['gender'])
    input_df['smoking_history'] = le_smoking.transform(input_df['smoking_history'])

    # Scale numerical features
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Show result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"High Risk of Diabetes — Probability: {probability*100:.2f}%")
    else:
        st.success(f"Low Risk of Diabetes — Probability: {probability*100:.2f}%")

    st.info("Note: This prediction is for educational purposes and not a substitute for medical diagnosis.")
