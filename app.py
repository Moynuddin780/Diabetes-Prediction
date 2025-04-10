import streamlit as st
import numpy as np
import pickle

# 🎨 Page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# ✅ Load Model and Scaler
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("diabetes_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 💬 Title
st.markdown("<h1 style='text-align:center;'>🩺 Diabetes Prediction</h1>", unsafe_allow_html=True)

# 📥 Input Fields
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        glucose = st.number_input("Glucose", value=120.0)
        ldl = st.number_input("LDL", value=100.0)
        bmi = st.number_input("BMI", value=25.6)  # ✅ Added BMI here
    with col2:
        pregnancies = st.number_input("Pregnancies", min_value=0, value=2)
        blood_pressure = st.number_input("Blood Pressure", value=80.0)
        hba1c = st.number_input("HbA1c", value=6.5)

    col3, col4 = st.columns(2)
    with col3:
        hdl = st.number_input("HDL", value=50.0)
        waist = st.number_input("Waist Circumference", value=85.0)
    with col4:
        triglycerides = st.number_input("Triglycerides", value=150.0)
        hip = st.number_input("Hip Circumference", value=95.0)

    col5, col6 = st.columns(2)
    with col5:
        whr = st.number_input("WHR", value=0.8)
        family_history = st.selectbox("Family History (Yes=1 / No=0)", [1, 0])
    with col6:
        diet_type = st.selectbox("Diet Type (Healthy=1 / Unhealthy=0)", [1, 0])
        hypertension = st.selectbox("Hypertension (Yes=1 / No=0)", [1, 0])

    medication_use = st.selectbox("Medication Use_
