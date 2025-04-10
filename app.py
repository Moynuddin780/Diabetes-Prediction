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
        age = st.number_input("Age", min_value=0, max_value=120, value=0.0)
        glucose = st.number_input("Glucose", value=0.0)
        ldl = st.number_input("LDL", value=0.0)
    with col2:
        pregnancies = st.number_input("Pregnancies", min_value=0, value=0.0)
        blood_pressure = st.number_input("Blood Pressure", value=0.0)
        hba1c = st.number_input("HbA1c", value=0.0)

    col3, col4 = st.columns(2)
    with col3:
        hdl = st.number_input("HDL", value=0.0)
        waist = st.number_input("Waist Circumference", value=0.0)
    with col4:
        triglycerides = st.number_input("Triglycerides", value=0.0)
        hip = st.number_input("Hip Circumference", value=0.0)

    col5, col6 = st.columns(2)
    with col5:
        whr = st.number_input("WHR", value=0.0)
        family_history = st.selectbox("Family History (Yes=1 / No=0)", [1, 0])
    with col6:
        diet_type = st.selectbox("Diet Type (Healthy=1 / Unhealthy=0)", [1, 0])
        hypertension = st.selectbox("Hypertension (Yes=1 / No=0)", [1, 0])

    medication_use = st.selectbox("Medication Use (Yes=1 / No=0)", [1, 0])

    # 🔘 Predict Button
    submit = st.form_submit_button("Predict")

    if submit:
        input_data = np.array([
            age, pregnancies, 25.6, glucose, blood_pressure, hba1c,
            ldl, hdl, triglycerides, waist, hip, whr,
            family_history, diet_type, hypertension, medication_use
        ])

        input_scaled = scaler.transform(input_data.reshape(1, -1))
        prediction = model.predict(input_scaled)

        st.markdown("---")
        if prediction[0] == 1:
            st.markdown("<h3 style='color: #ff6666; text-align: center;'>⚠️ **You may be Diabetic.**</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 1.2rem; text-align: center;'>Take care 💙 — with proper treatment and lifestyle, you can stay healthy.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: #66bb6a; text-align: center;'>✅ **You are not Diabetic.**</h3>", unsafe_allow_html=True)
            st.balloons()
            st.markdown("<p style='font-size: 1.2rem; text-align: center;'>🎉 Congrats! Keep maintaining a healthy lifestyle.</p>", unsafe_allow_html=True)
