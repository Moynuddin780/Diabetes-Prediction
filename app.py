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
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("<h1>🩺 Diabetes Prediction</h1>", unsafe_allow_html=True)

# 📥 Input Fields
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        pregnancies = st.number_input("Pregnancies", min_value=0, value=2)
        bmi = st.number_input("BMI", value=25.6)
        glucose = st.number_input("Glucose", value=120.0)
        blood_pressure = st.number_input("Blood Pressure", value=80.0)
        hba1c = st.number_input("HbA1c", value=6.5)
        ldl = st.number_input("LDL", value=100.0)
        hdl = st.number_input("HDL", value=50.0)
    with col2:
        triglycerides = st.number_input("Triglycerides", value=150.0)
        waist = st.number_input("Waist Circumference", value=85.0)
        hip = st.number_input("Hip Circumference", value=95.0)
        whr = st.number_input("Waist-Hip Ratio (WHR)", value=0.8)
        family_history = st.selectbox("Family History (Yes=1 / No=0)", [1, 0])
        diet_type = st.selectbox("Diet Type (Healthy=1 / Unhealthy=0)", [1, 0])
        hypertension = st.selectbox("Hypertension (Yes=1 / No=0)", [1, 0])
        medication_use = st.selectbox("Medication Use (Yes=1 / No=0)", [1, 0])

    # 🔘 Predict Button
    submit = st.form_submit_button("Predict")

    if submit:
        input_data = np.array([
            age, pregnancies, bmi, glucose, blood_pressure, hba1c,
            ldl, hdl, triglycerides, waist, hip, whr,
            family_history, diet_type, hypertension, medication_use
        ])

        input_scaled = scaler.transform(input_data.reshape(1, -1))
        prediction = model.predict(input_scaled)

        st.markdown("---")
        if prediction[0] == 1:
            st.markdown("""
                <div style="font-size: 2rem; font-weight: bold; color: #b71c1c;">
                    ⚠️ You may be Diabetic.
                </div>
                <div style="font-size: 1.7rem; color: #0d47a1; margin-top: 1rem;">
                    Take care 💙 — with proper treatment and lifestyle, you can stay healthy.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="font-size: 2rem; font-weight: bold; color: #9ACBD0;">
                    ✅ You are not Diabetic.
                </div>
                <div style="font-size: 1.7rem; color: #48A6A7; margin-top: 1rem;">
                    🎉 Congrats! Keep maintaining a healthy lifestyle.
                </div>
            """, unsafe_allow_html=True)
            st.balloons()

st.markdown("</div>", unsafe_allow_html=True)
