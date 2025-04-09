import streamlit as st
import numpy as np
import pickle

# 💻 Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# 🌈 Custom CSS for background and style
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #dfe9f3, #ffffff);
        font-family: 'Segoe UI', sans-serif;
    }

    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    }

    h1 {
        color: #0d47a1;
        text-align: center;
    }

    .stButton>button {
        background-color: #0d47a1;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #1565c0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Load model and scaler
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("diabetes_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 🚀 App Title
st.markdown("<h1>🩺 Diabetes Prediction</h1>", unsafe_allow_html=True)

# 🧾 Input Form (3 rows, 2 columns each)
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        glucose = st.number_input("Glucose", value=120.0)
        ldl = st.number_input("LDL", value=100.0)
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
        family_history = st.selectbox("Family History", [0, 1])
    with col6:
        diet_type = st.selectbox("Diet Type", [0, 1])
        hypertension = st.selectbox("Hypertension", [0, 1])

    medication_use = st.selectbox("Medication Use", [0, 1])

    # 🧪 Predict Button
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = np.array([age, pregnancies, 25.6, glucose, blood_pressure, hba1c,
                               ldl, hdl, triglycerides, waist, hip, whr,
                               family_history, diet_type, hypertension, medication_use])

        input_scaled = scaler.transform(input_data.reshape(1, -1))
        prediction = model.predict(input_scaled)

        st.markdown("---")
        if prediction[0] == 1:
            st.error("⚠️ You may be Diabetic.\n\nPlease consult a medical professional.")
            st.info("Stay strong 💙, with proper care and lifestyle, you can lead a healthy life.")
        else:
            st.success("✅ You are not Diabetic.")
            st.balloons()
            st.markdown("🎉 Keep up the good health! Maintain your lifestyle and stay happy!")

