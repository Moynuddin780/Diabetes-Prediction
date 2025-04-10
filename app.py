import streamlit as st
import numpy as np
import pickle

# 🎨 Page config
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# 🌈 Custom CSS for Output Section
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #d3eaf2, #b0d0d9, #a3b7c8); /* Lighter background colors */
        font-family: 'Segoe UI', sans-serif;
        color: #0d1b2a;
    }

    h1 {
        color: #01579b;
        text-align: center;
        margin-bottom: 2rem;
    }

    .stButton>button {
        background: linear-gradient(to right, #00acc1, #26c6da);
        color: white;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease-in-out;
        text-transform: uppercase;
    }

    .stButton>button:hover {
        background: linear-gradient(to right, #0097a7, #00acc1);
        transform: scale(1.03);
    }

    .stSelectbox>div>div,
    .stNumberInput>div>div {
        border-radius: 8px;
    }

    .stTextInput>div>div,
    .stNumberInput>div>div {
        background-color: rgba(255, 255, 255, 0.75);
        border-radius: 8px;
        color: #0d1b2a;
    }

    .stSelectbox, .stNumberInput {
        font-size: 1rem;
        color: #0d1b2a;
    }

    .stForm label {
        color: #0d1b2a;
        font-weight: bold;
    }

    /* Custom Styling for Output Messages */
    .stAlert {
        font-size: 1.2rem;  /* Increased font size */
        font-weight: bold;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .stAlert[data-testid="stAlert-danger"] {
        background-color: #ffcccc; /* Light red for background */
        border-left: 8px solid #d32f2f; /* Stronger red for the border */
        color: #8b0000; /* Darker red for text */
        font-size: 1.3rem; /* Slightly larger font */
        text-align: center;
    }

    .stAlert[data-testid="stAlert-success"] {
        background-color: #d4edda; /* Light green for background */
        border-left: 8px solid #388e3c; /* Stronger green for the border */
        color: #155724; /* Darker green for text */
        font-size: 1.3rem; /* Slightly larger font */
        text-align: center;
    }

    .stAlert[data-testid="stAlert-info"] {
        background-color: #cce5ff; /* Light blue for background */
        border-left: 8px solid #007bff; /* Stronger blue for the border */
        color: #004085; /* Darker blue for text */
        font-size: 1.3rem; /* Slightly larger font */
        text-align: center;
    }

    </style>
""", unsafe_allow_html=True)



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
            st.error("⚠️ **You may be Diabetic.**")
            st.info("Take care 💙 — with proper treatment and lifestyle, you can stay healthy.")
        else:
            st.success("✅ **You are not Diabetic.**")
            st.balloons()
            st.markdown("🎉 Congrats! Keep maintaining a healthy lifestyle.")

st.markdown("</div>", unsafe_allow_html=True)
