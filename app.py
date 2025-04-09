import streamlit as st
import numpy as np
import pickle
import base64

# === 🔹 Set Background Image ===
def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local("background.jpg")  # Make sure this file exists

# === 🔹 Load Model and Scaler ===
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("diabetes_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🩺 Diabetes Prediction App")

# === 🔹 Input Fields (Two Columns) ===
st.markdown("### Enter Patient Details:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0)
    bmi = st.number_input("BMI")
    glucose = st.number_input("Glucose")
    hba1c = st.number_input("HbA1c")

with col2:
    pregnancies = st.number_input("Pregnancies", min_value=0)
    blood_pressure = st.number_input("Blood Pressure")
    ldl = st.number_input("LDL")
    hdl = st.number_input("HDL")

# === 🔹 Last 2 Rows ===
triglycerides = st.number_input("Triglycerides")
waist = st.number_input("Waist Circumference")
hip = st.number_input("Hip Circumference")
whr = st.number_input("WHR")
family_history = st.selectbox("Family History", [0, 1])
diet_type = st.selectbox("Diet Type (0 = Normal, 1 = Controlled)", [0, 1])
hypertension = st.selectbox("Hypertension", [0, 1])
medication_use = st.selectbox("Medication Use", [0, 1])

# === 🔹 Predict Button ===
if st.button("Predict"):
    input_data = [
        glucose, blood_pressure, hba1c, ldl, hdl, triglycerides,
        waist, hip, whr, family_history, diet_type,
        hypertension, medication_use, age, pregnancies, bmi
    ]

    input_np = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_np)

    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("🎉 You are not diabetic. Keep living a healthy life!")
        st.image("not_diabetic.jpg", width=300)  # Add this image to the folder
    else:
        st.warning("⚠️ You might be diabetic. Take care and consult a doctor.")
        st.image("diabetic.jpg", width=300)  # Add this image to the folder
