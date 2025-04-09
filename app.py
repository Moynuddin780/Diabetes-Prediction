import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model and scaler
try:
    with open("diabetes_model.pkl", "rb") as f:
        model_classifier = pickle.load(f)
    with open("diabetes_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Model or Scaler files not found. Please upload the files.")
    st.stop()  # Stop the execution if files are not found

# Streamlit user input
st.title("Diabetes Prediction")
st.write("Enter values for the following features:")

# Create two columns for input
col1, col2 = st.columns(2)

# First column input fields
with col1:
    age = st.number_input('Age', min_value=1, max_value=100)
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20)
    bmi = st.number_input('BMI', min_value=0.0, max_value=100.0)
    glucose = st.number_input('Glucose', min_value=0, max_value=300)
    blood_pressure = st.number_input('BloodPressure', min_value=0, max_value=200)
    hba1c = st.number_input('HbA1c', min_value=0.0, max_value=10.0)
    ldl = st.number_input('LDL', min_value=0, max_value=300)
    hdl = st.number_input('HDL', min_value=0, max_value=100)

# Second column input fields
with col2:
    triglycerides = st.number_input('Triglycerides', min_value=0, max_value=500)
    waist_circumference = st.number_input('WaistCircumference', min_value=0, max_value=200)
    hip_circumference = st.number_input('HipCircumference', min_value=0, max_value=200)
    whr = st.number_input('WHR', min_value=0.0, max_value=1.0)
    family_history = st.selectbox('FamilyHistory', options=[0, 1])
    diet_type = st.selectbox('DietType', options=[0, 1])
    hypertension = st.selectbox('Hypertension', options=[0, 1])
    medication_use = st.selectbox('MedicationUse', options=[0, 1])

# Gather the input data
input_data = np.array([age, pregnancies, bmi, glucose, blood_pressure, hba1c, ldl, hdl,
                       triglycerides, waist_circumference, hip_circumference, whr, family_history,
                       diet_type, hypertension, medication_use])

# Reshape and scale the input data
input_data_reshape = input_data.reshape(1, -1)
input_data_scaled = scaler.transform(input_data_reshape)

# Button to trigger prediction
if st.button('Predict'):
    # Predict the outcome using the model
    prediction = model_classifier.predict(input_data_scaled)
    
    # Display the result with personalized messages
    if prediction[0] == 0:
        st.success("Congratulations! You're not diabetic. Keep up the healthy lifestyle!")
    else:
        st.warning("It seems you are diabetic. Stay strong and take care of your health!")
