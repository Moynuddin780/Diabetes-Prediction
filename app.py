import streamlit as st
import numpy as np
import joblib  # Assuming you're loading pre-trained model and scaler using joblib

# Load your trained model and scaler
model_classifier = joblib.load('path_to_your_trained_model.pkl')  # Replace with your actual model path
scaler = joblib.load('path_to_your_scaler.pkl')  # Replace with your actual scaler path

# Streamlit interface
st.title("Diabetes Prediction App")

# Input form
st.subheader("Enter details to predict if the person is diabetic:")

# Input fields for 16 features (example with placeholders, modify accordingly)
age = st.number_input("Age", min_value=1, max_value=120, value=45)
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.6)
glucose = st.number_input("Glucose", min_value=40, max_value=250, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)
hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=15.0, value=6.5)
ldl = st.number_input("LDL (mg/dL)", min_value=50, max_value=200, value=100)
hdl = st.number_input("HDL (mg/dL)", min_value=20, max_value=100, value=50)
triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=50, max_value=300, value=150)
waist_circumference = st.number_input("Waist Circumference (cm)", min_value=50, max_value=150, value=85)
hip_circumference = st.number_input("Hip Circumference (cm)", min_value=50, max_value=150, value=95)
wh_ratio = st.number_input("Waist-to-Hip Ratio (WHR)", min_value=0.5, max_value=1.0, value=0.8)
family_history = st.selectbox("Family History (0: No, 1: Yes)", [0, 1], index=0)
diet_type = st.selectbox("Diet Type (0: Non-Veg, 1: Veg)", [0, 1], index=0)
hypertension = st.selectbox("Hypertension (0: No, 1: Yes)", [0, 1], index=0)
medication_use = st.selectbox("Medication Use (0: No, 1: Yes)", [0, 1], index=0)

# Collect input data into a list
input_data = [
    age, pregnancies, bmi, glucose, blood_pressure, hba1c,
    ldl, hdl, triglycerides, waist_circumference, hip_circumference,
    wh_ratio, family_history, diet_type, hypertension, medication_use
]

# Ensure the input data has the correct number of features (16)
if len(input_data) != 16:
    st.error("Input data does not have the correct number of features (16). Please provide data for all features.")
else:
    # Convert input to NumPy array and reshape it for the model
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

    try:
        # Scale the input data using the previously fitted scaler
        input_scaled = scaler.transform(input_data_reshape)

        # Make the prediction using the trained model
        prediction = model_classifier.predict(input_scaled)

        # Display the prediction result
        if prediction[0] == 0:
            st.write("Prediction: No Diabetic")
        else:
            st.write("Prediction: Diabetic")

    except Exception as e:
        # Handle errors (e.g., scaler not properly fitted or input data issues)
        st.error(f"Error during prediction: {e}")
