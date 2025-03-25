import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
import pickle

# Load dataset
diabetes_dataset = pd.read_csv("diabetes_dataset.csv")

# Preprocessing
x = diabetes_dataset.iloc[:, :-1]
y = diabetes_dataset['Outcome']
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2, stratify=y)

# Train model
model_classifier = svm.SVC(kernel='linear')
model_classifier.fit(x_train, y_train)

# Streamlit UI
st.title("Diabetes Prediction App")

# Input Fields
st.sidebar.header("Enter Patient Details:")
pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, step=1)
glucose = st.sidebar.number_input("Glucose Level", 0, 200, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 150, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 100, step=1)
insulin = st.sidebar.number_input("Insulin", 0.0, 300.0, step=0.1)
bmi = st.sidebar.number_input("BMI", 0.0, 50.0, step=0.1)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5, step=0.01)
age = st.sidebar.number_input("Age", 0, 120, step=1)

# Prediction Button
if st.sidebar.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model_classifier.predict(input_scaled)

    # Output Result
    if prediction[0] == 0:
        st.success("✅ The patient is NOT Diabetic.")
    else:
        st.error("⚠️ The patient is Diabetic.")

# Show Model Accuracy
st.write("### Model Accuracy")
train_accuracy = accuracy_score(model_classifier.predict(x_train), y_train)
test_accuracy = accuracy_score(model_classifier.predict(x_test), y_test)
st.write(f"Train Accuracy: **{train_accuracy:.2f}**")
st.write(f"Test Accuracy: **{test_accuracy:.2f}**")
