import streamlit as st
import pandas as pd
import pickle

# Load model and feature names
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Manually specify feature names in the same order as training
feature_names = ["age", "gender", "bp", "chol", "smoker"]


# Manually define feature names (based on your dataset)
feature_names = ["age", "gender", "bp", "chol", "smoker"]


# Collect user inputs
age = st.number_input("Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female"])  # categorical
bp = st.number_input("Blood Pressure")
chol = st.number_input("Cholesterol Level")
smoker = st.selectbox("Do you smoke?", ["Yes", "No"])  # categorical

# Convert inputs into the proper format
# List of features used during training (exact order matters)
feature_names = ["age", "gender", "bp", "chol", "smoker"]

# Create dictionary from inputs
input_data = {
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "bp": bp,
    "chol": chol,
    "smoker": 1 if smoker == "Yes" else 0
}

# Create DataFrame and reorder columns to match model training
input_df = pd.DataFrame([input_data])
input_df = input_df[feature_names]  # ensure correct order


# Convert to DataFrame in correct order
input_df = pd.DataFrame([input_data])
input_df = input_df[feature_names]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {prediction}")
