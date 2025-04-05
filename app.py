import streamlit as st
import pandas as pd
import pickle

# Load model and features
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features"]

# Example inputs (replace with st.number_input etc.)
input_data = {
    "age": st.number_input("Age"),
    "bp": st.number_input("Blood Pressure"),
    "cholesterol": st.number_input("Cholesterol")
}

# Make sure the input DataFrame matches training features
input_df = pd.DataFrame([input_data])
input_df = input_df[feature_names]  # Ensure correct column order

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Prediction: {prediction}")
