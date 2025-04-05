import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Health Risk Predictor 💉")
st.write("Enter the details below to check if someone is at risk.")

# Dynamically create inputs based on model features
input_features = [
    'age', 'gender', 'chest_pain', 'high_blood_pressure', 'high_cholesterol',
    'diabetes', 'family_history', 'smoking', 'physical_inactivity',
    'stress', 'neck_jaw_pain', 'cold_hands_feet', 'stroke_risk_percentage'
]

user_input = {}
for feature in input_features:
    user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", step=1.0)

# When the user clicks Predict
if st.button("Predict Risk"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ The person is at risk!")
    else:
        st.success("✅ The person is not at risk.")
