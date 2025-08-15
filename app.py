# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

# Load trained models and feature columns
model_at_risk = joblib.load("model_at_risk.pkl")
model_stroke_percentage = joblib.load("model_stroke_at_risk_percentage.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🧠 Stroke Risk Predictor")
st.markdown("Predict whether a patient is at risk and estimate stroke risk percentage.")

st.header("Enter Patient Details:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=["FEMALE", "MALE"])
gender_val = 1 if gender == "MALE" else 0

def binary_input(label):
    return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0

chest_pain = binary_input("Chest Pain")
high_blood_pressure = binary_input("High Blood Pressure")
irregular_heartbeat = binary_input("Irregular Heartbeat")
shortness_of_breath = binary_input("Shortness of Breath")
fatigue_weakness = binary_input("Fatigue/Weakness")
dizziness = binary_input("Dizziness")
swelling_edema = binary_input("Swelling/Edema")
neck_jaw_pain = binary_input("Neck/Jaw Pain")
excessive_sweating = binary_input("Excessive Sweating")
persistent_cough = binary_input("Persistent Cough")
nausea_vomiting = binary_input("Nausea/Vomiting")
chest_discomfort = binary_input("Chest Discomfort")
cold_hands_feet = binary_input("Cold Hands/Feet")
snoring_sleep_apnea = binary_input("Snoring/Sleep Apnea")
anxiety_doom = binary_input("Anxiety/Doom")

# Prepare dataframe for prediction
patient_data = {
    'age': age,
    'gender': gender_val,
    'chest_pain': chest_pain,
    'high_blood_pressure': high_blood_pressure,
    'irregular_heartbeat': irregular_heartbeat,
    'shortness_of_breath': shortness_of_breath,
    'fatigue_weakness': fatigue_weakness,
    'dizziness': dizziness,
    'swelling_edema': swelling_edema,
    'neck_jaw_pain': neck_jaw_pain,
    'excessive_sweating': excessive_sweating,
    'persistent_cough': persistent_cough,
    'nausea_vomiting': nausea_vomiting,
    'chest_discomfort': chest_discomfort,
    'cold_hands_feet': cold_hands_feet,
    'snoring_sleep_apnea': snoring_sleep_apnea,
    'anxiety_doom': anxiety_doom
}

patient_df = pd.DataFrame([patient_data])
patient_df = patient_df.reindex(columns=feature_columns, fill_value=0)

# Predict button
if st.button("Predict"):
    at_risk_pred = model_at_risk.predict(patient_df)[0]
    stroke_pred = model_stroke_percentage.predict(patient_df)[0]

    st.subheader("Prediction Results")
    st.write(f"**At Risk:** {'Yes' if at_risk_pred == 1 else 'No'}")
    st.write(f"**Stroke Risk Percentage:** {stroke_pred:.2f}%")
