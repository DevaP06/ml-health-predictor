import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features"]

st.title("Heart Health Risk Predictor")

st.markdown("### Please fill in the following information:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female"])

chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])
high_blood_pressure = st.selectbox("High Blood Pressure", ["Yes", "No"])
irregular_heartbeat = st.selectbox("Irregular Heartbeat", ["Yes", "No"])
shortness_of_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
fatigue_weakness = st.selectbox("Fatigue or Weakness", ["Yes", "No"])
dizziness = st.selectbox("Dizziness", ["Yes", "No"])
swelling_edema = st.selectbox("Swelling or Edema", ["Yes", "No"])
neck_jaw_pain = st.selectbox("Neck or Jaw Pain", ["Yes", "No"])
excessive_sweating = st.selectbox("Excessive Sweating", ["Yes", "No"])
persistent_cough = st.selectbox("Persistent Cough", ["Yes", "No"])
nausea_vomiting = st.selectbox("Nausea or Vomiting", ["Yes", "No"])
chest_discomfort = st.selectbox("Chest Discomfort", ["Yes", "No"])
cold_hands_feet = st.selectbox("Cold Hands or Feet", ["Yes", "No"])
snoring_sleep_apnea = st.selectbox("Snoring or Sleep Apnea", ["Yes", "No"])
anxiety_doom = st.selectbox("Anxiety or Sense of Doom", ["Yes", "No"])
stroke_risk_percentage = st.number_input("Stroke Risk Percentage", min_value=0.0, max_value=100.0)

# Convert inputs to match model input
def convert(val): return 1 if val == "Yes" else 0

input_data = {
    "age": age,
    "gender": 1 if gender == "Male" else 0,
    "chest_pain": convert(chest_pain),
    "high_blood_pressure": convert(high_blood_pressure),
    "irregular_heartbeat": convert(irregular_heartbeat),
    "shortness_of_breath": convert(shortness_of_breath),
    "fatigue_weakness": convert(fatigue_weakness),
    "dizziness": convert(dizziness),
    "swelling_edema": convert(swelling_edema),
    "neck_jaw_pain": convert(neck_jaw_pain),
    "excessive_sweating": convert(excessive_sweating),
    "persistent_cough": convert(persistent_cough),
    "nausea_vomiting": convert(nausea_vomiting),
    "chest_discomfort": convert(chest_discomfort),
    "cold_hands_feet": convert(cold_hands_feet),
    "snoring_sleep_apnea": convert(snoring_sleep_apnea),
    "anxiety_doom": convert(anxiety_doom),
    "stroke_risk_percentage": stroke_risk_percentage
}

input_df = pd.DataFrame([input_data])

# Ensure feature order matches model training
input_df = input_df[feature_names]

# Predict
if st.button("Predict Risk"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ The patient is at risk.")
    else:
        st.success("✅ The patient is not at risk.")
