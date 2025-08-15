# app.py

import pandas as pd
import joblib

def get_patient_input():
    print("\nEnter Patient Details:")

    age = float(input("Age: "))

    gender = input("Gender (0 = FEMALE, 1 = MALE): ").strip()
    while gender not in ['0', '1']:
        gender = input("Invalid input. Enter '0' for FEMALE or '1' for MALE: ").strip()
    gender = int(gender)

    def ask_binary(question):
        return int(input(f"{question} (0 = No, 1 = Yes): "))

    patient_data = {
        'age': age,
        'gender': gender,
        'chest_pain': ask_binary("Chest Pain"),
        'high_blood_pressure': ask_binary("High Blood Pressure"),
        'irregular_heartbeat': ask_binary("Irregular Heartbeat"),
        'shortness_of_breath': ask_binary("Shortness of Breath"),
        'fatigue_weakness': ask_binary("Fatigue/Weakness"),
        'dizziness': ask_binary("Dizziness"),
        'swelling_edema': ask_binary("Swelling/Edema"),
        'neck_jaw_pain': ask_binary("Neck/Jaw Pain"),
        'excessive_sweating': ask_binary("Excessive Sweating"),
        'persistent_cough': ask_binary("Persistent Cough"),
        'nausea_vomiting': ask_binary("Nausea/Vomiting"),
        'chest_discomfort': ask_binary("Chest Discomfort"),
        'cold_hands_feet': ask_binary("Cold Hands/Feet"),
        'snoring_sleep_apnea': ask_binary("Snoring/Sleep Apnea"),
        'anxiety_doom': ask_binary("Anxiety/Doom")
    }

    return pd.DataFrame([patient_data])

def main():
    # Load trained models and feature columns
    model_at_risk = joblib.load("model_at_risk.pkl")
    model_stroke_percentage = joblib.load("model_stroke_at_risk_percentage.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    patient_df = get_patient_input()
    # Ensure columns match training features
    patient_df = patient_df.reindex(columns=feature_columns, fill_value=0)

    # Predict At Risk
    at_risk_pred = model_at_risk.predict(patient_df)[0]
    print(f"\nPredicted At Risk: {'Yes' if at_risk_pred == 1 else 'No'}")

    # Predict Stroke Risk Percentage
    stroke_pred = model_stroke_percentage.predict(patient_df)[0]
    print(f"Predicted Stroke Risk Percentage: {stroke_pred:.2f}%")

if __name__ == "__main__":
    main()
