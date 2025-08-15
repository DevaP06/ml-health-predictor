# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_absolute_error
# import matplotlib.pyplot as plt

# RANDOM_STATE = 55

# df = pd.read_csv("dataset_new__final_cleaned.csv")

# # (MALE: 1, FEMALE: 0)
# features = [
#     'age', 'gender', 'chest_pain', 'high_blood_pressure',
#     'irregular_heartbeat', 'shortness_of_breath', 'fatigue_weakness',
#     'dizziness', 'swelling_edema', 'neck_jaw_pain', 'excessive_sweating',
#     'persistent_cough', 'nausea_vomiting', 'chest_discomfort', 'cold_hands_feet',
#     'snoring_sleep_apnea', 'anxiety_doom'
# ]

# df_selected = df[features + ['at_risk', 'stroke_risk_percentage']]

# X = df.drop(columns=['at_risk', 'stroke_risk_percentage'])
# y_at_risk = df['at_risk']
# y_stroke_risk_percentage = df['stroke_risk_percentage']

# X_temp, X_test, y_at_risk_temp, y_at_risk_test, y_stroke_risk_percentage_temp, y_stroke_risk_percentage_test = train_test_split(
#     X, y_at_risk, y_stroke_risk_percentage, test_size=0.2, random_state=RANDOM_STATE
# )

# X_train, X_val, y_at_risk_train, y_at_risk_val, y_stroke_risk_percentage_train, y_stroke_risk_percentage_val = train_test_split(
#     X_temp, y_at_risk_temp, y_stroke_risk_percentage_temp, test_size=0.2, random_state=RANDOM_STATE
# )

# model_at_risk = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
# model_at_risk.fit(X_train, y_at_risk_train)

# model_stroke_at_risk_percentage = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
# model_stroke_at_risk_percentage.fit(X_train, y_stroke_risk_percentage_train)

# acc_at_risk_train = accuracy_score(y_at_risk_train, model_at_risk.predict(X_train))
# acc_at_risk_val = accuracy_score(y_at_risk_val, model_at_risk.predict(X_val))
# acc_at_risk_test = accuracy_score(y_at_risk_test, model_at_risk.predict(X_test))

# J_at_risk_train = 1 - acc_at_risk_train
# J_at_risk_val = 1 - acc_at_risk_val
# J_at_risk_test = 1 - acc_at_risk_test

# mae_stroke_train = mean_absolute_error(y_stroke_risk_percentage_train, model_stroke_at_risk_percentage.predict(X_train))
# mae_stroke_val = mean_absolute_error(y_stroke_risk_percentage_val, model_stroke_at_risk_percentage.predict(X_val))
# mae_stroke_test = mean_absolute_error(y_stroke_risk_percentage_test, model_stroke_at_risk_percentage.predict(X_test))

# '''
# print("\n--- Model Performance ---")
# print(f"At Risk - Train Accuracy: {acc_at_risk_train:.4f}, Error: {J_at_risk_train:.4f}")
# print(f"At Risk - Validation Accuracy: {acc_at_risk_val:.4f}, Error: {J_at_risk_val:.4f}")
# print(f"At Risk - Test Accuracy: {acc_at_risk_test:.4f}, Error: {J_at_risk_test:.4f}")

# print(f"Stroke Risk Percentage - Train MAE: {mae_stroke_train:.4f}")
# print(f"Stroke Risk Percentage - Validation MAE: {mae_stroke_val:.4f}")
# print(f"Stroke Risk Percentage - Test MAE: {mae_stroke_test:.4f}")

# '''

# print("\nEnter Details for Prediction:")

# patient_age = float(input("Age: "))
# gender_input = input("Gender (0 = FEMALE, 1 = MALE): ").strip()
# while gender_input not in ['0', '1']:
#     gender_input = input("Invalid gender. Enter '0' for FEMALE or '1' for MALE: ").strip()
# gender_input = int(gender_input)

# chest_pain = int(input("Chest Pain (0 = No, 1 = Yes): "))
# high_blood_pressure = int(input("High Blood Pressure (0 = No, 1 = Yes): "))
# irregular_heartbeat = int(input("Irregular Heartbeat (0 = No, 1 = Yes): "))
# shortness_of_breath = int(input("Shortness of Breath (0 = No, 1 = Yes): "))
# fatigue_weakness = int(input("Fatigue/Weakness (0 = No, 1 = Yes): "))
# dizziness = int(input("Dizziness (0 = No, 1 = Yes): "))
# swelling_edema = int(input("Swelling/Edema (0 = No, 1 = Yes): "))
# neck_jaw_pain = int(input("Neck/Jaw Pain (0 = No, 1 = Yes): "))
# excessive_sweating = int(input("Excessive Sweating (0 = No, 1 = Yes): "))
# persistent_cough = int(input("Persistent Cough (0 = No, 1 = Yes): "))
# nausea_vomiting = int(input("Nausea/Vomiting (0 = No, 1 = Yes): "))
# chest_discomfort = int(input("Chest Discomfort (0 = No, 1 = Yes): "))
# cold_hands_feet = int(input("Cold Hands/Feet (0 = No, 1 = Yes): "))
# snoring_sleep_apnea = int(input("Snoring/Sleep Apnea (0 = No, 1 = Yes): "))
# anxiety_doom = int(input("Anxiety/Doom (0 = No, 1 = Yes): "))

# sample_input = {
#     'age': patient_age,
#     'gender': gender_input,  # MALE:1, FEMALE:0
#     'chest_pain': chest_pain,
#     'high_blood_pressure': high_blood_pressure,
#     'irregular_heartbeat': irregular_heartbeat,
#     'shortness_of_breath': shortness_of_breath,
#     'fatigue_weakness': fatigue_weakness,
#     'dizziness': dizziness,
#     'swelling_edema': swelling_edema,
#     'neck_jaw_pain': neck_jaw_pain,
#     'excessive_sweating': excessive_sweating,
#     'persistent_cough': persistent_cough,
#     'nausea_vomiting': nausea_vomiting,
#     'chest_discomfort': chest_discomfort,
#     'cold_hands_feet': cold_hands_feet,
#     'snoring_sleep_apnea': snoring_sleep_apnea,
#     'anxiety_doom': anxiety_doom
# }

# sample_df = pd.DataFrame([sample_input])
# sample_df = sample_df.reindex(columns=X.columns, fill_value=0)

# risk_pred = model_at_risk.predict(sample_df)[0]
# risk_pred_label = "Yes" if risk_pred == 1 else "No"
# print("\nPredicted Risk:", risk_pred_label)

# stroke_at_risk_percentage_pred = model_stroke_at_risk_percentage.predict(sample_df)[0]
# print(f"Predicted Risk Percentage: {stroke_at_risk_percentage_pred:.2f}%")



# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib  # <- new: for saving models

RANDOM_STATE = 55

# Load dataset
df = pd.read_csv("dataset_new__final_cleaned.csv")

features = [
    'age', 'gender', 'chest_pain', 'high_blood_pressure',
    'irregular_heartbeat', 'shortness_of_breath', 'fatigue_weakness',
    'dizziness', 'swelling_edema', 'neck_jaw_pain', 'excessive_sweating',
    'persistent_cough', 'nausea_vomiting', 'chest_discomfort', 'cold_hands_feet',
    'snoring_sleep_apnea', 'anxiety_doom'
]

X = df[features]
y_at_risk = df['at_risk']
y_stroke_risk_percentage = df['stroke_risk_percentage']

# Split data into train/val/test
X_temp, X_test, y_at_risk_temp, y_at_risk_test, y_stroke_temp, y_stroke_test = train_test_split(
    X, y_at_risk, y_stroke_risk_percentage, test_size=0.2, random_state=RANDOM_STATE
)
X_train, X_val, y_at_risk_train, y_at_risk_val, y_stroke_train, y_stroke_val = train_test_split(
    X_temp, y_at_risk_temp, y_stroke_temp, test_size=0.2, random_state=RANDOM_STATE
)

# Train models
model_at_risk = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
model_at_risk.fit(X_train, y_at_risk_train)

model_stroke_at_risk_percentage = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
model_stroke_at_risk_percentage.fit(X_train, y_stroke_train)

# Evaluate models (optional)
acc_val = accuracy_score(y_at_risk_val, model_at_risk.predict(X_val))
mae_val = mean_absolute_error(y_stroke_val, model_stroke_at_risk_percentage.predict(X_val))
print(f"Validation Accuracy (At Risk): {acc_val:.4f}")
print(f"Validation MAE (Stroke Risk %): {mae_val:.2f}")

# Save models and feature columns for app.py
joblib.dump(model_at_risk, "model_at_risk.pkl")
joblib.dump(model_stroke_at_risk_percentage, "model_stroke_at_risk_percentage.pkl")
joblib.dump(X.columns, "feature_columns.pkl")

print("Models and feature columns saved successfully!")
