import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

# Load dataset
df = pd.read_csv("dataset.csv")  # make sure this file exists

# Print column names to verify
print("Columns in dataset:", df.columns)

# Preprocess
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
# df["smoker"] = df["smoker"].map({"Yes": 1, "No": 0})

# Define features and label — change 'target' to your actual target column name
X = df.drop("at_risk", axis=1)
y = df["at_risk"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model + feature names
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "features": list(X_train.columns)
    }, f)
