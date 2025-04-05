import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset.csv")

# Clean invalid data
df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))  # Convert to numeric
df = df.dropna()  # Remove any row with NaN

# Split features and target
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
        "features": list(X.columns)
    }, f)

print("✅ Model training complete. Model saved as model.pkl")


