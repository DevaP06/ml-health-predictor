import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Load dataset
df = pd.read_csv("dataset.csv")

# Step 2: Clean bad values
# Replace '--', 'NULL', '0.0NULL', and other junk
df.replace(to_replace=[r"--", r"NULL", r"0.0NULL"], value=pd.NA, regex=True, inplace=True)

# Convert everything possible to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows where 'at_risk' is missing (target column must be valid)
df = df.dropna(subset=["at_risk"])

# Optional: Fill other missing values with median of the column
df = df.fillna(df.median(numeric_only=True))

# Step 3: Features and target
X = df.drop("at_risk", axis=1)
y = df["at_risk"]

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
