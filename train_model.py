import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset.csv")
print(f"📊 Initial dataset shape: {df.shape}")

# Define value cleaning functions
def clean_binary(val):
    if isinstance(val, str):
        if '1' in val:
            return 1
        elif '0' in val:
            return 0
    return pd.NA

def clean_gender(g):
    if isinstance(g, str):
        g = g.lower()
        if 'male' in g:
            return 1
        elif 'female' in g:
            return 0
    return pd.NA

# Clean binary columns
binary_cols = df.columns.drop(['age', 'gender', 'stroke_risk_percentage', 'at_risk'])
for col in binary_cols:
    df[col] = df[col].apply(clean_binary)

# Clean gender
df['gender'] = df['gender'].apply(clean_gender)

# Clean age and stroke risk
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['stroke_risk_percentage'] = pd.to_numeric(df['stroke_risk_percentage'], errors='coerce')

# Clean target column
df['at_risk'] = df['at_risk'].apply(clean_binary)
df['at_risk'] = df['at_risk'].astype("Int64")
df = df[df['at_risk'].isin([0, 1])]  # keep only rows with valid class labels


# Drop rows with any remaining NaNs
print(f"\n🔍 Missing values before dropping:\n{df.isna().sum()}")
df.dropna(inplace=True)
print(f"✅ Dataset shape after cleaning: {df.shape}")

# Split
X = df.drop("at_risk", axis=1)
y = df["at_risk"]

if len(df) == 0:
    raise ValueError("❌ No valid data left after preprocessing. Please check your dataset.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "features": list(X.columns)
    }, f)

print("✅ Model trained and saved successfully.")
