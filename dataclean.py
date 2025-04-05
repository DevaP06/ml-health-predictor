import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("dataset.csv")  # Replace with your file path if needed

# Step 1: General cell cleaning
def clean_cell(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    for bad in ['--', '?', 'null', 'none', 'nan']:
        value = value.replace(bad, '')
    return value if value != '' else np.nan

df = df.applymap(clean_cell)

# Step 2: Clean 'gender' column
def clean_gender(val):
    if pd.isna(val):
        return np.nan
    val = val.lower()
    if 'm' in val:
        return 'Male'
    elif 'f' in val:
        return 'Female'
    return np.nan

df['gender'] = df['gender'].apply(clean_gender)

# Step 3: Convert numerical columns
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['stroke_risk_percentage'] = pd.to_numeric(df['stroke_risk_percentage'], errors='coerce')

# Step 4: Convert binary columns
binary_columns = df.columns.difference(['age', 'gender', 'stroke_risk_percentage', 'at_risk'])
for col in binary_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].apply(lambda x: 1.0 if x == 1 else (0.0 if x == 0 else np.nan))

# Convert 'at_risk' to float
df['at_risk'] = pd.to_numeric(df['at_risk'], errors='coerce')

# Step 5: Handle missing values
df['age'] = df['age'].fillna(df['age'].mean())
df['stroke_risk_percentage'] = df['stroke_risk_percentage'].fillna(df['stroke_risk_percentage'].mean())
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

# Fill binary columns with mode
for col in binary_columns.union(['at_risk']):
    df[col] = df[col].fillna(df[col].mode()[0])

# Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)
print("✅ Dataset cleaned and saved as 'cleaned_dataset.csv'")
