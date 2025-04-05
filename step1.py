import pandas as pd

# Load the dataset
df = pd.read_csv("dataset.csv")

# View the first few rows
print(df.head())

# Basic info about columns and data types
print(df.info())

# Check for missing values
print(df.isnull().sum())
# Drop or fill missing values
df = df.dropna()  # or df.fillna(method='ffill')

# Encode categorical variables
df = pd.get_dummies(df)

# Separate features and target
X = df.drop("target_column", axis=1)
y = df["target_column"]

