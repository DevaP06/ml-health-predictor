import pandas as pd

df = pd.read_csv("dataset.csv")

print("\n📊 Initial dataset shape:", df.shape)

# Check for missing values
print("\n🔍 Missing values per column:")
print(df.isnull().sum())

# Check unique values in object columns
print("\n🧠 Unique values in categorical (object) columns:")
for col in df.select_dtypes(include="object").columns:
    print(f"{col}: {df[col].unique()[:10]}")  # print first 10 unique values
