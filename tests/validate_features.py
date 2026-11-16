import pandas as pd

df = pd.read_csv("processed/user_daily_features.csv")

print("\n--- DATASET VALIDATION ---")

print("Total records:", len(df))
print("Total columns:", len(df.columns))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate user-day records:")
print(df.duplicated(subset=["user", "day"]).sum())

print("\nNegative numerical values:")
numeric = df.select_dtypes(include="number")
print((numeric < 0).sum())

print("\nFeature statistics:")
print(numeric.describe().round(2).to_string())

print("\nUsers:", df["user"].nunique())
print("Days:", df["day"].nunique())

print("\nValidation completed.")