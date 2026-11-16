
import pandas as pd
from pathlib import Path

# Project folder location
project_path = Path(__file__).resolve().parent.parent

# Raw dataset location
dataset_path = project_path / "dataset" / "r1" / "r1"

# Output folder
output_path = project_path / "processed"
output_path.mkdir(exist_ok=True)

# Dataset files
files = ["logon.csv", "device.csv", "http.csv"]

for file in files:

    print("\nProcessing:", file)

    file_path = dataset_path / file

    # Load CSV files
    if file == "http.csv":
        df = pd.read_csv(
            file_path,
            header=None,
            names=["id", "date", "user", "pc", "url"]
        )
    else:
        df = pd.read_csv(file_path)

    # Convert date column
    df["date"] = pd.to_datetime(
        df["date"],
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce"
    )

    # Count invalid dates
    invalid_dates = df["date"].isna().sum()

    # Remove rows with invalid dates
    df = df.dropna(subset=["date"]).copy()

    # Identify after-hours activity
    df["is_after_hours"] = (
        (df["date"].dt.hour < 6) |
        (df["date"].dt.hour >= 18)
    )

    # Identify weekend activity
    df["is_weekend"] = df["date"].dt.dayofweek >= 5

    # Save processed dataset
    output_file = output_path / f"cleaned_{file}"
    df.to_csv(output_file, index=False)

    print("Original records:", invalid_dates + len(df))
    print("Invalid dates:", invalid_dates)
    print("Cleaned records:", len(df))
    print("After-hours records:", df["is_after_hours"].sum())
    print("Weekend records:", df["is_weekend"].sum())
    print("Saved to:", output_file)

print("\n================================")
print("DATA PREPROCESSING COMPLETED")
print("================================")
