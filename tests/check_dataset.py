
import pandas as pd
import os

folder = "processed"

files = [
    "cleaned_logon.csv",
    "cleaned_device.csv",
    "cleaned_http.csv"
]

for filename in files:
    path = os.path.join(folder, filename)
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print("\n" + "=" * 45)
    print("FILE:", filename)
    print("Total records:", len(df))
    print("Unique users:", df["user"].nunique())
    print("Start date:", df["date"].min())
    print("End date:", df["date"].max())
    print("Missing dates:", df["date"].isna().sum())
    print("After-hours events:", df["is_after_hours"].sum())
