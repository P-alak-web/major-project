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

    print("\n" + "=" * 50)
    print("FILE:", filename)

    if not os.path.exists(path):
        print("File not found!")
        continue

    df = pd.read_csv(path, nrows=5)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))