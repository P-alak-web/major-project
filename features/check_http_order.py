import pandas as pd

path = "processed/cleaned_http.csv"

ordered = True
previous = None
rows = 0

print("Checking HTTP dataset chronological order...")

for chunk in pd.read_csv(
    path,
    usecols=["date"],
    chunksize=200000
):
    dates = pd.to_datetime(chunk["date"])

    if not dates.is_monotonic_increasing:
        ordered = False

    if previous is not None and dates.iloc[0] < previous:
        ordered = False

    previous = dates.iloc[-1]
    rows += len(chunk)

print("\nTotal HTTP records:", rows)
print("Chronologically sorted:", ordered)