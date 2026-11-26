import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTTP_FILE = ROOT / "processed" / "cleaned_http.csv"

previous_date = None
processed = 0
out_of_order = 0
invalid_dates = 0

for chunk in pd.read_csv(
    HTTP_FILE,
    usecols=["date"],
    chunksize=200000
):
    dates = pd.to_datetime(chunk["date"], errors="coerce")

    invalid_dates += int(dates.isna().sum())

    if previous_date is not None and pd.notna(dates.iloc[0]):
        if dates.iloc[0] < previous_date:
            out_of_order += 1

    out_of_order += int((dates.diff().dt.total_seconds() < 0).sum())

    valid = dates.dropna()
    if not valid.empty:
        previous_date = valid.iloc[-1]

    processed += len(chunk)

    if processed % 2000000 == 0:
        print(f"Checked {processed:,} rows")

print("\nHTTP chronological-order audit")
print("Total rows:", processed)
print("Out-of-order transitions:", out_of_order)
print("Invalid dates:", invalid_dates)
