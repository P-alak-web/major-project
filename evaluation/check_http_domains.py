import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTTP_FILE = ROOT / "processed" / "cleaned_http.csv"

previous_date = None
processed = 0
out_of_order = 0
examples = []

for chunk in pd.read_csv(
    HTTP_FILE,
    usecols=["date"],
    chunksize=200000
):
    dates = pd.to_datetime(chunk["date"], errors="coerce")

    if previous_date is not None:
        if dates.iloc[0] < previous_date:
            out_of_order += 1

            if len(examples) < 5:
                examples.append(
                    (previous_date, dates.iloc[0])
                )

    internal_disorder = (
        dates.diff().dt.total_seconds() < 0
    ).sum()

    out_of_order += int(internal_disorder)

    previous_date = dates.iloc[-1]
    processed += len(chunk)

    if processed % 2000000 == 0:
        print(f"Checked {processed:,} rows")

print("\nHTTP chronological-order audit")
print("Total rows:", processed)
print("Out-of-order transitions:", out_of_order)
print("Invalid dates: not checked in this version")
print("Boundary examples:", examples)

if out_of_order == 0:
    print("RESULT: HTTP records are chronologically ordered.")
else:
    print("RESULT: HTTP records are NOT chronologically ordered.")