import pandas as pd
import os
from pathlib import Path

# Project directory
PROJECT_PATH = Path(__file__).resolve().parent.parent

# CERT r4.2 dataset directory
DATASET_PATH = Path(
    os.environ.get(
        "CERT_DATASET_PATH",
        str(PROJECT_PATH / "dataset" / "r4.2")
    )
)

# Processed output directory
OUTPUT_PATH = PROJECT_PATH / "processed"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Files to preprocess
FILES = [
    "logon.csv",
    "device.csv",
    "http.csv",
    "email.csv",
    "file.csv",
]

# Read 50,000 rows at a time
CHUNK_SIZE = 50000

DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

for filename in FILES:

    print(f"\nProcessing: {filename}", flush=True)

    matches = list(DATASET_PATH.rglob(filename))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {filename}, found {len(matches)}"
        )

    input_file = matches[0]
    output_file = OUTPUT_PATH / f"cleaned_{filename}"

    total_records = 0
    invalid_dates = 0
    cleaned_records = 0
    after_hours_records = 0
    weekend_records = 0

    first_chunk = True
    for chunk in pd.read_csv(
        input_file,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        total_records += len(chunk)

        chunk["date"] = pd.to_datetime(
            chunk["date"],
            format=DATE_FORMAT,
            errors="coerce",
        )

        invalid_dates += chunk["date"].isna().sum()

        chunk = chunk.drop(
    columns=["content"],
    errors="ignore"
)

        # Remove records without a user
        chunk = chunk.dropna(subset=["user"])

        # After-hours activity: before 6 AM or after 6 PM
        chunk["is_after_hours"] = (
            (chunk["date"].dt.hour < 6)
            | (chunk["date"].dt.hour >= 18)
        )

        # Saturday or Sunday
        chunk["is_weekend"] = (
            chunk["date"].dt.dayofweek >= 5
        )

        cleaned_records += len(chunk)

        after_hours_records += chunk["is_after_hours"].sum()
        weekend_records += chunk["is_weekend"].sum()

        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False

        if total_records % 500000 == 0:
            print(
                f"{filename}: {total_records:,} records read",
                flush=True,
            )
    
    print(f"Original records: {total_records:,}")
    print(f"Invalid dates: {invalid_dates:,}")
    print(f"Cleaned records: {cleaned_records:,}")
    print(f"After-hours records: {after_hours_records:,}")
    print(f"Weekend records: {weekend_records:,}")
    print(f"Saved to: {output_file}")

print("\n================================")
print("DATA PREPROCESSING COMPLETED")
print("================================")