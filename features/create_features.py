import pandas as pd
from pathlib import Path
from urllib.parse import urlsplit
from collections import defaultdict

PROJECT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_PATH / "processed"

CHUNK_SIZE = 50000

# Stores daily aggregated features
daily_features = defaultdict(lambda: defaultdict(int))


def get_domain(url):
    """Extract hostname from a URL."""
    try:
        value = str(url)
        if "://" not in value:
            value = "http://" + value
        return urlsplit(value).hostname or ""
    except ValueError:
        return ""


def process_file(filename):
    path = PROCESSED_PATH / filename

    print(f"\nProcessing {filename}...", flush=True)

    if not path.exists():
        raise FileNotFoundError(path)

    total_rows = 0

    for chunk in pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunk["date"] = pd.to_datetime(
            chunk["date"],
            errors="coerce"
        )

        chunk = chunk.dropna(subset=["date", "user"])
        chunk["day"] = chunk["date"].dt.strftime("%Y-%m-%d")

        # Aggregate each chunk before updating daily totals
        grouped = chunk.groupby(["user", "day"], sort=False)

        if filename == "cleaned_logon.csv":

            chunk["logon_count"] = (
                chunk["activity"] == "Logon"
            ).astype(int)

            chunk["logoff_count"] = (
                chunk["activity"] == "Logoff"
            ).astype(int)

            chunk["after_hours_logons"] = (
                (chunk["activity"] == "Logon")
                & (chunk["is_after_hours"].astype(str) == "True")
            ).astype(int)

            aggregated = chunk.groupby(["user", "day"]).agg(
                total_logon_events=("activity", "size"),
                logon_count=("logon_count", "sum"),
                logoff_count=("logoff_count", "sum"),
                after_hours_logons=("after_hours_logons", "sum")
            )

        elif filename == "cleaned_device.csv":

            chunk["usb_connect_count"] = (
                chunk["activity"] == "Connect"
            ).astype(int)

            chunk["usb_disconnect_count"] = (
                chunk["activity"] == "Disconnect"
            ).astype(int)

            chunk["after_hours_usb"] = (
                (chunk["activity"] == "Connect")
                & (chunk["is_after_hours"].astype(str) == "True")
            ).astype(int)

            aggregated = chunk.groupby(["user", "day"]).agg(
                usb_events=("activity", "size"),
                usb_connect_count=("usb_connect_count", "sum"),
                usb_disconnect_count=("usb_disconnect_count", "sum"),
                after_hours_usb=("after_hours_usb", "sum")
            )

        elif filename == "cleaned_http.csv":

            aggregated = chunk.groupby(["user", "day"]).agg(
                http_requests=("id", "size"),
                after_hours_http=(
                    "is_after_hours",
                    lambda x: (x.astype(str) == "True").sum()
                )
            )

        elif filename == "cleaned_email.csv":

            aggregated = chunk.groupby(["user", "day"]).agg(
                email_count=("id", "size"),
                after_hours_email=(
                    "is_after_hours",
                    lambda x: (x.astype(str) == "True").sum()
                )
            )

        elif filename == "cleaned_file.csv":

            aggregated = chunk.groupby(["user", "day"]).agg(
                file_operations=("id", "size"),
                after_hours_file=(
                    "is_after_hours",
                    lambda x: (x.astype(str) == "True").sum()
                )
            )

        else:
            raise ValueError(f"Unknown file: {filename}")

        for (user, day), row in aggregated.iterrows():
            record = daily_features[(user, day)]

            for column, value in row.items():
                record[column] += int(value)

        total_rows += len(chunk)

        if total_rows % 500000 == 0:
            print(
                f"{filename}: {total_rows:,} rows processed",
                flush=True
            )

    print(f"Completed: {total_rows:,} rows")


FILES = [
    "cleaned_logon.csv",
    "cleaned_device.csv",
    "cleaned_http.csv",
    "cleaned_email.csv",
    "cleaned_file.csv"
]

for filename in FILES:
    process_file(filename)


# Convert aggregated results into a DataFrame
records = []

for (user, day), values in daily_features.items():
    record = {"user": user, "day": day}
    record.update(values)
    records.append(record)

features = pd.DataFrame(records)

# Fill missing activity counts with zero
feature_columns = [
    column for column in features.columns
    if column not in ["user", "day"]
]

features[feature_columns] = (
    features[feature_columns].fillna(0).astype(int)
)

features = features.sort_values(["user", "day"])

output_file = PROCESSED_PATH / "user_daily_features.csv"
features.to_csv(output_file, index=False)

print("\n================================")
print("FEATURE ENGINEERING COMPLETED")
print("================================")

print("Total user-day records:", len(features))
print("Total columns:", len(features.columns))

print("\nFeature columns:")
print(features.columns.tolist())

print("\nFirst 10 records:")
print(features.head(10).to_string(index=False))

print("\nSaved to:", output_file)