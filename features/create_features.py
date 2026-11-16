import pandas as pd
import os
from urllib.parse import urlparse

folder = "processed"

# Load datasets
logon = pd.read_csv(os.path.join(folder, "cleaned_logon.csv"))
device = pd.read_csv(os.path.join(folder, "cleaned_device.csv"))
http = pd.read_csv(os.path.join(folder, "cleaned_http.csv"))

# Convert dates
for df in [logon, device, http]:
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.date


# 1. LOGON FEATURES

logon["after_hours_logon"] = (
    (logon["activity"] == "Logon") &
    (logon["is_after_hours"])
)

logon_features = logon.groupby(["user", "day"]).agg(
    total_logon_events=("activity", "count"),
    logon_count=("activity", lambda x: (x == "Logon").sum()),
    logoff_count=("activity", lambda x: (x == "Logoff").sum()),
    after_hours_logons=("after_hours_logon", "sum"),
    unique_logon_pcs=("pc", "nunique")
).reset_index()


# 2. USB DEVICE FEATURES

device["after_hours_usb_connect"] = (
    (device["activity"] == "Connect") &
    (device["is_after_hours"])
)

device_features = device.groupby(["user", "day"]).agg(
    usb_events=("activity", "count"),
    usb_connect_count=("activity", lambda x: (x == "Connect").sum()),
    usb_disconnect_count=("activity", lambda x: (x == "Disconnect").sum()),
    after_hours_usb=("after_hours_usb_connect", "sum"),
    unique_device_pcs=("pc", "nunique")
).reset_index()


# 3. HTTP FEATURES
http["domain"] = http["url"].apply(
    lambda url: urlparse(str(url)).netloc.lower()
)

http_features = http.groupby(["user", "day"]).agg(
    http_requests=("url", "count"),
    after_hours_http=("is_after_hours", "sum"),
    unique_urls=("url", "nunique"),
    unique_domains=("domain", "nunique")
).reset_index()


# 4. COMBINE ALL FEATURES
features = logon_features.merge(
    device_features,
    on=["user", "day"],
    how="outer"
)

features = features.merge(
    http_features,
    on=["user", "day"],
    how="outer"
)

# Replace missing event counts with zero
count_columns = features.select_dtypes(
    include="number"
).columns

features[count_columns] = features[count_columns].fillna(0)

# Sort records
features = features.sort_values(
    ["user", "day"]
)

# Save output
output_path = os.path.join(
    folder, "user_daily_features.csv"
)

features.to_csv(output_path, index=False)

print("\nFEATURE ENGINEERING COMPLETED")
print("Total user-day records:", len(features))
print("Total features:", len(features.columns))

print("\nFeature columns:")
print(features.columns.tolist())

print("\nFirst 10 records:")
print(features.head(10).to_string(index=False))

print("\nSaved to:", output_path)