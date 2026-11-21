import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "processed"

base = pd.read_csv(
    PROCESSED / "user_daily_features_advanced.csv"
)

http = pd.read_csv(
    PROCESSED / "http_daily_features.csv"
)

base["day"] = pd.to_datetime(base["day"]).dt.strftime("%Y-%m-%d")
http["day"] = pd.to_datetime(http["day"]).dt.strftime("%Y-%m-%d")

features = [
    "unique_domains",
    "new_domain_count",
    "cloud_storage_visits",
    "job_site_visits",
]

result = base.merge(
    http,
    on=["user", "day"],
    how="left",
    validate="one_to_one",
)

result[features] = result[features].fillna(0)

output = PROCESSED / "user_daily_features_advanced_v2.csv"

result.to_csv(output, index=False)

print("Final dataset shape:", result.shape)
print("Saved to:", output)
print(result[features].describe().to_string())