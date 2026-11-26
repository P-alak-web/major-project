import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.train_xgboost import load_data, SPLIT_DATE

df = load_data()

df["month"] = df["day"].dt.to_period("M")

monthly = df.groupby("month").agg(
    employee_days=("label", "size"),
    employees=("user", "nunique"),
    malicious_days=("label", "sum")
)

monthly["normal_days"] = (
    monthly["employee_days"] - monthly["malicious_days"]
)

monthly["malicious_percentage"] = (
    monthly["malicious_days"]
    / monthly["employee_days"] * 100
).round(4)

insiders = (
    df[df["label"] == 1]
    .groupby("month")["user"]
    .nunique()
)

monthly["insiders"] = (
    insiders.reindex(monthly.index)
    .fillna(0)
    .astype(int)
)

print("\nMONTHLY DATA DISTRIBUTION")
print("=" * 85)
print(monthly.to_string())

print("\nExisting test split:", SPLIT_DATE.date())