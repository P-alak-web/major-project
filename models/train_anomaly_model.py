import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

# Project paths
PROJECT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_PATH / "processed"
MODELS_PATH = PROJECT_PATH / "saved_models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)
INPUT_FILE = PROCESSED_PATH / "user_daily_features.csv"
OUTPUT_FILE = PROCESSED_PATH / "anomaly_results.csv"

# Load behavioural features
df = pd.read_csv(INPUT_FILE)

df["day"] = pd.to_datetime(
    df["day"],
    format="%Y-%m-%d",
    errors="raise"
)

# CERT r4.2 behavioural features
FEATURES = [
    "total_logon_events",
    "logon_count",
    "logoff_count",
    "after_hours_logons",
    "usb_events",
    "usb_connect_count",
    "usb_disconnect_count",
    "after_hours_usb",
    "http_requests",
    "after_hours_http",
    "email_count",
    "after_hours_email",
    "file_operations",
    "after_hours_file"
]

# Validate required columns
missing = [f for f in FEATURES if f not in df.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")

MINIMUM_RECORDS = 60
TRAIN_RATIO = 0.70

results = []
skipped_users = 0

# Train one model for each employee
for user, user_data in df.groupby("user"):

    user_data = user_data.sort_values("day").copy()

    if len(user_data) < MINIMUM_RECORDS:
        skipped_users += 1
        continue

    # Chronological 70/30 split
    split_index = int(len(user_data) * TRAIN_RATIO)

    train_data = user_data.iloc[:split_index]
    test_data = user_data.iloc[split_index:]

    X_train = train_data[FEATURES]
    X_test = test_data[FEATURES]

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)
    model_file = MODELS_PATH / f"{user}_isolation_forest.joblib"
    joblib.dump(model, model_file)

    predictions = model.predict(X_test)

    # Higher scores indicate greater anomaly
    anomaly_scores = -model.decision_function(X_test)

    user_results = test_data[["user", "day"]].copy()

    user_results["anomaly"] = (
        predictions == -1
    ).astype(int)

    user_results["anomaly_score"] = anomaly_scores

    results.append(user_results)

# Combine all employee results
if not results:
    raise ValueError(
        "No employees have sufficient historical records."
    )

final_results = pd.concat(
    results,
    ignore_index=True
)

# Relative ranking within each employee's test observations
final_results["risk_score"] = (
    final_results.groupby("user")["anomaly_score"]
    .rank(pct=True) * 100
).round(2)

# Save results
final_results.to_csv(
    OUTPUT_FILE,
    index=False
)

# Summary
print("\n========== ISOLATION FOREST RESULTS ==========")

print("Employees modelled:", final_results["user"].nunique())
print("Employees skipped:", skipped_users)
print("Records evaluated:", len(final_results))
print("Anomalous records:", int(final_results["anomaly"].sum()))

print(
    "Anomaly percentage:",
    round(final_results["anomaly"].mean() * 100, 2),
    "%"
)

print("\nTop 10 anomaly records:")

print(
    final_results.sort_values(
        "anomaly_score",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\nSaved to:", OUTPUT_FILE)