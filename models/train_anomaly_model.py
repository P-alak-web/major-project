
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Load daily features
df = pd.read_csv("processed/user_daily_features.csv")

df["day"] = pd.to_datetime(df["day"])

# Remove redundant feature
# unique_urls and unique_domains are identical in this dataset
features = [
    "total_logon_events",
    "logon_count",
    "logoff_count",
    "after_hours_logons",
    "unique_logon_pcs",
    "usb_events",
    "usb_connect_count",
    "usb_disconnect_count",
    "after_hours_usb",
    "unique_device_pcs",
    "http_requests",
    "after_hours_http",
    "unique_domains"
]

results = []

minimum_records = 60
train_ratio = 0.70

# Process each employee separately
for user, user_data in df.groupby("user"):

    user_data = user_data.sort_values("day").copy()

    # Skip employees with insufficient history
    if len(user_data) < minimum_records:
        continue

    # Chronological split
    split_index = int(len(user_data) * train_ratio)

    train_data = user_data.iloc[:split_index]
    test_data = user_data.iloc[split_index:]

    X_train = train_data[features]
    X_test = test_data[features]

    # Train individual Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    model.fit(X_train)

    # Predict anomalies on future observations
    predictions = model.predict(X_test)

    # Lower decision_function values indicate more anomalous records
    anomaly_scores = -model.decision_function(X_test)

    user_results = test_data[
        ["user", "day"]
    ].copy()

    user_results["anomaly"] = (
        predictions == -1
    ).astype(int)

    user_results["anomaly_score"] = anomaly_scores

    results.append(user_results)


# Combine results
if not results:
    raise ValueError("No employees had sufficient historical records.")

final_results = pd.concat(results, ignore_index=True)

# Calculate relative anomaly percentile within each employee
final_results["risk_score"] = (
    final_results.groupby("user")["anomaly_score"]
    .rank(pct=True) * 100
).round(2)

# Save results
final_results.to_csv(
    "processed/anomaly_results.csv",
    index=False
)

# Summary
print("\n--- ISOLATION FOREST RESULTS ---")

print("Employees modelled:", final_results["user"].nunique())

print("Records evaluated:", len(final_results))

print(
    "Anomalous records:",
    final_results["anomaly"].sum()
)

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

print("\nSaved: processed/anomaly_results.csv")
