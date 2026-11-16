import pandas as pd

df = pd.read_csv("processed/user_daily_features.csv")
df["day"] = pd.to_datetime(df["day"])

features = [
    "total_logon_events",
    "logon_count",
    "after_hours_logons",
    "usb_connect_count",
    "after_hours_usb",
    "http_requests",
    "after_hours_http",
    "unique_domains"
]

train_parts = []
test_parts = []

for user, data in df.groupby("user"):

    data = data.sort_values("day")

    if len(data) < 60:
        continue

    split = int(len(data) * 0.70)

    train = data.iloc[:split].copy()
    test = data.iloc[split:].copy()

    train_parts.append(train)
    test_parts.append(test)

train_df = pd.concat(train_parts)
test_df = pd.concat(test_parts)

print("\n--- TRAINING PERIOD ---")
print("Start:", train_df["day"].min())
print("End:", train_df["day"].max())
print("Records:", len(train_df))

print("\n--- TESTING PERIOD ---")
print("Start:", test_df["day"].min())
print("End:", test_df["day"].max())
print("Records:", len(test_df))

print("\n--- FEATURE COMPARISON ---")

comparison = pd.DataFrame({
    "Training mean": train_df[features].mean(),
    "Testing mean": test_df[features].mean()
})

comparison["Percentage change"] = (
    (comparison["Testing mean"] - comparison["Training mean"]) /
    comparison["Training mean"].replace(0, float("nan")) * 100
)

print(comparison.round(2).to_string())

print("\n--- MONTHLY TEST ANOMALY RATES ---")

results = pd.read_csv("processed/anomaly_results.csv")
results["day"] = pd.to_datetime(results["day"])
results["month"] = results["day"].dt.to_period("M")

monthly = results.groupby("month")["anomaly"].agg(
    total_records="count",
    anomalies="sum",
    anomaly_rate="mean"
)

monthly["anomaly_rate"] *= 100

print(monthly.round(2).to_string())