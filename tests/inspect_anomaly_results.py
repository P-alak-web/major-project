import pandas as pd

df = pd.read_csv("processed/anomaly_results.csv")

print("\n--- ANOMALY DISTRIBUTION ---")

# Overall summary
print("\nOverall anomaly counts:")
print(df["anomaly"].value_counts())

# Employee-level anomaly rates
user_stats = df.groupby("user").agg(
    total_records=("anomaly", "count"),
    anomalous_records=("anomaly", "sum"),
    average_score=("anomaly_score", "mean")
)

user_stats["anomaly_rate"] = (
    user_stats["anomalous_records"] /
    user_stats["total_records"] * 100
)

print("\nEmployee anomaly-rate statistics:")
print(user_stats["anomaly_rate"].describe().round(2))

print("\nEmployees with highest anomaly rates:")
print(
    user_stats.sort_values(
        "anomaly_rate", ascending=False
    ).head(10).round(2)
)

print("\nEmployees with lowest anomaly rates:")
print(
    user_stats.sort_values(
        "anomaly_rate", ascending=True
    ).head(10).round(2)
)

# Distribution of anomaly scores
print("\nAnomaly score statistics:")
print(df["anomaly_score"].describe().round(4))

# Monthly anomaly distribution
df["day"] = pd.to_datetime(df["day"])
df["month"] = df["day"].dt.to_period("M")

monthly = df.groupby("month").agg(
    total_records=("anomaly", "count"),
    anomalies=("anomaly", "sum")
)

monthly["anomaly_rate"] = (
    monthly["anomalies"] /
    monthly["total_records"] * 100
)

print("\nMonthly anomaly rates:")
print(monthly.round(2).to_string())