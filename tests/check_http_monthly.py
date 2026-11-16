import pandas as pd

df = pd.read_csv("processed/user_daily_features.csv")
df["day"] = pd.to_datetime(df["day"])

df["month"] = df["day"].dt.to_period("M")

monthly = df.groupby("month").agg(
    records=("user", "count"),
    average_http_requests=("http_requests", "mean"),
    average_after_hours_http=("after_hours_http", "mean"),
    average_unique_domains=("unique_domains", "mean")
)

print("\n--- MONTHLY HTTP ACTIVITY ---")
print(monthly.round(2).to_string())

print("\n--- MONTHLY LOGON ACTIVITY ---")
logon = df.groupby("month").agg(
    records=("user", "count"),
    average_logons=("logon_count", "mean"),
    average_after_hours_logons=("after_hours_logons", "mean")
)

print(logon.round(2).to_string())