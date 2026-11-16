import pandas as pd

df = pd.read_csv("processed/user_daily_features.csv")

print("\n--- FEATURE LOGIC CHECK ---")

checks = {
    "After-hours logons <= total logons":
        (df["after_hours_logons"] <= df["logon_count"]).all(),

    "USB connections <= USB events":
        (df["usb_connect_count"] <= df["usb_events"]).all(),

    "USB disconnections <= USB events":
        (df["usb_disconnect_count"] <= df["usb_events"]).all(),

    "After-hours USB <= USB connections":
        (df["after_hours_usb"] <= df["usb_connect_count"]).all(),

    "After-hours HTTP <= HTTP requests":
        (df["after_hours_http"] <= df["http_requests"]).all(),

    "Unique URLs <= HTTP requests":
        (df["unique_urls"] <= df["http_requests"]).all()
}

for name, result in checks.items():
    print(f"{name}: {'PASS' if result else 'FAIL'}")

print("\nUnique URLs and domains comparison:")
print("Equal values:",
      (df["unique_urls"] == df["unique_domains"]).sum())
print("Total records:", len(df))