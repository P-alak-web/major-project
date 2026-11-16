import pandas as pd

df = pd.read_csv("processed/user_daily_features.csv")

user_days = df.groupby("user").size()

print("\n--- USER HISTORY ANALYSIS ---")

print("Total users:", len(user_days))
print("\nDaily records per user:")
print(user_days.describe())

print("\nUsers with fewer than 30 records:",
      (user_days < 30).sum())

print("Users with fewer than 60 records:",
      (user_days < 60).sum())

print("Users with more than 300 records:",
      (user_days > 300).sum())

print("\nFive users with the fewest records:")
print(user_days.nsmallest(5))

print("\nFive users with the most records:")
print(user_days.nlargest(5))