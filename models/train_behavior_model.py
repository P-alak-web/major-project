import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_PATH / "processed"

INPUT_FILE = PROCESSED_PATH / "user_daily_features.csv"
OUTPUT_FILE = PROCESSED_PATH / "behavior_model_results.csv"

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(INPUT_FILE)

df["day"] = pd.to_datetime(
    df["day"],
    format="%Y-%m-%d",
    errors="raise"
)

df = df.sort_values(["user", "day"]).reset_index(drop=True)

# ==========================================
# 3. SELECT BEHAVIOURAL FEATURES
# ==========================================

ACTIVITY_FEATURES = [
    "logon_count",
    "after_hours_logons",
    "usb_connect_count",
    "after_hours_usb",
    "http_requests",
    "after_hours_http",
    "email_count",
    "after_hours_email",
    "file_operations",
    "after_hours_file"
]

MINIMUM_RECORDS = 60
TRAIN_RATIO = 0.70

results = []
skipped_users = 0

# ==========================================
# 4. TRAIN ONE MODEL PER EMPLOYEE
# ==========================================

for user, user_data in df.groupby("user"):

    user_data = user_data.sort_values("day").copy()

    if len(user_data) < MINIMUM_RECORDS:
        skipped_users += 1
        continue

    split_index = int(len(user_data) * TRAIN_RATIO)

    train_data = user_data.iloc[:split_index].copy()
    test_data = user_data.iloc[split_index:].copy()

    # --------------------------------------
    # Calculate baseline from TRAINING only
    # --------------------------------------

    historical_median = train_data[ACTIVITY_FEATURES].median()

    historical_iqr = (
        train_data[ACTIVITY_FEATURES].quantile(0.75)
        - train_data[ACTIVITY_FEATURES].quantile(0.25)
    )

    # Avoid division by zero
    historical_iqr = historical_iqr.replace(0, 1)

    # --------------------------------------
    # Employee-relative deviations
    # --------------------------------------

    X_train = (
        train_data[ACTIVITY_FEATURES] - historical_median
    ) / historical_iqr

    X_test = (
        test_data[ACTIVITY_FEATURES] - historical_median
    ) / historical_iqr

    # --------------------------------------
    # Add calendar context
    # --------------------------------------

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["is_weekend"] = (
        train_data["day"].dt.dayofweek >= 5
    ).astype(int).to_numpy()

    X_test["is_weekend"] = (
        test_data["day"].dt.dayofweek >= 5
    ).astype(int).to_numpy()

    # --------------------------------------
    # Train Isolation Forest
    # --------------------------------------

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train)

    predictions = model.predict(X_test)

    anomaly_scores = -model.decision_function(X_test)

    user_results = test_data[["user", "day"]].copy()

    user_results["anomaly_B"] = (
        predictions == -1
    ).astype(int)

    user_results["anomaly_score_B"] = anomaly_scores

    results.append(user_results)

# ==========================================
# 5. COMBINE RESULTS
# ==========================================

if not results:
    raise ValueError("No employees qualified for training.")

final_results = pd.concat(results, ignore_index=True)

final_results.to_csv(OUTPUT_FILE, index=False)

print("\n========== MODEL B RESULTS ==========")

print("Employees modelled:", final_results["user"].nunique())
print("Employees skipped:", skipped_users)
print("Records evaluated:", len(final_results))
print("Anomalies:", int(final_results["anomaly_B"].sum()))

print(
    "Anomaly percentage:",
    round(final_results["anomaly_B"].mean() * 100, 2),
    "%"
)

print("\nSaved to:", OUTPUT_FILE)