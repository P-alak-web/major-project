
import pandas as pd

GT_PATH = "outputs/ground_truth/ground_truth_user_day.csv"
PRED_PATH = "processed/r4_2_model_predictions.csv"

gt = pd.read_csv(GT_PATH)
pred = pd.read_csv(PRED_PATH)

# Normalize matching keys
gt["date"] = pd.to_datetime(gt["date"]).dt.strftime("%Y-%m-%d")
pred["day"] = pd.to_datetime(pred["day"]).dt.strftime("%Y-%m-%d")

gt["user"] = gt["user"].astype(str).str.strip()
pred["user"] = pred["user"].astype(str).str.strip()

# Validate ground truth and predictions
assert gt[["user", "date"]].duplicated().sum() == 0
assert pred[["user", "day"]].duplicated().sum() == 0
assert gt["label"].eq(1).all()

# Match confirmed positives to predictions
matched = gt.merge(
    pred[["user", "day", "anomaly"]],
    left_on=["user", "date"],
    right_on=["user", "day"],
    how="inner",
    validate="one_to_one"
)

total = len(matched)
detected = int(matched["anomaly"].eq(1).sum())
missed = total - detected

print("\n=== OVERALL EVALUATION ===")
print("Total predictions:", len(pred))
print("Confirmed positive user-days:", len(gt))
print("Matched positive user-days:", total)
print("Detected positives:", detected)
print("Missed positives:", missed)
print("Positive-label recall: {:.2f}%".format(
    100 * detected / total if total else 0
))

# Scenario-wise recall
summary = matched.groupby("scenario").agg(
    positive_days=("label", "count"),
    detected=("anomaly", "sum")
)

summary["missed"] = summary["positive_days"] - summary["detected"]
summary["recall_pct"] = (
    100 * summary["detected"] / summary["positive_days"]
).round(2)

print("\n=== SCENARIO-WISE EVALUATION ===")
print(summary.to_string())

print("\nNOTE: Unlabeled user-days are not verified negatives.")
print("Precision, F1 and accuracy are not calculated.")


# User-level detection on confirmed positive days
user_results = matched.groupby(
    ["scenario", "user"]
)["anomaly"].max().reset_index()

user_summary = user_results.groupby("scenario").agg(
    total_users=("user", "count"),
    detected_users=("anomaly", "sum")
)

user_summary["missed_users"] = (
    user_summary["total_users"] - user_summary["detected_users"]
)

user_summary["detection_pct"] = (
    100 * user_summary["detected_users"]
    / user_summary["total_users"]
).round(2)

print("\n=== SCENARIO-WISE USER DETECTION ===")
print(user_summary.to_string())

total_users = matched["user"].nunique()
detected_users = matched.groupby("user")["anomaly"].max().sum()

print("\n=== OVERALL USER DETECTION ===")
print("Known insider users:", total_users)
print("Detected users:", int(detected_users))
print("Missed users:", int(total_users - detected_users))
print("User detection rate: {:.2f}%".format(
    100 * detected_users / total_users
))
