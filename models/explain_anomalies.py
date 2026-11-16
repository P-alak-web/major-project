import pandas as pd
import joblib
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_PATH / "processed"
MODELS_PATH = PROJECT_PATH / "saved_models"

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

# Load data
features_df = pd.read_csv(
    PROCESSED_PATH / "user_daily_features.csv"
)

results_df = pd.read_csv(
    PROCESSED_PATH / "anomaly_results.csv"
)

# Explain only the highest-scoring flagged anomalies
top_anomalies = (
    results_df[results_df["anomaly"] == 1]
    .sort_values("anomaly_score", ascending=False)
    .head(20)
)

explanations = []

for _, anomaly in top_anomalies.iterrows():

    user = anomaly["user"]
    day = anomaly["day"]

    model_path = MODELS_PATH / f"{user}_isolation_forest.joblib"

    model = joblib.load(model_path)

    user_history = features_df[
        features_df["user"] == user
    ].sort_values("day")

    # Reconstruct the original 70% training period
    split_index = int(len(user_history) * 0.70)
    train_history = user_history.iloc[:split_index]

    # Use training-period medians as reference values
    reference = train_history[FEATURES].median()

    record = features_df[
        (features_df["user"] == user) &
        (features_df["day"] == day)
    ]

    if len(record) != 1:
        raise ValueError(
            f"Expected one record for {user} on {day}"
        )

    original = record[FEATURES].copy()

    original_score = -model.decision_function(original)[0]

    feature_effects = []

    for feature in FEATURES:

        modified = original.copy()

        # Replace one feature with its historical median
        modified.loc[:, feature] = reference[feature]

        modified_score = -model.decision_function(modified)[0]

        # Positive effect means replacing this feature
        # reduces the anomaly score.
        effect = original_score - modified_score

        feature_effects.append((feature, effect))

    feature_effects.sort(
        key=lambda item: item[1],
        reverse=True
    )

    top_feature, top_effect = feature_effects[0]

    explanations.append({
        "user": user,
        "day": day,
        "anomaly_score": original_score,
        "top_feature": top_feature,
        "feature_value": original.iloc[0][top_feature],
        "historical_median": reference[top_feature],
        "score_reduction": top_effect
    })

explanation_df = pd.DataFrame(explanations)

output_path = PROCESSED_PATH / "anomaly_explanations.csv"

explanation_df.to_csv(output_path, index=False)

print("\n========== ANOMALY EXPLANATIONS ==========")
print(explanation_df.to_string(index=False))
print("\nSaved to:", output_path)