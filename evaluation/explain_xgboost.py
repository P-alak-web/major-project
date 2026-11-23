import pandas as pd
import numpy as np
import joblib
import shap

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = ROOT / "processed" / "user_daily_features_advanced_v2.csv"
PREDICTIONS_FILE = ROOT / "outputs" / "eval" / "predictions_xgboost.csv"

MODEL_FILE = ROOT / "models" / "saved" / "xgboost_model.pkl"
FEATURE_NAMES_FILE = ROOT / "models" / "saved" / "xgboost_features.pkl"

OUTPUT_DIR = ROOT / "outputs" / "explanations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading XGBoost model...")

    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURE_NAMES_FILE)

    print("Loading behavioral features...")

    features = pd.read_csv(FEATURES_FILE)

    features["day"] = pd.to_datetime(
        features["day"]
    ).dt.strftime("%Y-%m-%d")

    print("Loading XGBoost predictions...")

    predictions = pd.read_csv(PREDICTIONS_FILE)

    predictions["day"] = pd.to_datetime(
        predictions["day"]
    ).dt.strftime("%Y-%m-%d")

    # Select the ten highest-scoring employee-days.
    top_alerts = predictions.nlargest(
        10,
        "score"
    ).copy()

    top_alerts = top_alerts.merge(
        features,
        on=["user", "day"],
        how="left",
        validate="one_to_one"
    )

    if top_alerts[feature_cols].isna().any().any():
        raise ValueError(
            "Missing behavioral features for selected alerts."
        )

    X = top_alerts[feature_cols]

    print("Calculating SHAP explanations...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X)

    # SHAP values explain the model's raw output
    # (log-odds for this binary XGBoost classifier).
    contributions = pd.DataFrame(
        shap_values.values,
        columns=feature_cols
    )

    contributions.insert(
        0,
        "day",
        top_alerts["day"].to_numpy()
    )

    contributions.insert(
        0,
        "user",
        top_alerts["user"].to_numpy()
    )

    contributions.insert(
        2,
        "risk_score",
        top_alerts["score"].to_numpy()
    )

    output_file = OUTPUT_DIR / "xgboost_shap_top10.csv"

    contributions.to_csv(
        output_file,
        index=False
    )

    print("\nTop 10 alerts explained:")
    print(
        top_alerts[
            ["user", "day", "score"]
        ].to_string(index=False)
    )

    print("\nExplanation for highest-scoring alert:")

    first = contributions.iloc[0][feature_cols]

    top_features = first.reindex(
        first.abs().sort_values(
            ascending=False
        ).index
    ).head(10)

    print(top_features.to_string())

    print("\nSaved SHAP explanations to:")
    print(output_file)

    print("\nSHAP pipeline completed successfully!")


if __name__ == "__main__":
    main()