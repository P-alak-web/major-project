import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent

EVAL_DIR = ROOT / "outputs" / "eval"
TRUTH_FILE = (
    ROOT / "outputs" / "ground_truth" / "ground_truth_user_day.csv"
)

MODELS = {
    "Per-user Isolation Forest": "predictions_per_user.csv",
    "Pooled Isolation Forest": "predictions_pooled.csv",
    "Random Forest": "predictions_random_forest.csv",
    "XGBoost": "predictions_xgboost.csv",
    "EBM": "predictions_ebm.csv",
}

ALERT_BUDGETS = [10, 50, 100]


def load_ground_truth():
    truth = pd.read_csv(TRUTH_FILE)

    truth["day"] = pd.to_datetime(
        truth["date"]
    ).dt.strftime("%Y-%m-%d")

    truth = (
        truth.groupby(["user", "day"], as_index=False)["label"]
        .max()
    )

    return truth


def load_predictions(filename, truth):
    predictions = pd.read_csv(EVAL_DIR / filename)

    predictions["day"] = pd.to_datetime(
        predictions["day"]
    ).dt.strftime("%Y-%m-%d")

    predictions = predictions.merge(
        truth,
        on=["user", "day"],
        how="left",
        validate="one_to_one",
    )

    predictions["label"] = (
        predictions["label"].fillna(0).astype(int)
    )

    return predictions


def evaluate_daily_alerts(df, budget):
    # Rank highest anomaly/risk scores first each day.
    ranked = df.sort_values(
        ["day", "score"],
        ascending=[True, False],
    )

    alerts = ranked.groupby(
        "day", group_keys=False
    ).head(budget)

    true_positives = int(alerts["label"].sum())
    total_positives = int(df["label"].sum())

    precision = (
        true_positives / len(alerts)
        if len(alerts) else 0
    )

    recall = (
        true_positives / total_positives
        if total_positives else 0
    )

    known_insiders = set(
        df.loc[df["label"] == 1, "user"]
    )

    detected_insiders = set(
        alerts.loc[alerts["label"] == 1, "user"]
    )

    return {
        f"precision@{budget}/day": precision,
        f"recall@{budget}/day": recall,
        f"insiders_found@{budget}/day": len(detected_insiders),
        f"total_insiders": len(known_insiders),
    }


def main():
    print("Loading CERT ground truth...")
    truth = load_ground_truth()

    results = []
    prediction_keys = {}

    for model_name, filename in MODELS.items():
        print(f"\nEvaluating {model_name}...")

        df = load_predictions(filename, truth)

        if df.duplicated(["user", "day"]).any():
            raise ValueError(
                f"Duplicate user-day predictions: {model_name}"
            )

        if not np.isfinite(df["score"]).all():
            raise ValueError(
                f"Invalid scores: {model_name}"
            )

        prediction_keys[model_name] = set(
            zip(df["user"], df["day"])
        )

        metrics = {
            "model": model_name,
            "test_rows": len(df),
            "test_positives": int(df["label"].sum()),
            "pr_auc": average_precision_score(
                df["label"], df["score"]
            ),
        }

        for budget in ALERT_BUDGETS:
            metrics.update(
                evaluate_daily_alerts(df, budget)
            )

        results.append(metrics)

    # Ensure all models were evaluated on identical rows.
    names = list(prediction_keys)
    reference = prediction_keys[names[0]]

    for name in names[1:]:
        if prediction_keys[name] != reference:
            raise ValueError(
                f"Prediction rows differ for {name}. "
                "A direct comparison is not valid."
            )

    summary = pd.DataFrame(results)

    summary = summary.sort_values(
        "pr_auc", ascending=False
    )

    output = EVAL_DIR / "all_models_comparison.csv"
    summary.to_csv(output, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(summary.round(5).to_string(index=False))

    print("\nSaved comparison to:")
    print(output)


if __name__ == "__main__":
    main()