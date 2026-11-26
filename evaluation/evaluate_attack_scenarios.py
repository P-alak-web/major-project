import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import TEST_DATE, ALERT_BUDGETS


CONTROLLED_DIR = ROOT / "outputs" / "eval" / "controlled"

TRUTH_FILE = (
    ROOT / "outputs" / "ground_truth"
    / "ground_truth_user_day.csv"
)

OUTPUT_DIR = (
    CONTROLLED_DIR / "explainability" / "scenarios"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODELS = {
    "EBM": "ebm",
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
    "Pooled Isolation Forest": "pooled_isolation_forest",
    "Per-user Isolation Forest": "per_user_isolation_forest"
}


def main():

    print("=" * 65)
    print("SCENARIO-WISE INSIDER THREAT EVALUATION")
    print("=" * 65)

    truth = pd.read_csv(TRUTH_FILE)

    truth["date"] = pd.to_datetime(truth["date"])

    truth = truth.rename(columns={"date": "day"})

    truth = truth[
        (truth["day"] >= TEST_DATE)
        & (truth["label"] == 1)
    ].copy()

    truth = truth.drop_duplicates(
        ["user", "day", "scenario"]
    )

    print("\nTEST SCENARIO DISTRIBUTION")

    distribution = truth.groupby("scenario").agg(
        malicious_days=("user", "size"),
        distinct_insiders=("user", "nunique")
    ).reset_index()

    print(distribution.to_string(index=False))

    distribution.to_csv(
        OUTPUT_DIR / "test_scenario_distribution.csv",
        index=False
    )

    results = []

    for model_name, filename in MODELS.items():

        predictions = pd.read_csv(
            CONTROLLED_DIR
            / f"predictions_{filename}.csv"
        )

        predictions["day"] = pd.to_datetime(
            predictions["day"]
        )

        merged = truth.merge(
            predictions[
                ["user", "day", "score", "rank"]
            ],
            on=["user", "day"],
            how="left",
            validate="many_to_one"
        )

        assert merged["rank"].notna().all(), (
            f"Missing predictions for {model_name}"
        )

        for scenario, group in merged.groupby("scenario"):

            total_days = len(group)

            total_insiders = group["user"].nunique()

            record = {
                "model": model_name,
                "scenario": scenario,
                "malicious_days": total_days,
                "total_insiders": total_insiders
            }

            for k in ALERT_BUDGETS:

                detected = group[group["rank"] <= k]

                record[f"detected_days_at_{k}"] = len(
                    detected
                )

                record[f"recall_at_{k}"] = (
                    len(detected) / total_days
                )

                record[f"insiders_detected_at_{k}"] = (
                    detected["user"].nunique()
                )

            results.append(record)

    summary = pd.DataFrame(results)

    summary.to_csv(
        OUTPUT_DIR / "scenario_model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("SCENARIO-WISE DETECTION RESULTS")
    print("=" * 65)

    print(
        summary[
            [
                "model",
                "scenario",
                "malicious_days",
                "detected_days_at_10",
                "recall_at_10",
                "insiders_detected_at_10",
                "total_insiders"
            ]
        ].to_string(index=False)
    )

    print("\nRECALL@10/DAY COMPARISON")

    pivot = summary.pivot(
        index="model",
        columns="scenario",
        values="recall_at_10"
    )

    print(pivot.round(4).to_string())

    pivot.to_csv(
        OUTPUT_DIR / "scenario_recall_comparison.csv"
    )

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()