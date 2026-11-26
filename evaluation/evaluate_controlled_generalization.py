import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

from config import FEATURE_COLUMNS, ALERT_BUDGETS

from evaluation.prepare_controlled_data import prepare_data

from evaluation.controlled_model_comparison import (
    fit_model,
    predict_scores
)


CONTROLLED_DIR = ROOT / "outputs" / "eval" / "controlled"

OUTPUT_DIR = CONTROLLED_DIR / "generalization"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]

HOLDOUT_FRACTION = 0.20

MODELS = ["EBM", "XGBoost"]


def evaluate_predictions(model_name, seed, test, scores):

    predictions = test[
        ["user", "day", "label"]
    ].copy()

    predictions["score"] = scores

    predictions = predictions.sort_values(
        ["day", "score", "user"],
        ascending=[True, False, True]
    )

    predictions["rank"] = (
        predictions.groupby("day").cumcount() + 1
    )

    positives = predictions[
        predictions["label"] == 1
    ]

    record = {
        "model": model_name,
        "seed": seed,
        "test_employees":
            predictions["user"].nunique(),
        "test_records": len(predictions),
        "test_positives": len(positives),
        "test_insiders": positives["user"].nunique(),
        "prevalence": predictions["label"].mean(),
        "pr_auc": average_precision_score(
            predictions["label"],
            predictions["score"]
        )
    }

    for k in ALERT_BUDGETS:

        alerts = predictions[
            predictions["rank"] <= k
        ]

        detected = alerts[
            alerts["label"] == 1
        ]

        record[f"precision_at_{k}"] = (
            len(detected) / len(alerts)
        )

        record[f"recall_at_{k}"] = (
            len(detected) / len(positives)
        )

        record[f"insiders_detected_at_{k}"] = (
            detected["user"].nunique()
        )

    # Additional proportional alert budget:
    # approximately 20% of the original top-10 budget.

    alerts = predictions[
        predictions["rank"] <= 2
    ]

    detected = alerts[
        alerts["label"] == 1
    ]

    record["precision_at_2"] = (
        len(detected) / len(alerts)
    )

    record["recall_at_2"] = (
        len(detected) / len(positives)
    )

    record["insiders_detected_at_2"] = (
        detected["user"].nunique()
    )

    return record


def main():

    print("=" * 65)
    print("CONTROLLED UNSEEN-EMPLOYEE GENERALIZATION")
    print("=" * 65)

    train, validation, test = prepare_data()

    development = pd.concat(
        [train, validation],
        ignore_index=True
    )

    with open(
        CONTROLLED_DIR / "selected_parameters.json",
        "r"
    ) as file:

        selected_parameters = json.load(file)

    employees = test.groupby("user").agg(
        insider=("label", "max")
    ).reset_index()

    print("\nTotal test employees:", len(employees))

    print(
        "Total test insiders:",
        employees["insider"].sum()
    )

    results = []

    for seed in SEEDS:

        print("\n" + "=" * 65)
        print("HOLDOUT SEED:", seed)
        print("=" * 65)

        _, heldout = train_test_split(
            employees,
            test_size=HOLDOUT_FRACTION,
            stratify=employees["insider"],
            random_state=seed
        )

        heldout_users = set(heldout["user"])

        training_data = development[
            ~development["user"].isin(heldout_users)
        ].copy()

        testing_data = test[
            test["user"].isin(heldout_users)
        ].copy()

        training_data = training_data.reset_index(
            drop=True
        )

        testing_data = testing_data.reset_index(
            drop=True
        )

        overlap = (
            set(training_data["user"])
            & set(testing_data["user"])
        )

        assert len(overlap) == 0

        assert (
            training_data["day"].max()
            < testing_data["day"].min()
        )

        assert testing_data["label"].sum() > 0

        print(
            "Training employees:",
            training_data["user"].nunique()
        )

        print(
            "Held-out employees:",
            testing_data["user"].nunique()
        )

        print(
            "Training records:",
            len(training_data)
        )

        print(
            "Test records:",
            len(testing_data)
        )

        print(
            "Test malicious days:",
            testing_data["label"].sum()
        )

        print("Employee overlap:", len(overlap))

        for model_name in MODELS:

            print("\nTraining:", model_name)

            params = selected_parameters[model_name]["parameters"]

            model = fit_model(
                model_name,
                training_data,
                params
            )

            scores = predict_scores(
                model_name,
                model,
                testing_data
            )

            result = evaluate_predictions(
                model_name,
                seed,
                testing_data,
                scores
            )

            results.append(result)

            print(
                "PR-AUC:",
                round(result["pr_auc"], 6)
            )

            print(
                "Recall@2/day:",
                round(result["recall_at_2"], 6)
            )

            print(
                "Recall@10/day:",
                round(result["recall_at_10"], 6)
            )

            print(
                "Insiders detected@10/day:",
                result["insiders_detected_at_10"],
                "/",
                result["test_insiders"]
            )

            # Save intermediate results.

            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "generalization_results.csv",
                index=False
            )

            del model

    results = pd.DataFrame(results)

    summary = results.groupby("model").agg(

        experiments=("seed", "size"),

        mean_pr_auc=("pr_auc", "mean"),

        std_pr_auc=("pr_auc", "std"),

        mean_precision_at_2=(
            "precision_at_2", "mean"
        ),

        mean_recall_at_2=(
            "recall_at_2", "mean"
        ),

        mean_precision_at_10=(
            "precision_at_10", "mean"
        ),

        mean_recall_at_10=(
            "recall_at_10", "mean"
        ),

        std_recall_at_10=(
            "recall_at_10", "std"
        )

    ).reset_index()

    summary.to_csv(
        OUTPUT_DIR / "generalization_summary.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("UNSEEN-EMPLOYEE RESULTS")
    print("=" * 65)

    print(
        results[
            [
                "model",
                "seed",
                "test_employees",
                "test_positives",
                "pr_auc",
                "recall_at_2",
                "recall_at_10",
                "insiders_detected_at_10",
                "test_insiders"
            ]
        ].round(6).to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("GENERALIZATION SUMMARY")
    print("=" * 65)

    print(
        summary.round(6).to_string(index=False)
    )

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()