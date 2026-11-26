import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import FEATURE_COLUMNS, RANDOM_SEED

from evaluation.prepare_controlled_data import prepare_data


# --------------------------------------------------
# 1. Paths and settings
# --------------------------------------------------

CONTROLLED_DIR = ROOT / "outputs" / "eval" / "controlled"

MODEL_FILE = CONTROLLED_DIR / "models" / "xgboost.pkl"

PREDICTIONS_FILE = CONTROLLED_DIR / "predictions_xgboost.csv"

OUTPUT_DIR = CONTROLLED_DIR / "explainability" / "xgboost"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALERT_BUDGET = 10
GLOBAL_SAMPLE_SIZE = 5000


# --------------------------------------------------
# 2. Load model and test dataset
# --------------------------------------------------

def load_data():

    _, _, test = prepare_data()

    model = joblib.load(MODEL_FILE)

    predictions = pd.read_csv(PREDICTIONS_FILE)

    predictions["day"] = pd.to_datetime(
        predictions["day"]
    )

    df = test.merge(
        predictions[["user", "day", "score", "rank"]],
        on=["user", "day"],
        how="inner",
        validate="one_to_one"
    )

    assert len(df) == len(test) == len(predictions)

    df["alert"] = (
        df["rank"] <= ALERT_BUDGET
    ).astype(int)

    conditions = [
        (df["label"] == 1) & (df["alert"] == 1),
        (df["label"] == 0) & (df["alert"] == 1),
        (df["label"] == 1) & (df["alert"] == 0),
        (df["label"] == 0) & (df["alert"] == 0)
    ]

    df["category"] = np.select(
        conditions,
        [
            "True Positive",
            "False Positive",
            "False Negative",
            "True Negative"
        ],
        default="Unknown"
    )

    return model, df


# --------------------------------------------------
# 3. Global SHAP analysis
# --------------------------------------------------

def global_explanation(explainer, df):

    print("\nGenerating global SHAP explanations...")

    sample = df.sample(
        n=min(GLOBAL_SAMPLE_SIZE, len(df)),
        random_state=RANDOM_SEED
    )

    X = sample[FEATURE_COLUMNS]

    explanation = explainer(X)

    values = explanation.values

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "mean_absolute_shap": np.abs(values).mean(axis=0)
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "global_shap_importance.csv",
        index=False
    )

    print("\nTop 10 global features:")
    print(importance.head(10).to_string(index=False))

    # Global feature importance chart
    shap.plots.bar(
        explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "global_shap_bar.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")

    # SHAP beeswarm chart
    shap.plots.beeswarm(
        explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "global_shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")


# --------------------------------------------------
# 4. Explain prediction categories
# --------------------------------------------------

def category_explanations(explainer, df):

    records = []
    local_records = []

    categories = [
        "True Positive",
        "False Positive",
        "False Negative",
        "True Negative"
    ]

    for category in categories:

        group = df[df["category"] == category]

        if group.empty:
            continue

        # Explain every known malicious employee-day.
        # Sample up to 401 records from negative categories.

        if category in ["True Positive", "False Negative"]:
            sample = group.copy()

        else:
            sample = group.sample(
                n=min(401, len(group)),
                random_state=RANDOM_SEED
            )

        print("\nCategory:", category)
        print("Total records:", len(group))
        print("Explained records:", len(sample))

        X = sample[FEATURE_COLUMNS]

        explanation = explainer(X)

        values = explanation.values

        mean_absolute = np.abs(values).mean(axis=0)

        mean_signed = values.mean(axis=0)

        for i, feature in enumerate(FEATURE_COLUMNS):

            records.append({
                "category": category,
                "total_records": len(group),
                "explained_records": len(sample),
                "feature": feature,
                "mean_absolute_shap": mean_absolute[i],
                "mean_signed_shap": mean_signed[i]
            })

        # Save individual employee-day contributions
        for row_number, (_, row) in enumerate(
            sample.iterrows()
        ):

            for feature_number, feature in enumerate(
                FEATURE_COLUMNS
            ):

                local_records.append({
                    "user": row["user"],
                    "day": row["day"],
                    "category": category,
                    "label": row["label"],
                    "score": row["score"],
                    "rank": row["rank"],
                    "feature": feature,
                    "feature_value": row[feature],
                    "shap_value": values[
                        row_number,
                        feature_number
                    ]
                })

        # Representative local explanation
        # Select highest-scoring record in each sample.

        position = int(
            np.argmax(sample["score"].to_numpy())
        )

        shap.plots.waterfall(
            explanation[position],
            max_display=12,
            show=False
        )

        plt.tight_layout()

        filename = category.lower().replace(" ", "_")

        plt.savefig(
            OUTPUT_DIR / f"waterfall_{filename}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close("all")

    summary = pd.DataFrame(records)

    summary.to_csv(
        OUTPUT_DIR / "category_shap_importance.csv",
        index=False
    )

    pd.DataFrame(local_records).to_csv(
        OUTPUT_DIR / "local_shap_explanations.csv",
        index=False
    )

    print("\nTOP 5 FEATURES BY CATEGORY")

    for category in categories:

        subset = summary[
            summary["category"] == category
        ]

        subset = subset.sort_values(
            "mean_absolute_shap",
            ascending=False
        )

        print("\n", category)

        print(
            subset[
                [
                    "feature",
                    "mean_absolute_shap",
                    "mean_signed_shap"
                ]
            ].head(5).to_string(index=False)
        )


# --------------------------------------------------
# 5. Main
# --------------------------------------------------

def main():

    print("=" * 65)
    print("CONTROLLED XGBOOST - SHAP EXPLAINABILITY")
    print("=" * 65)

    model, df = load_data()

    print("\nTest records:", len(df))

    print("\nCONFUSION MATRIX AT TOP-10 ALERTS/DAY")

    counts = df["category"].value_counts()

    print(counts.to_string())

    counts.rename_axis("category").reset_index(
        name="count"
    ).to_csv(
        OUTPUT_DIR / "confusion_summary.csv",
        index=False
    )

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(
        model,
        model_output="raw"
    )

    global_explanation(explainer, df)

    category_explanations(explainer, df)

    print("\n" + "=" * 65)
    print("SHAP ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()