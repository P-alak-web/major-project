import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.special import expit

from config import FEATURE_COLUMNS, RANDOM_SEED
from evaluation.prepare_controlled_data import prepare_data


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CONTROLLED_DIR = ROOT / "outputs" / "eval" / "controlled"

MODEL_FILE = CONTROLLED_DIR / "models" / "ebm.pkl"

PREDICTIONS_FILE = CONTROLLED_DIR / "predictions_ebm.csv"

OUTPUT_DIR = CONTROLLED_DIR / "explainability" / "ebm"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALERT_BUDGET = 10
GLOBAL_SAMPLE_SIZE = 5000


# --------------------------------------------------
# Load model and predictions
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
# Extract intrinsic EBM contributions
# --------------------------------------------------

def get_contributions(model, X):

    contributions = np.asarray(
        model.eval_terms(X)
    )

    intercept = float(
        np.asarray(model.intercept_).ravel()[0]
    )

    reconstructed = expit(
        intercept + contributions.sum(axis=1)
    )

    predicted = model.predict_proba(X)[:, 1]

    max_error = np.max(
        np.abs(reconstructed - predicted)
    )

    assert np.allclose(
        reconstructed,
        predicted,
        atol=1e-7
    ), "EBM contribution reconstruction failed."

    return contributions, intercept, max_error


# --------------------------------------------------
# Global explanation
# --------------------------------------------------

def global_explanation(model, df):

    print("\nGenerating global EBM explanations...")

    sample = df.sample(
        n=min(GLOBAL_SAMPLE_SIZE, len(df)),
        random_state=RANDOM_SEED
    )

    X = sample[FEATURE_COLUMNS]

    values, intercept, error = get_contributions(
        model,
        X
    )

    importance = pd.DataFrame({
        "term": model.term_names_,
        "term_type": [
            "Main Feature" if len(term) == 1
            else "Interaction"
            for term in model.term_features_
        ],
        "mean_absolute_contribution":
            np.abs(values).mean(axis=0),
        "mean_signed_contribution":
            values.mean(axis=0)
    })

    importance = importance.sort_values(
        "mean_absolute_contribution",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "global_ebm_importance.csv",
        index=False
    )

    print("\nTop 10 global terms:")

    print(
        importance.head(10).to_string(index=False)
    )

    print("\nSelected feature interactions:")

    interactions = importance[
        importance["term_type"] == "Interaction"
    ]

    print(interactions.to_string(index=False))

    interactions.to_csv(
        OUTPUT_DIR / "interaction_importance.csv",
        index=False
    )

    print("\nMaximum reconstruction error:", error)

    top = importance.head(15).sort_values(
        "mean_absolute_contribution"
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top["term"],
        top["mean_absolute_contribution"]
    )

    plt.xlabel("Mean absolute contribution (log-odds)")
    plt.title("Global EBM Term Importance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "global_ebm_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# Local explanation chart
# --------------------------------------------------

def local_plot(model, row, values, category):

    local = pd.DataFrame({
        "term": model.term_names_,
        "contribution": values
    })

    local["absolute"] = (
        local["contribution"].abs()
    )

    local = local.nlargest(
        12,
        "absolute"
    ).sort_values("contribution")

    colors = [
        "firebrick" if value > 0 else "steelblue"
        for value in local["contribution"]
    ]

    plt.figure(figsize=(10, 7))

    plt.barh(
        local["term"],
        local["contribution"],
        color=colors
    )

    plt.axvline(0, color="black", linewidth=0.8)

    plt.xlabel("Contribution to risk (log-odds)")

    plt.title(
        f"{category}\n"
        f"Employee: {row['user']} | "
        f"Score: {row['score']:.4f}"
    )

    plt.tight_layout()

    filename = category.lower().replace(" ", "_")

    plt.savefig(
        OUTPUT_DIR / f"local_{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# Category-wise explanation
# --------------------------------------------------

def category_explanations(model, df):

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

        if category in [
            "True Positive",
            "False Negative"
        ]:

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

        values, intercept, error = get_contributions(
            model,
            X
        )

        for i, term in enumerate(model.term_names_):

            records.append({
                "category": category,
                "total_records": len(group),
                "explained_records": len(sample),
                "term": term,
                "term_type":
                    "Main Feature"
                    if len(model.term_features_[i]) == 1
                    else "Interaction",
                "mean_absolute_contribution":
                    np.abs(values[:, i]).mean(),
                "mean_signed_contribution":
                    values[:, i].mean()
            })

        for row_number, (_, row) in enumerate(
            sample.iterrows()
        ):

            for term_number, term in enumerate(
                model.term_names_
            ):

                local_records.append({
                    "user": row["user"],
                    "day": row["day"],
                    "category": category,
                    "label": row["label"],
                    "score": row["score"],
                    "rank": row["rank"],
                    "term": term,
                    "intercept": intercept,
                    "contribution":
                        values[row_number, term_number]
                })

        position = int(
            np.argmax(sample["score"].to_numpy())
        )

        local_plot(
            model,
            sample.iloc[position],
            values[position],
            category
        )

        print("Reconstruction error:", error)

    summary = pd.DataFrame(records)

    summary.to_csv(
        OUTPUT_DIR / "category_ebm_importance.csv",
        index=False
    )

    pd.DataFrame(local_records).to_csv(
        OUTPUT_DIR / "local_ebm_explanations.csv",
        index=False
    )

    print("\nTOP 5 TERMS BY CATEGORY")

    for category in categories:

        subset = summary[
            summary["category"] == category
        ]

        subset = subset.sort_values(
            "mean_absolute_contribution",
            ascending=False
        )

        print("\n", category)

        print(
            subset[
                [
                    "term",
                    "mean_absolute_contribution",
                    "mean_signed_contribution"
                ]
            ].head(5).to_string(index=False)
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 65)
    print("CONTROLLED EBM - INTRINSIC EXPLAINABILITY")
    print("=" * 65)

    model, df = load_data()

    print("\nTest records:", len(df))

    print("Total model terms:", len(model.term_names_))

    print(
        "Interaction terms:",
        sum(
            len(term) > 1
            for term in model.term_features_
        )
    )

    print("\nCONFUSION MATRIX AT TOP-10 ALERTS/DAY")

    counts = df["category"].value_counts()

    print(counts.to_string())

    counts.rename_axis("category").reset_index(
        name="count"
    ).to_csv(
        OUTPUT_DIR / "confusion_summary.csv",
        index=False
    )

    global_explanation(model, df)

    category_explanations(model, df)

    print("\n" + "=" * 65)
    print("EBM EXPLAINABILITY ANALYSIS COMPLETED")
    print("=" * 65)

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()