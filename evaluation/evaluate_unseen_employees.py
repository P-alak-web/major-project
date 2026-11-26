import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.train_xgboost import load_data, train_model, SPLIT_DATE


OUTPUT_DIR = ROOT / "outputs" / "eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
HOLDOUT_FRACTION = 0.20
N_REPEATS = 5


def evaluate_one_split(df, seed):
    """
    Hold out 20% of employees completely from training.

    Training:
        Non-held-out employees before SPLIT_DATE.

    Testing:
        Held-out employees on or after SPLIT_DATE.
    """

    rng = np.random.default_rng(seed)

    # Identify employees with records in the test period.
    test_period = df[df["day"] >= SPLIT_DATE]

    all_users = np.array(
        sorted(test_period["user"].unique())
    )

    # Stratify employees by whether they have known
    # positive days in the test period.
    positive_users = np.array(
        sorted(
            test_period.loc[
                test_period["label"] == 1, "user"
            ].unique()
        )
    )

    negative_users = np.array(
        sorted(set(all_users) - set(positive_users))
    )

    heldout_positive = rng.choice(
        positive_users,
        size=max(1, round(len(positive_users) * HOLDOUT_FRACTION)),
        replace=False,
    )

    heldout_negative = rng.choice(
        negative_users,
        size=max(1, round(len(negative_users) * HOLDOUT_FRACTION)),
        replace=False,
    )

    heldout_users = set(heldout_positive) | set(heldout_negative)

    train = df[
        (df["day"] < SPLIT_DATE)
        & (~df["user"].isin(heldout_users))
    ].copy()

    test = df[
        (df["day"] >= SPLIT_DATE)
        & (df["user"].isin(heldout_users))
    ].copy()

    feature_cols = [
        col for col in df.columns
        if col not in ["user", "day", "label"]
    ]

    X_train = train[feature_cols].fillna(0)
    y_train = train["label"]

    X_test = test[feature_cols].fillna(0)
    y_test = test["label"]

    assert set(train["user"]).isdisjoint(set(test["user"]))

    print("\n" + "=" * 60)
    print(f"HELD-OUT EMPLOYEE EVALUATION — SEED {seed}")
    print("=" * 60)

    print("Training employees:", train["user"].nunique())
    print("Testing employees:", test["user"].nunique())
    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print("Training positives:", int(y_train.sum()))
    print("Testing positives:", int(y_test.sum()))

    if y_train.sum() == 0 or y_test.sum() == 0:
        raise ValueError(
            "This split has no positive training or test examples."
        )

    model = train_model(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, scores)

    predictions = test[["user", "day", "label"]].copy()
    predictions["score"] = scores

    predictions = predictions.sort_values(
        ["day", "score", "user"],
        ascending=[True, False, True],
    )

    predictions["rank"] = (
        predictions.groupby("day").cumcount() + 1
    )

    predictions["top10"] = predictions["rank"] <= 10

    positives = predictions[predictions["label"] == 1]

    detected_days = int(positives["top10"].sum())
    total_positive_days = len(positives)

    recall_at_10 = detected_days / total_positive_days

    detected_insiders = positives.loc[
        positives["top10"], "user"
    ].nunique()

    total_insiders = positives["user"].nunique()

    print("\nRESULTS")
    print("PR-AUC:", round(pr_auc, 5))
    print("Random baseline:", round(float(y_test.mean()), 5))
    print("Recall@10/day:", round(recall_at_10, 5))
    print(
        "Insiders detected:",
        f"{detected_insiders}/{total_insiders}",
    )

    return {
        "seed": seed,
        "train_employees": train["user"].nunique(),
        "test_employees": test["user"].nunique(),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positives": int(y_train.sum()),
        "test_positives": int(y_test.sum()),
        "test_prevalence": float(y_test.mean()),
        "pr_auc": pr_auc,
        "recall_at_10": recall_at_10,
        "insiders_detected": detected_insiders,
        "total_test_insiders": total_insiders,
    }


def main():
    print("=" * 60)
    print("CERT r4.2 — UNSEEN EMPLOYEE GENERALIZATION")
    print("=" * 60)

    df = load_data()

    results = []

    for i in range(N_REPEATS):
        result = evaluate_one_split(df, SEED + i)
        results.append(result)

    results_df = pd.DataFrame(results)

    output = OUTPUT_DIR / "xgboost_unseen_employee_results.csv"
    results_df.to_csv(output, index=False)

    print("\n" + "=" * 60)
    print("SUMMARY ACROSS FIVE SPLITS")
    print("=" * 60)

    print(
        results_df[
            [
                "seed",
                "test_employees",
                "test_positives",
                "test_prevalence",
                "pr_auc",
                "recall_at_10",
                "insiders_detected",
                "total_test_insiders",
            ]
        ].to_string(index=False)
    )

    print("\nMean PR-AUC:", round(results_df["pr_auc"].mean(), 5))
    print("Std PR-AUC:", round(results_df["pr_auc"].std(), 5))

    print(
        "Mean Recall@10/day:",
        round(results_df["recall_at_10"].mean(), 5),
    )

    print("\nSaved results:", output)


if __name__ == "__main__":
    main()