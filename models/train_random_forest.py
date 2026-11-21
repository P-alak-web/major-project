import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent

FEATURES = ROOT / "processed/user_daily_features_advanced_v2.csv"
TRUTH = ROOT / "outputs/ground_truth/ground_truth_user_day.csv"

OUTPUT = ROOT / "outputs/eval/predictions_random_forest.csv"

SPLIT_DATE = pd.Timestamp("2010-11-01")


def main():
    print("Loading enhanced features...")

    df = pd.read_csv(FEATURES)
    truth = pd.read_csv(TRUTH)

    df["day"] = pd.to_datetime(df["day"])
    truth["date"] = pd.to_datetime(truth["date"])

    truth = truth.rename(columns={"date": "day"})

    truth = (
        truth.groupby(["user", "day"], as_index=False)["label"]
        .max()
    )

    df = df.merge(
        truth,
        on=["user", "day"],
        how="left",
        validate="one_to_one"
    )

    df["label"] = df["label"].fillna(0).astype(int)

    feature_cols = [
        c for c in df.columns
        if c not in ["user", "day", "label"]
    ]

    train = df[df["day"] < SPLIT_DATE].copy()
    test = df[df["day"] >= SPLIT_DATE].copy()

    print("Training rows:", len(train))
    print("Testing rows:", len(test))
    print("Training positives:", train["label"].sum())
    print("Testing positives:", test["label"].sum())

    X_train = train[feature_cols].fillna(0)
    y_train = train["label"]

    X_test = test[feature_cols].fillna(0)
    y_test = test["label"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, scores)

    print("\nRandom Forest Evaluation")
    print("PR-AUC:", round(pr_auc, 5))

    predictions = test[["user", "day"]].copy()
    predictions["score"] = scores

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT, index=False)

    print("Saved predictions to:", OUTPUT)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 10 Important Features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()