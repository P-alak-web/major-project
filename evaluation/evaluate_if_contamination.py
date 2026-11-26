import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.run_global_eval import ACTIVITY

FEATURES_FILE = ROOT / "processed" / "user_daily_features.csv"
TRUTH_FILE = ROOT / "outputs" / "ground_truth" / "ground_truth_user_day.csv"

OUTPUT_DIR = ROOT / "outputs" / "eval" / "contamination"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DATE = pd.Timestamp("2010-11-01")
MIN_TRAIN_DAYS = 30
BUDGETS = [10, 50, 100]
SEED = 42


def load_dataset():
    print("Loading behavioral features...")

    df = pd.read_csv(FEATURES_FILE)
    df["day"] = pd.to_datetime(df["day"])

    truth = pd.read_csv(TRUTH_FILE)
    truth["day"] = pd.to_datetime(truth["date"])

    # Preserve only known positive employee-days.
    truth = truth.loc[
        truth["label"] == 1,
        ["user", "day", "label"]
    ]

    truth = truth.groupby(
        ["user", "day"],
        as_index=False
    )["label"].max()

    df = df.merge(
        truth,
        on=["user", "day"],
        how="left",
        validate="one_to_one"
    )

    df["label"] = df["label"].fillna(0).astype(int)
    df[ACTIVITY] = df[ACTIVITY].fillna(0)

    return df


def prepare_data(df):
    train_all = df[df["day"] < SPLIT_DATE].copy()
    test_all = df[df["day"] >= SPLIT_DATE].copy()

    counts = train_all.groupby("user").size()

    eligible_users = set(
        counts[counts >= MIN_TRAIN_DAYS].index
    )

    train = train_all[
        train_all["user"].isin(eligible_users)
    ].copy()

    test = test_all[
        test_all["user"].isin(eligible_users)
    ].copy()

    print("\nDATA COVERAGE")
    print("Total test rows:", len(test_all))
    print("Eligible test rows:", len(test))
    print("Excluded test rows:", len(test_all) - len(test))

    print("Total test positives:", int(test_all["label"].sum()))
    print("Eligible test positives:", int(test["label"].sum()))

    print("Training rows:", len(train))
    print("Known positive training rows:", int(train["label"].sum()))

    return train, test


def create_relative_features(train, test):
    """
    Calculate one shared historical baseline.

    Both Setting A and Setting B use exactly the same
    transformation to isolate the effect of IF contamination.
    """

    baseline = train[train["label"] == 0]

    grouped = baseline.groupby("user")[ACTIVITY]

    median = grouped.median()

    iqr = (
        grouped.quantile(0.75)
        - grouped.quantile(0.25)
    ).replace(0, 1)

    def transform(data):
        x = (
            data[ACTIVITY].to_numpy()
            - median.loc[data["user"]].to_numpy()
        ) / iqr.loc[data["user"]].to_numpy()

        x = pd.DataFrame(
            x,
            columns=ACTIVITY,
            index=data.index
        )

        x["is_weekend"] = (
            data["day"].dt.dayofweek >= 5
        ).astype(int)

        return x

    return transform(train), transform(test)


def train_and_score(train, test, x_train, x_test, setting):
    """
    Setting A: all training employee-days.
    Setting B: exclude known positive training days.
    """

    if setting == "A":
        mask = np.ones(len(train), dtype=bool)
    else:
        mask = train["label"].to_numpy() == 0

    model = IsolationForest(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(x_train.loc[mask])

    scores = -model.decision_function(x_test)

    result = test[["user", "day", "label"]].copy()
    result["score"] = scores

    return result


def evaluate(predictions, setting):
    predictions = predictions.sort_values(
        ["day", "score", "user"],
        ascending=[True, False, True]
    ).copy()

    predictions["rank"] = (
        predictions.groupby("day").cumcount() + 1
    )

    positives = predictions[
        predictions["label"] == 1
    ]

    result = {
        "setting": setting,
        "test_rows": len(predictions),
        "test_positives": len(positives),
        "test_insiders": positives["user"].nunique(),
        "pr_auc": average_precision_score(
            predictions["label"],
            predictions["score"]
        ),
        "random_baseline": predictions["label"].mean()
    }

    for k in BUDGETS:
        alerts = predictions[
            predictions["rank"] <= k
        ]

        hits = alerts[alerts["label"] == 1]

        result[f"precision_at_{k}"] = (
            len(hits) / len(alerts)
            if len(alerts) else 0
        )

        result[f"recall_at_{k}"] = (
            len(hits) / len(positives)
            if len(positives) else 0
        )

        result[f"insiders_at_{k}"] = (
            hits["user"].nunique()
        )

    return result


def main():
    print("=" * 60)
    print("ISOLATION FOREST CONTAMINATION ABLATION")
    print("=" * 60)

    df = load_dataset()

    train, test = prepare_data(df)

    x_train, x_test = create_relative_features(train, test)

    results = []
    predictions_by_setting = {}

    for setting in ["A", "B"]:
        print(f"\nTraining Setting {setting}...")

        pred = train_and_score(
            train,
            test,
            x_train,
            x_test,
            setting
        )

        predictions_by_setting[setting] = pred

        metrics = evaluate(pred, setting)
        results.append(metrics)

        pred.to_csv(
            OUTPUT_DIR / f"predictions_setting_{setting}.csv",
            index=False
        )

        print("PR-AUC:", round(metrics["pr_auc"], 5))
        print(
            "Recall@10/day:",
            round(metrics["recall_at_10"], 5)
        )

    # Verify identical test employee-days.
    keys_a = predictions_by_setting["A"][
        ["user", "day"]
    ].sort_values(["user", "day"]).reset_index(drop=True)

    keys_b = predictions_by_setting["B"][
        ["user", "day"]
    ].sort_values(["user", "day"]).reset_index(drop=True)

    assert keys_a.equals(keys_b), (
        "Setting A and B scored different employee-days!"
    )

    summary = pd.DataFrame(results)

    summary.to_csv(
        OUTPUT_DIR / "contamination_comparison.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(summary.to_string(index=False))

    print("\nSaved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()