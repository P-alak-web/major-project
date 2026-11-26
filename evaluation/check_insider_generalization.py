import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DATE = pd.Timestamp("2010-11-01")

truth = pd.read_csv(
    ROOT / "outputs/ground_truth/ground_truth_user_day.csv"
)

pred = pd.read_csv(
    ROOT / "outputs/eval/predictions_xgboost.csv"
)

truth["date"] = pd.to_datetime(truth["date"])
pred["day"] = pd.to_datetime(pred["day"])

train_insiders = set(
    truth.loc[
        (truth["date"] < SPLIT_DATE)
        & (truth["label"] == 1),
        "user"
    ]
)

test_truth = truth[
    truth["date"] >= SPLIT_DATE
].copy()

test_truth = (
    test_truth.groupby(
        ["user", "date"],
        as_index=False
    )["label"].max()
)

pred = pred.sort_values(
    ["day", "score", "user"],
    ascending=[True, False, True]
).copy()

pred["rank"] = (
    pred.groupby("day").cumcount() + 1
)

pred["top10"] = pred["rank"] <= 10

merged = pred.merge(
    test_truth,
    left_on=["user", "day"],
    right_on=["user", "date"],
    how="left",
    validate="one_to_one"
)

merged["label"] = (
    merged["label"].fillna(0).astype(int)
)

positives = merged[merged["label"] == 1].copy()

positives["group"] = positives["user"].apply(
    lambda u: (
        "Previously known insider"
        if u in train_insiders
        else "New test insider"
    )
)

print("\nXGBoost generalization audit")
print("=" * 60)

for group, data in positives.groupby("group"):

    insider_count = data["user"].nunique()
    positive_days = len(data)

    detected_days = int(data["top10"].sum())

    detected_insiders = (
        data.loc[data["top10"], "user"].nunique()
    )

    recall = (
        detected_days / positive_days
        if positive_days else 0
    )

    print(f"\nGroup: {group}")
    print("Insiders:", insider_count)
    print("Malicious employee-days:", positive_days)
    print("Detected malicious days:", detected_days)
    print("Recall@10/day:", round(recall, 4))
    print(
        "Distinct insiders detected:",
        f"{detected_insiders}/{insider_count}"
    )