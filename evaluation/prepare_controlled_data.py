import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    VALIDATION_DATE,
    TEST_DATE,
    FEATURE_COLUMNS
)

from models.train_xgboost import load_data


def prepare_data():

    df = load_data()

    # Check required features
    missing = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Check duplicate employee-days
    assert not df.duplicated(["user", "day"]).any()

    # Check binary labels
    assert set(df["label"].unique()).issubset({0, 1})

    # Handle missing feature values
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    # Chronological split
    train = df[
        df["day"] < VALIDATION_DATE
    ].copy()

    validation = df[
        (df["day"] >= VALIDATION_DATE)
        & (df["day"] < TEST_DATE)
    ].copy()

    test = df[
        df["day"] >= TEST_DATE
    ].copy()

    # Verify chronological separation
    assert train["day"].max() < validation["day"].min()
    assert validation["day"].max() < test["day"].min()

    assert len(train) + len(validation) + len(test) == len(df)

    for data in [train, validation, test]:
        assert data["label"].nunique() == 2

    return train, validation, test


def print_summary(name, data):

    print(f"\n{name}")
    print("-" * 50)

    print("Start date:", data["day"].min().date())
    print("End date:", data["day"].max().date())
    print("Employee-days:", len(data))
    print("Employees:", data["user"].nunique())
    print("Malicious days:", int(data["label"].sum()))
    print("Normal days:", int((data["label"] == 0).sum()))

    print(
        "Distinct insiders:",
        data.loc[data["label"] == 1, "user"].nunique()
    )

    print(
        "Positive prevalence:",
        round(data["label"].mean() * 100, 4),
        "%"
    )


def main():

    print("=" * 60)
    print("CONTROLLED MODEL COMPARISON - DATA PREPARATION")
    print("=" * 60)

    train, validation, test = prepare_data()

    print_summary("TRAINING SET", train)
    print_summary("VALIDATION SET", validation)
    print_summary("TESTING SET", test)

    print("\nCommon features:", len(FEATURE_COLUMNS))

    print("\nAll data validation checks passed!")


if __name__ == "__main__":
    main()