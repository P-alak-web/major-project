import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import MIN_TRAIN_DAYS
from evaluation.prepare_controlled_data import prepare_data


train, validation, test = prepare_data()


def check_eligibility(history, target, name):

    counts = history.groupby("user").size()

    eligible_users = counts[
        counts >= MIN_TRAIN_DAYS
    ].index

    eligible = target[
        target["user"].isin(eligible_users)
    ]

    excluded = target[
        ~target["user"].isin(eligible_users)
    ]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Total employees:", target["user"].nunique())
    print("Eligible employees:", eligible["user"].nunique())
    print("Excluded employees:", excluded["user"].nunique())

    print("Total rows:", len(target))
    print("Eligible rows:", len(eligible))
    print("Excluded rows:", len(excluded))

    print("Total positives:", int(target["label"].sum()))
    print("Eligible positives:", int(eligible["label"].sum()))
    print("Excluded positives:", int(excluded["label"].sum()))

    print(
        "Coverage:",
        round(len(eligible) / len(target) * 100, 4),
        "%"
    )


# Validation: only training history is available
check_eligibility(
    train,
    validation,
    "VALIDATION ELIGIBILITY"
)


# Final testing: training + validation history
final_train = pd.concat(
    [train, validation],
    ignore_index=True
)

check_eligibility(
    final_train,
    test,
    "TEST ELIGIBILITY"
)