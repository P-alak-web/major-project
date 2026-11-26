import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import shap

from config import FEATURE_COLUMNS, TEST_DATE
from evaluation.prepare_controlled_data import prepare_data


CONTROLLED_DIR = ROOT / "outputs" / "eval" / "controlled"

MODEL_DIR = CONTROLLED_DIR / "models"

OUTPUT_DIR = (
    CONTROLLED_DIR / "explainability" / "alignment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_FILE = (
    ROOT / "outputs" / "ground_truth"
    / "ground_truth_events.csv"
)

TOP_K = 5

ACTIVITY_FEATURES = {

    "logon": [
        "total_logon_events",
        "logon_count",
        "logoff_count",
        "after_hours_logons",
        "new_pc_count",
        "unique_pc_count"
    ],

    "device": [
        "usb_events",
        "usb_connect_count",
        "usb_disconnect_count",
        "after_hours_usb"
    ],

    "http": [
        "http_requests",
        "after_hours_http",
        "unique_domains",
        "new_domain_count",
        "cloud_storage_visits",
        "job_site_visits"
    ],

    "email": [
        "email_count",
        "after_hours_email",
        "external_email_count",
        "external_attachment_count"
    ],

    "file": [
        "file_operations",
        "after_hours_file"
    ]
}


def load_ground_truth():

    events = pd.read_csv(EVENT_FILE)

    events["day"] = pd.to_datetime(
        events["date"]
    ).dt.normalize()

    events["activity_type"] = (
        events["activity_type"]
        .str.strip()
        .str.lower()
    )

    events = events[
        (events["day"] >= TEST_DATE)
        & (events["label"] == 1)
    ].copy()

    events = events.drop_duplicates(
        ["event_id", "user", "day", "activity_type"]
    )

    unknown = (
        set(events["activity_type"].unique())
        - set(ACTIVITY_FEATURES)
    )

    if unknown:
        raise ValueError(
            f"Unmapped activity types: {unknown}"
        )

    print("\nMALICIOUS TEST EVENTS BY SCENARIO")

    distribution = pd.crosstab(
        events["scenario"],
        events["activity_type"]
    )

    print(distribution.to_string())

    distribution.to_csv(
        OUTPUT_DIR / "scenario_activity_distribution.csv"
    )

    grouped = events.groupby(
        ["user", "day"]
    ).agg(

        scenario=(
            "scenario",
            lambda x: ",".join(
                sorted(set(x.astype(str)))
            )
        ),

        activity_types=(
            "activity_type",
            lambda x: sorted(set(x))
        ),

        malicious_events=("event_id", "size")

    ).reset_index()

    return grouped


def load_predictions(filename):

    predictions = pd.read_csv(
        CONTROLLED_DIR
        / f"predictions_{filename}.csv"
    )

    predictions["day"] = pd.to_datetime(
        predictions["day"]
    )

    return predictions[
        ["user", "day", "score", "rank"]
    ].rename(columns={
        "score": f"{filename}_score",
        "rank": f"{filename}_rank"
    })


def get_ebm_contributions(model, X):

    terms = np.asarray(
        model.eval_terms(X)
    )

    values = np.zeros(
        (len(X), len(FEATURE_COLUMNS))
    )

    for term_index, feature_indices in enumerate(
        model.term_features_
    ):

        contribution = (
            terms[:, term_index]
            / len(feature_indices)
        )

        for feature_index in feature_indices:

            values[:, feature_index] += contribution

    assert np.allclose(
        values.sum(axis=1),
        terms.sum(axis=1)
    )

    return values


def evaluate_alignment(name, values, sample):

    records = []

    for i, row in sample.iterrows():

        activities = set(
            row["activity_types"]
        )

        expected_features = set()

        for activity in activities:

            expected_features.update(
                ACTIVITY_FEATURES[activity]
            )

        contributions = values[i]

        # Select only risk-increasing contributions.
        positive_indices = np.where(
            contributions > 0
        )[0]

        ranked = positive_indices[
            np.argsort(
                -contributions[positive_indices],
                kind="stable"
            )
        ]

        selected = ranked[:TOP_K]

        top_features = {
            FEATURE_COLUMNS[j]
            for j in selected
        }

        matched_features = (
            top_features & expected_features
        )

        matched_activities = {

            activity

            for activity in activities

            if top_features.intersection(
                ACTIVITY_FEATURES[activity]
            )
        }

        positive_mass = np.maximum(
            contributions, 0
        ).sum()

        expected_indices = [

            j

            for j, feature in enumerate(
                FEATURE_COLUMNS
            )

            if feature in expected_features
        ]

        relevant_mass = np.maximum(
            contributions[expected_indices],
            0
        ).sum()

        if name == "XGBoost":
            rank = row["xgboost_rank"]
        else:
            rank = row["ebm_rank"]

        records.append({

            "model": name,

            "user": row["user"],

            "day": row["day"],

            "scenario": row["scenario"],

            "malicious_events":
                row["malicious_events"],

            "detected_at_10": int(rank <= 10),

            "activity_types":
                ", ".join(sorted(activities)),

            "expected_features":
                ", ".join(sorted(expected_features)),

            "top_features":
                ", ".join(
                    FEATURE_COLUMNS[j]
                    for j in selected
                ),

            "matched_features":
                ", ".join(sorted(matched_features)),

            "selected_feature_count":
                len(top_features),

            "feature_alignment": (
                len(matched_features)
                / len(top_features)
                if top_features else np.nan
            ),

            "activity_coverage": (
                len(matched_activities)
                / len(activities)
            ),

            "positive_attribution_alignment": (
                relevant_mass / positive_mass
                if positive_mass > 0 else np.nan
            ),

            "hit_at_5": int(
                len(matched_features) > 0
            ),

            "expected_feature_fraction": (
                len(expected_features)
                / len(FEATURE_COLUMNS)
            )
        })

    return pd.DataFrame(records)


def main():

    print("=" * 65)
    print("GROUND TRUTH EXPLANATION ALIGNMENT")
    print("=" * 65)

    _, _, test = prepare_data()

    truth = load_ground_truth()

    malicious = test[
        test["label"] == 1
    ].copy()

    sample = malicious.merge(

        truth,

        on=["user", "day"],

        how="left",

        validate="one_to_one"
    )

    assert sample["activity_types"].notna().all(), (
        "Missing event-level ground truth."
    )

    sample = sample.merge(
        load_predictions("xgboost"),
        on=["user", "day"],
        validate="one_to_one"
    )

    sample = sample.merge(
        load_predictions("ebm"),
        on=["user", "day"],
        validate="one_to_one"
    )

    sample = sample.reset_index(drop=True)

    assert len(sample) == len(malicious)

    print("\nMalicious employee-days:", len(sample))

    X = sample[FEATURE_COLUMNS]

    xgb = joblib.load(
        MODEL_DIR / "xgboost.pkl"
    )

    ebm = joblib.load(
        MODEL_DIR / "ebm.pkl"
    )

    print("\nGenerating XGBoost SHAP values...")

    explainer = shap.TreeExplainer(
        xgb,
        model_output="raw"
    )

    xgb_values = np.asarray(
        explainer(X).values
    )

    print("Generating EBM contributions...")

    ebm_values = get_ebm_contributions(
        ebm, X
    )

    xgb_results = evaluate_alignment(
        "XGBoost", xgb_values, sample
    )

    ebm_results = evaluate_alignment(
        "EBM", ebm_values, sample
    )

    results = pd.concat(
        [xgb_results, ebm_results],
        ignore_index=True
    )

    results.to_csv(
        OUTPUT_DIR / "local_alignment_results.csv",
        index=False
    )

    metrics = {

        "records": ("user", "size"),

        "mean_feature_alignment": (
            "feature_alignment", "mean"
        ),

        "mean_activity_coverage": (
            "activity_coverage", "mean"
        ),

        "mean_positive_attribution_alignment": (
            "positive_attribution_alignment", "mean"
        ),

        "hit_at_5": (
            "hit_at_5", "mean"
        ),

        "random_feature_baseline": (
            "expected_feature_fraction", "mean"
        ),

        "valid_feature_alignment_records": (
            "feature_alignment", "count"
        ),

        "valid_attribution_records": (
            "positive_attribution_alignment", "count"
        )
    }

    overall = results.groupby(
        "model"
    ).agg(**metrics).reset_index()

    scenario_summary = results.groupby(
        ["model", "scenario"]
    ).agg(**metrics).reset_index()

    detection_summary = results.groupby(
        ["model", "detected_at_10"]
    ).agg(**metrics).reset_index()

    overall.to_csv(
        OUTPUT_DIR / "overall_alignment.csv",
        index=False
    )

    scenario_summary.to_csv(
        OUTPUT_DIR / "scenario_alignment.csv",
        index=False
    )

    detection_summary.to_csv(
        OUTPUT_DIR / "detection_alignment.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("OVERALL EXPLANATION ALIGNMENT")
    print("=" * 65)

    print(overall.round(5).to_string(index=False))

    print("\n" + "=" * 65)
    print("SCENARIO-WISE EXPLANATION ALIGNMENT")
    print("=" * 65)

    print(
        scenario_summary.round(5).to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("DETECTED VS MISSED EXPLANATION ALIGNMENT")
    print("=" * 65)

    print(
        detection_summary.round(5).to_string(index=False)
    )

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()