import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import shap

from scipy.stats import spearmanr

from config import FEATURE_COLUMNS, RANDOM_SEED
from evaluation.prepare_controlled_data import prepare_data


CONTROLLED_DIR = ROOT / "outputs" / "eval" / "controlled"
MODEL_DIR = CONTROLLED_DIR / "models"

OUTPUT_DIR = (
    CONTROLLED_DIR / "explainability" / "stability"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
PERTURBATION_LEVELS = [0.05, 0.10, 0.20]
REPEATS = 5

# Perturb only selected count features.
# Related totals and subsets must remain consistent.

PERTURBABLE_FEATURES = [
    "http_requests",
    "email_count",
    "file_operations"
]

LOWER_BOUNDS = {
    "http_requests": [
        "after_hours_http",
        "unique_domains",
        "cloud_storage_visits",
        "job_site_visits"
    ],
    "email_count": [
        "after_hours_email",
        "external_email_count"
    ],
    "file_operations": [
        "after_hours_file"
    ]
}


def perturb_data(X, level, rng):

    modified = X.copy()

    for feature in PERTURBABLE_FEATURES:

        original = X[feature].to_numpy(dtype=float)

        noise = rng.uniform(
            -level,
            level,
            size=len(X)
        )

        proposed = np.rint(
            original * (1 + noise)
        )

        lower_bound = X[
            LOWER_BOUNDS[feature]
        ].max(axis=1).to_numpy()

        proposed = np.maximum(
            proposed,
            lower_bound
        )

        # Preserve zero-valued activity counts.
        proposed = np.where(
            original == 0,
            original,
            proposed
        )

        # Avoid changing inconsistent source rows.
        proposed = np.where(
            original < lower_bound,
            original,
            proposed
        )

        modified[feature] = proposed

    return modified


def ebm_contributions(model, X):

    terms = np.asarray(
        model.eval_terms(X)
    )

    values = np.zeros(
        (len(X), len(FEATURE_COLUMNS))
    )

    for term_index, indices in enumerate(
        model.term_features_
    ):

        contribution = (
            terms[:, term_index] / len(indices)
        )

        for feature_index in indices:

            values[:, feature_index] += contribution

    return values


def compare_rows(
    name,
    sample,
    original_values,
    modified_values,
    original_scores,
    modified_scores,
    changed,
    level,
    repeat
):

    records = []

    for i in range(len(sample)):

        original = original_values[i]
        modified = modified_values[i]

        original_top = set(
            np.argsort(
                -np.abs(original),
                kind="stable"
            )[:TOP_K]
        )

        modified_top = set(
            np.argsort(
                -np.abs(modified),
                kind="stable"
            )[:TOP_K]
        )

        overlap = len(
            original_top & modified_top
        ) / TOP_K

        correlation = spearmanr(
            np.abs(original),
            np.abs(modified)
        ).statistic

        active = (
            (np.abs(original) > 1e-8)
            | (np.abs(modified) > 1e-8)
        )

        direction_agreement = (
            np.mean(
                np.sign(original[active])
                == np.sign(modified[active])
            )
            if active.any()
            else np.nan
        )

        score_change = abs(
            original_scores[i]
            - modified_scores[i]
        )

        records.append({

            "model": name,
            "user": sample.iloc[i]["user"],
            "day": sample.iloc[i]["day"],
            "label": sample.iloc[i]["label"],
            "perturbation": level,
            "repeat": repeat,
            "input_changed": bool(changed[i]),
            "top5_stability": overlap,
            "spearman_stability": correlation,
            "direction_agreement":
                direction_agreement,
            "absolute_score_change": score_change

        })

    return records


def main():

    print("=" * 65)
    print("EXPLANATION STABILITY EVALUATION")
    print("=" * 65)

    _, _, test = prepare_data()

    malicious = test[
        test["label"] == 1
    ].copy()

    normal = test[
        test["label"] == 0
    ].sample(
        n=401,
        random_state=RANDOM_SEED
    )

    sample = pd.concat(
        [malicious, normal],
        ignore_index=True
    )

    X = sample[FEATURE_COLUMNS].copy()

    print("\nTotal evaluated records:", len(sample))
    print("Malicious records:", len(malicious))
    print("Assumed-normal records:", len(normal))

    xgb = joblib.load(
        MODEL_DIR / "xgboost.pkl"
    )

    ebm = joblib.load(
        MODEL_DIR / "ebm.pkl"
    )

    explainer = shap.TreeExplainer(
        xgb,
        model_output="raw"
    )

    original_xgb = np.asarray(
        explainer(X).values
    )

    original_ebm = ebm_contributions(
        ebm, X
    )

    xgb_scores = xgb.predict_proba(X)[:, 1]
    ebm_scores = ebm.predict_proba(X)[:, 1]

    rng = np.random.default_rng(RANDOM_SEED)

    records = []

    for level in PERTURBATION_LEVELS:

        print(
            f"\nTesting perturbation: {level:.0%}"
        )

        for repeat in range(1, REPEATS + 1):

            modified = perturb_data(
                X, level, rng
            )

            changed = (
                modified.to_numpy()
                != X.to_numpy()
            ).any(axis=1)

            modified_xgb = np.asarray(
                explainer(modified).values
            )

            modified_ebm = ebm_contributions(
                ebm, modified
            )

            new_xgb_scores = (
                xgb.predict_proba(modified)[:, 1]
            )

            new_ebm_scores = (
                ebm.predict_proba(modified)[:, 1]
            )

            records.extend(
                compare_rows(
                    "XGBoost",
                    sample,
                    original_xgb,
                    modified_xgb,
                    xgb_scores,
                    new_xgb_scores,
                    changed,
                    level,
                    repeat
                )
            )

            records.extend(
                compare_rows(
                    "EBM",
                    sample,
                    original_ebm,
                    modified_ebm,
                    ebm_scores,
                    new_ebm_scores,
                    changed,
                    level,
                    repeat
                )
            )

            print(
                f"Repeat {repeat}/{REPEATS} | "
                f"Changed records: {changed.sum()}"
            )

    results = pd.DataFrame(records)

    results.to_csv(
        OUTPUT_DIR / "local_stability_results.csv",
        index=False
    )

    # Exclude unchanged inputs from stability averages.

    changed_results = results[
        results["input_changed"]
    ].copy()

    metrics = {

        "records": ("user", "size"),

        "mean_top5_stability": (
            "top5_stability", "mean"
        ),

        "mean_spearman": (
            "spearman_stability", "mean"
        ),

        "mean_direction_agreement": (
            "direction_agreement", "mean"
        ),

        "mean_score_change": (
            "absolute_score_change", "mean"
        )
    }

    summary = changed_results.groupby(
        ["model", "perturbation"]
    ).agg(**metrics).reset_index()

    category_summary = changed_results.groupby(
        ["model", "perturbation", "label"]
    ).agg(**metrics).reset_index()

    summary.to_csv(
        OUTPUT_DIR / "stability_summary.csv",
        index=False
    )

    category_summary.to_csv(
        OUTPUT_DIR / "stability_by_label.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("OVERALL EXPLANATION STABILITY")
    print("=" * 65)

    print(summary.round(6).to_string(index=False))

    print("\n" + "=" * 65)
    print("STABILITY BY LABEL")
    print("=" * 65)

    print(
        category_summary.round(6)
        .to_string(index=False)
    )

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()