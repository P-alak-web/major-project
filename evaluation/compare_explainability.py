import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import shap

from scipy.stats import spearmanr
from scipy.special import expit

from config import FEATURE_COLUMNS, RANDOM_SEED
from evaluation.prepare_controlled_data import prepare_data


CONTROLLED_DIR = ROOT / "outputs" / "eval" / "controlled"

MODEL_DIR = CONTROLLED_DIR / "models"

OUTPUT_DIR = CONTROLLED_DIR / "explainability" / "comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 5
ALERT_BUDGET = 10


def main():

    print("=" * 65)
    print("XGBOOST VS EBM - EXPLAINABILITY COMPARISON")
    print("=" * 65)

    _, _, test = prepare_data()

    xgb = joblib.load(MODEL_DIR / "xgboost.pkl")
    ebm = joblib.load(MODEL_DIR / "ebm.pkl")

    # Load prediction rankings.

    xgb_predictions = pd.read_csv(
        CONTROLLED_DIR / "predictions_xgboost.csv"
    )

    ebm_predictions = pd.read_csv(
        CONTROLLED_DIR / "predictions_ebm.csv"
    )

    xgb_predictions["day"] = pd.to_datetime(
        xgb_predictions["day"]
    )

    ebm_predictions["day"] = pd.to_datetime(
        ebm_predictions["day"]
    )

    xgb_predictions = xgb_predictions[
        ["user", "day", "score", "rank"]
    ].rename(columns={
        "score": "xgb_score",
        "rank": "xgb_rank"
    })

    ebm_predictions = ebm_predictions[
        ["user", "day", "score", "rank"]
    ].rename(columns={
        "score": "ebm_score",
        "rank": "ebm_rank"
    })

    df = test.merge(
        xgb_predictions,
        on=["user", "day"],
        validate="one_to_one"
    )

    df = df.merge(
        ebm_predictions,
        on=["user", "day"],
        validate="one_to_one"
    )

    assert len(df) == len(test)

    df["xgb_alert"] = df["xgb_rank"] <= ALERT_BUDGET
    df["ebm_alert"] = df["ebm_rank"] <= ALERT_BUDGET

    # Explain all malicious days.

    malicious = df[df["label"] == 1].copy()

    # Sample assumed-normal days flagged by either model.

    false_alerts = df[
        (df["label"] == 0)
        & (df["xgb_alert"] | df["ebm_alert"])
    ]

    false_alerts = false_alerts.sample(
        n=min(401, len(false_alerts)),
        random_state=RANDOM_SEED
    )

    # Sample assumed-normal days not alerted by either.

    normal = df[
        (df["label"] == 0)
        & (~df["xgb_alert"])
        & (~df["ebm_alert"])
    ]

    normal = normal.sample(
        n=min(401, len(normal)),
        random_state=RANDOM_SEED
    )

    sample = pd.concat(
        [malicious, false_alerts, normal],
        ignore_index=True
    )

    X = sample[FEATURE_COLUMNS]

    print("\nCommon explained records:", len(sample))
    print("Malicious records:", len(malicious))
    print("Assumed-normal alert records:", len(false_alerts))
    print("Assumed-normal non-alert records:", len(normal))

    # --------------------------------------------------
    # XGBoost SHAP
    # --------------------------------------------------

    print("\nGenerating XGBoost SHAP explanations...")

    start = time.perf_counter()

    explainer = shap.TreeExplainer(
        xgb,
        model_output="raw"
    )

    xgb_setup_time = time.perf_counter() - start

    start = time.perf_counter()

    shap_explanation = explainer(X)

    xgb_time = time.perf_counter() - start

    xgb_values = np.asarray(
        shap_explanation.values
    )

    xgb_base = np.asarray(
        shap_explanation.base_values
    ).reshape(-1)

    xgb_reconstructed = expit(
        xgb_base + xgb_values.sum(axis=1)
    )

    xgb_error = np.max(
        np.abs(
            xgb_reconstructed
            - xgb.predict_proba(X)[:, 1]
        )
    )

    # --------------------------------------------------
    # EBM intrinsic explanations
    # --------------------------------------------------

    print("Generating EBM intrinsic explanations...")

    start = time.perf_counter()

    ebm_terms = np.asarray(
        ebm.eval_terms(X)
    )

    ebm_time = time.perf_counter() - start

    intercept = float(
        np.asarray(ebm.intercept_).ravel()[0]
    )

    ebm_reconstructed = expit(
        intercept + ebm_terms.sum(axis=1)
    )

    ebm_error = np.max(
        np.abs(
            ebm_reconstructed
            - ebm.predict_proba(X)[:, 1]
        )
    )

    # Allocate interaction contributions equally
    # among their participating features.
    # This is a comparison convention, not SHAP.

    ebm_values = np.zeros(
        (len(X), len(FEATURE_COLUMNS))
    )

    for term_index, feature_indices in enumerate(
        ebm.term_features_
    ):

        contribution = (
            ebm_terms[:, term_index]
            / len(feature_indices)
        )

        for feature_index in feature_indices:

            ebm_values[:, feature_index] += contribution

    assert np.allclose(
        ebm_values.sum(axis=1),
        ebm_terms.sum(axis=1)
    )

    # --------------------------------------------------
    # Compare feature rankings
    # --------------------------------------------------

    records = []

    for i in range(len(sample)):

        xgb_abs = np.abs(xgb_values[i])
        ebm_abs = np.abs(ebm_values[i])

        xgb_top = set(
            np.argsort(-xgb_abs, kind="stable")[:TOP_K]
        )

        ebm_top = set(
            np.argsort(-ebm_abs, kind="stable")[:TOP_K]
        )

        intersection = len(xgb_top & ebm_top)
        union = len(xgb_top | ebm_top)

        correlation = spearmanr(
            xgb_abs,
            ebm_abs
        ).statistic

        row = sample.iloc[i]

        if row["label"] == 1:

            if row["xgb_alert"] and row["ebm_alert"]:
                group = "Detected by both"

            elif row["xgb_alert"]:
                group = "Detected only by XGBoost"

            elif row["ebm_alert"]:
                group = "Detected only by EBM"

            else:
                group = "Missed by both"

        elif row["xgb_alert"] or row["ebm_alert"]:

            group = "Assumed-normal alert"

        else:

            group = "Assumed-normal non-alert"

        records.append({

            "user": row["user"],
            "day": row["day"],
            "label": row["label"],
            "group": group,

            "xgb_score": row["xgb_score"],
            "ebm_score": row["ebm_score"],

            "top5_common_features": intersection,

            "top5_overlap": intersection / TOP_K,

            "jaccard_similarity": intersection / union,

            "spearman_correlation": correlation,

            "xgb_top_features": ", ".join(
                FEATURE_COLUMNS[j]
                for j in np.argsort(
                    -xgb_abs, kind="stable"
                )[:TOP_K]
            ),

            "ebm_top_features": ", ".join(
                FEATURE_COLUMNS[j]
                for j in np.argsort(
                    -ebm_abs, kind="stable"
                )[:TOP_K]
            )
        })

    results = pd.DataFrame(records)

    results.to_csv(
        OUTPUT_DIR / "local_explanation_agreement.csv",
        index=False
    )

    summary = results.groupby("group").agg(

        records=("user", "size"),

        mean_common_features=(
            "top5_common_features", "mean"
        ),

        mean_top5_overlap=(
            "top5_overlap", "mean"
        ),

        mean_jaccard=(
            "jaccard_similarity", "mean"
        ),

        mean_spearman=(
            "spearman_correlation", "mean"
        ),

        valid_spearman_records=(
            "spearman_correlation", "count"
        )

    ).reset_index()

    summary.to_csv(
        OUTPUT_DIR / "explanation_agreement_summary.csv",
        index=False
    )

    # --------------------------------------------------
    # Global comparison on common records
    # --------------------------------------------------

    global_importance = pd.DataFrame({

        "feature": FEATURE_COLUMNS,

        "xgb_mean_absolute_shap":
            np.abs(xgb_values).mean(axis=0),

        "ebm_mean_absolute_contribution":
            np.abs(ebm_values).mean(axis=0)
    })

    global_importance["xgb_rank"] = (
        global_importance[
            "xgb_mean_absolute_shap"
        ].rank(ascending=False)
    )

    global_importance["ebm_rank"] = (
        global_importance[
            "ebm_mean_absolute_contribution"
        ].rank(ascending=False)
    )

    global_importance.to_csv(
        OUTPUT_DIR / "global_feature_comparison.csv",
        index=False
    )

    global_correlation = spearmanr(

        global_importance[
            "xgb_mean_absolute_shap"
        ],

        global_importance[
            "ebm_mean_absolute_contribution"
        ]

    ).statistic

    # --------------------------------------------------
    # Fidelity and runtime
    # --------------------------------------------------

    performance = pd.DataFrame([

        {
            "model": "XGBoost + SHAP",
            "explained_records": len(sample),
            "setup_seconds": xgb_setup_time,
            "explanation_seconds": xgb_time,
            "milliseconds_per_record":
                xgb_time * 1000 / len(sample),
            "max_reconstruction_error": xgb_error
        },

        {
            "model": "EBM",
            "explained_records": len(sample),
            "setup_seconds": 0,
            "explanation_seconds": ebm_time,
            "milliseconds_per_record":
                ebm_time * 1000 / len(sample),
            "max_reconstruction_error": ebm_error
        }
    ])

    performance.to_csv(
        OUTPUT_DIR / "explanation_performance.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("EXPLANATION AGREEMENT BY CATEGORY")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nGlobal Spearman correlation:",
          round(global_correlation, 5))

    print("\nOverall mean top-5 overlap:",
          round(results["top5_overlap"].mean(), 5))

    print("\n" + "=" * 65)
    print("EXPLANATION FIDELITY AND RUNTIME")
    print("=" * 65)

    print(performance.to_string(index=False))

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()