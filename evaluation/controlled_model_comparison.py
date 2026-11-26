import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import average_precision_score

from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier

from config import (
    FEATURE_COLUMNS,
    RANDOM_SEED,
    MIN_TRAIN_DAYS,
    ALERT_BUDGETS
)

from evaluation.prepare_controlled_data import prepare_data


OUTPUT_DIR = ROOT / "outputs" / "eval" / "controlled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Candidate hyperparameters
# --------------------------------------------------

PARAMETER_GRID = {

    "XGBoost": [
        {"n_estimators": 200, "max_depth": 3},
        {"n_estimators": 300, "max_depth": 5},
        {"n_estimators": 400, "max_depth": 6}
    ],

    "Random Forest": [
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 300, "max_depth": 20},
        {"n_estimators": 300, "max_depth": None}
    ],

    "EBM": [
        {"interactions": 0, "max_rounds": 300},
        {"interactions": 5, "max_rounds": 300},
        {"interactions": 10, "max_rounds": 500}
    ],

    "Pooled Isolation Forest": [
        {"n_estimators": 200, "max_samples": 256},
        {"n_estimators": 300, "max_samples": 512},
        {"n_estimators": 300, "max_samples": 1024}
    ],

    "Per-user Isolation Forest": [
        {"n_estimators": 100, "max_samples": "auto"},
        {"n_estimators": 200, "max_samples": "auto"}
    ]
}


# --------------------------------------------------
# 2. Train model
# --------------------------------------------------

def fit_model(name, train, params):

    X = train[FEATURE_COLUMNS]
    y = train["label"]

    if name == "XGBoost":

        weight = (y == 0).sum() / (y == 1).sum()

        model = XGBClassifier(
            **params,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        model.fit(X, y)

    elif name == "Random Forest":

        model = RandomForestClassifier(
            **params,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        model.fit(X, y)

    elif name == "EBM":

        model = ExplainableBoostingClassifier(
            **params,
            learning_rate=0.05,
            max_bins=128,
            outer_bags=4,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        model.fit(X, y)

    elif name == "Pooled Isolation Forest":

        model = IsolationForest(
            **params,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        model.fit(X)

    elif name == "Per-user Isolation Forest":

        model = {}

        for user, group in train.groupby("user"):

            if len(group) < MIN_TRAIN_DAYS:
                continue

            employee_model = IsolationForest(
                **params,
                random_state=RANDOM_SEED,
                n_jobs=1
            )

            employee_model.fit(group[FEATURE_COLUMNS])

            model[user] = employee_model

    else:
        raise ValueError(f"Unknown model: {name}")

    return model


# --------------------------------------------------
# 3. Generate scores
# --------------------------------------------------

def predict_scores(name, model, data):

    X = data[FEATURE_COLUMNS]

    if name in ["XGBoost", "Random Forest", "EBM"]:

        scores = model.predict_proba(X)[:, 1]

    elif name == "Pooled Isolation Forest":

        scores = -model.decision_function(X)

    elif name == "Per-user Isolation Forest":

        scores = pd.Series(
            np.nan,
            index=data.index,
            dtype=float
        )

        for user, group in data.groupby("user"):

            if user not in model:
                raise ValueError(
                    f"No trained model for employee {user}"
                )

            scores.loc[group.index] = (
                -model[user].decision_function(
                    group[FEATURE_COLUMNS]
                )
            )

        scores = scores.to_numpy()

    else:
        raise ValueError(f"Unknown model: {name}")

    if not np.isfinite(scores).all():
        raise ValueError("Invalid prediction scores found.")

    return scores


# --------------------------------------------------
# 4. Validation-based hyperparameter selection
# --------------------------------------------------

def select_parameters(name, train, validation):

    best_score = -np.inf
    best_params = None
    records = []

    for number, params in enumerate(
        PARAMETER_GRID[name], start=1
    ):

        print(f"\nCandidate {number}: {params}", flush=True)

        model = fit_model(name, train, params)

        scores = predict_scores(
            name,
            model,
            validation
        )

        ap = average_precision_score(
            validation["label"],
            scores
        )

        print("Validation PR-AUC:", round(ap, 5))

        records.append({
            "model": name,
            "parameters": json.dumps(params),
            "validation_pr_auc": ap
        })

        if ap > best_score:

            best_score = ap
            best_params = params.copy()

        del model

    print("\nBest parameters:", best_params)
    print("Best validation PR-AUC:", round(best_score, 5))

    return best_params, best_score, records


# --------------------------------------------------
# 5. Evaluate test predictions
# --------------------------------------------------

def evaluate(name, data, scores):

    predictions = data[
        ["user", "day", "label"]
    ].copy()

    predictions["score"] = scores

    predictions = predictions.sort_values(
        ["day", "score", "user"],
        ascending=[True, False, True]
    )

    predictions["rank"] = (
        predictions.groupby("day").cumcount() + 1
    )

    positives = predictions[
        predictions["label"] == 1
    ]

    total_insiders = positives["user"].nunique()

    results = {
        "model": name,
        "test_rows": len(predictions),
        "test_positives": len(positives),
        "test_insiders": total_insiders,
        "pr_auc": average_precision_score(
            predictions["label"],
            predictions["score"]
        ),
        "random_baseline": predictions["label"].mean()
    }

    for k in ALERT_BUDGETS:

        alerts = predictions[
            predictions["rank"] <= k
        ]

        hits = alerts[
            alerts["label"] == 1
        ]

        results[f"precision_at_{k}"] = (
            len(hits) / len(alerts)
        )

        results[f"recall_at_{k}"] = (
            len(hits) / len(positives)
        )

        results[f"insiders_at_{k}"] = (
            hits["user"].nunique()
        )

    return results, predictions


# --------------------------------------------------
# 6. Main experiment
# --------------------------------------------------

def main():

    print("=" * 65)
    print("CONTROLLED FIVE-MODEL COMPARISON")
    print("=" * 65)

    train, validation, test = prepare_data()

    final_train = pd.concat(
        [train, validation],
        ignore_index=True
    )

    print("\nTraining rows:", len(train))
    print("Validation rows:", len(validation))
    print("Final training rows:", len(final_train))
    print("Testing rows:", len(test))
    print("Common features:", len(FEATURE_COLUMNS))

    final_results = []
    validation_results = []
    selected_parameters = {}

    # Select parameters using validation data only.
    for name in PARAMETER_GRID:

        print("\n" + "=" * 65)
        print("VALIDATING:", name)
        print("=" * 65)

        best_params, best_ap, records = select_parameters(
            name,
            train,
            validation
        )

        validation_results.extend(records)

        selected_parameters[name] = {
            "parameters": best_params,
            "validation_pr_auc": best_ap
        }

        pd.DataFrame(validation_results).to_csv(
            OUTPUT_DIR / "validation_results.csv",
            index=False
        )

    with open(
        OUTPUT_DIR / "selected_parameters.json",
        "w"
    ) as file:

        json.dump(
            selected_parameters,
            file,
            indent=4
        )

    # Retrain using training + validation data.
    for name, selection in selected_parameters.items():

        print("\n" + "=" * 65)
        print("FINAL TESTING:", name)
        print("=" * 65)

        model = fit_model(
            name,
            final_train,
            selection["parameters"]
        )

        scores = predict_scores(
            name,
            model,
            test
        )

        metrics, predictions = evaluate(
            name,
            test,
            scores
        )

        metrics["validation_pr_auc"] = (
            selection["validation_pr_auc"]
        )

        final_results.append(metrics)

        filename = name.lower().replace(" ", "_").replace("-", "_")

        predictions.to_csv(
            OUTPUT_DIR / f"predictions_{filename}.csv",
            index=False
        )

        joblib.dump(
            model,
            MODEL_DIR / f"{filename}.pkl",
            compress=3
        )

        print("Test PR-AUC:", round(metrics["pr_auc"], 5))

        print(
            "Recall@10/day:",
            round(metrics["recall_at_10"], 5)
        )

        pd.DataFrame(final_results).to_csv(
            OUTPUT_DIR / "controlled_model_comparison.csv",
            index=False
        )

        del model

    summary = pd.DataFrame(final_results)

    summary = summary.sort_values(
        "pr_auc",
        ascending=False
    )

    summary.to_csv(
        OUTPUT_DIR / "controlled_model_comparison.csv",
        index=False
    )

    joblib.dump(
        FEATURE_COLUMNS,
        MODEL_DIR / "feature_columns.pkl"
    )

    print("\n" + "=" * 65)
    print("FINAL CONTROLLED COMPARISON")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()