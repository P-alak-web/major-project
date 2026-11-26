from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from scipy.special import expit


ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = (
    ROOT / "processed"
    / "user_daily_features_advanced_v2.csv"
)

CONTROLLED_DIR = (
    ROOT / "outputs" / "eval" / "controlled"
)

MODEL_DIR = CONTROLLED_DIR / "models"

EXPLAINABILITY_DIR = (
    CONTROLLED_DIR / "explainability"
)

GROUND_TRUTH_FILE = (
    ROOT / "outputs" / "ground_truth"
    / "ground_truth_user_day.csv"
)

FEATURE_NAMES_FILE = (
    MODEL_DIR / "feature_columns.pkl"
)

COMPARISON_FILE = (
    CONTROLLED_DIR / "controlled_model_comparison.csv"
)


MODEL_FILES = {
    "EBM": "ebm",
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
    "Pooled Isolation Forest": "pooled_isolation_forest",
    "Per-user Isolation Forest": "per_user_isolation_forest"
}

DETECTION_MODELS = ["EBM", "XGBoost"]

DEFAULT_MODEL = "EBM"


# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data
def load_features():

    df = pd.read_csv(FEATURES_FILE)

    df["day"] = pd.to_datetime(df["day"])

    return df


@st.cache_data
def load_predictions(model_name=DEFAULT_MODEL):

    filename = MODEL_FILES[model_name]

    path = (
        CONTROLLED_DIR
        / f"predictions_{filename}.csv"
    )

    df = pd.read_csv(path)

    df["day"] = pd.to_datetime(df["day"])

    return df


@st.cache_data
def load_comparison():

    return pd.read_csv(COMPARISON_FILE)


@st.cache_data
def load_ground_truth():

    df = pd.read_csv(GROUND_TRUTH_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.rename(columns={"date": "day"})

    return df


@st.cache_resource
def load_feature_names():

    return list(joblib.load(FEATURE_NAMES_FILE))


@st.cache_resource
def load_model(model_name=DEFAULT_MODEL):

    filename = MODEL_FILES[model_name]

    model = joblib.load(
        MODEL_DIR / f"{filename}.pkl"
    )

    return model, load_feature_names()


@st.cache_resource
def load_shap_explainer():

    model, _ = load_model("XGBoost")

    return shap.TreeExplainer(
        model,
        model_output="raw"
    )


# ==========================================
# THREAT DETECTION
# ==========================================

def get_daily_alerts(
    day,
    top_n=10,
    model_name=DEFAULT_MODEL
):

    predictions = load_predictions(model_name)

    selected_day = pd.Timestamp(day).normalize()

    selected = predictions[
        predictions["day"] == selected_day
    ].copy()

    return (
        selected[selected["rank"] <= top_n]
        .sort_values("rank")
        .reset_index(drop=True)
    )


def get_user_history(
    user,
    model_name=DEFAULT_MODEL
):

    predictions = load_predictions(model_name)

    return (
        predictions[predictions["user"] == user]
        .sort_values("day")
        .reset_index(drop=True)
    )


def get_evaluation_dates(
    model_name=DEFAULT_MODEL
):

    predictions = load_predictions(model_name)

    return sorted(
        predictions["day"].dt.date.unique(),
        reverse=True
    )


def get_recommended_date(
    model_name=DEFAULT_MODEL
):

    predictions = load_predictions(model_name)

    daily = predictions.groupby("day").agg(
        employees=("user", "size"),
        highest_score=("score", "max")
    )

    eligible = daily[
        daily["employees"] >= 100
    ]

    if eligible.empty:
        eligible = daily

    return eligible["highest_score"].idxmax().date()


# ==========================================
# LOCAL EXPLAINABILITY
# ==========================================

def explain_employee_day(
    user,
    day,
    model_name=DEFAULT_MODEL
):

    model, feature_names = load_model(model_name)

    features = load_features()

    selected_day = pd.Timestamp(day).normalize()

    selected = features[
        (features["user"] == user)
        & (features["day"] == selected_day)
    ]

    if selected.empty:
        return None

    if len(selected) != 1:
        raise ValueError(
            "Duplicate employee-day records found."
        )

    X = selected[feature_names].fillna(0)

    score = float(
        model.predict_proba(X)[0, 1]
    )

    if model_name == "XGBoost":

        explainer = load_shap_explainer()

        explanation = explainer(X)

        values = np.asarray(
            explanation.values[0]
        )

        intercept = float(
            np.asarray(
                explanation.base_values
            ).ravel()[0]
        )

        contributions = pd.DataFrame({
            "feature": feature_names,
            "feature_value": X.iloc[0].values,
            "contribution": values,
            "term_type": "Main feature"
        })

        method = "TreeSHAP"

    elif model_name == "EBM":

        values = np.asarray(
            model.eval_terms(X)
        )[0]

        intercept = float(
            np.asarray(
                model.intercept_
            ).ravel()[0]
        )

        feature_values = []
        term_types = []

        for indices in model.term_features_:

            if len(indices) == 1:

                feature_values.append(
                    str(X.iloc[0, indices[0]])
                )

                term_types.append(
                    "Main feature"
                )

            else:

                text = " | ".join(
                    f"{feature_names[i]}={X.iloc[0, i]}"
                    for i in indices
                )

                feature_values.append(text)

                term_types.append(
                    "Interaction"
                )

        contributions = pd.DataFrame({
            "feature": model.term_names_,
            "feature_value": feature_values,
            "contribution": values,
            "term_type": term_types
        })

        method = "Intrinsic EBM contributions"

    else:

        raise ValueError(
            "Local explanations are supported "
            "only for EBM and XGBoost."
        )

    contributions["absolute_impact"] = (
        contributions["contribution"].abs()
    )

    reconstructed_score = float(
        expit(
            intercept
            + contributions["contribution"].sum()
        )
    )

    reconstruction_error = abs(
        score - reconstructed_score
    )

    contributions = contributions.sort_values(
        "absolute_impact",
        ascending=False
    ).reset_index(drop=True)

    return {
        "model": model_name,
        "method": method,
        "user": user,
        "day": selected_day,
        "score": score,
        "intercept": intercept,
        "reconstructed_score": reconstructed_score,
        "reconstruction_error": reconstruction_error,
        "contributions": contributions
    }


# ==========================================
# GROUND TRUTH VALIDATION
# ==========================================

@st.cache_data
def calculate_validation_metrics(
    model_name=DEFAULT_MODEL,
    budget=10
):

    predictions = load_predictions(
        model_name
    ).copy()

    actual = predictions["label"].eq(1)

    alerted = predictions["rank"].le(budget)

    tp = int((actual & alerted).sum())

    fp = int((~actual & alerted).sum())

    fn = int((actual & ~alerted).sum())

    tn = int((~actual & ~alerted).sum())

    precision = (
        tp / (tp + fp)
        if tp + fp else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall else 0
    )

    accuracy = (
        (tp + tn) / len(predictions)
    )

    false_positive_rate = (
        fp / (fp + tn)
        if fp + tn else 0
    )

    detected_insiders = predictions.loc[
        actual & alerted, "user"
    ].nunique()

    total_insiders = predictions.loc[
        actual, "user"
    ].nunique()

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "false_positive_rate": false_positive_rate,
        "total_alerts": tp + fp,
        "detected_insiders": detected_insiders,
        "total_insiders": total_insiders
    }


# ==========================================
# RESEARCH EVALUATION RESULTS
# ==========================================

@st.cache_data
def load_scenario_results():

    return pd.read_csv(
        EXPLAINABILITY_DIR
        / "scenarios"
        / "scenario_model_comparison.csv"
    )


@st.cache_data
def load_alignment_results():

    return pd.read_csv(
        EXPLAINABILITY_DIR
        / "alignment"
        / "overall_alignment.csv"
    )


@st.cache_data
def load_scenario_alignment():

    return pd.read_csv(
        EXPLAINABILITY_DIR
        / "alignment"
        / "scenario_alignment.csv"
    )


@st.cache_data
def load_stability_results():

    return pd.read_csv(
        EXPLAINABILITY_DIR
        / "stability"
        / "stability_summary.csv"
    )


@st.cache_data
def load_generalization_results():

    return pd.read_csv(
        CONTROLLED_DIR
        / "generalization"
        / "generalization_summary.csv"
    )


@st.cache_data
def load_explanation_agreement():

    return pd.read_csv(
        EXPLAINABILITY_DIR
        / "comparison"
        / "explanation_agreement_summary.csv"
    )


@st.cache_data
def load_explanation_performance():

    return pd.read_csv(
        EXPLAINABILITY_DIR
        / "comparison"
        / "explanation_performance.csv"
    )


@st.cache_data
def load_global_explanations(
    model_name=DEFAULT_MODEL
):

    if model_name == "EBM":

        path = (
            EXPLAINABILITY_DIR
            / "ebm"
            / "global_ebm_importance.csv"
        )

    elif model_name == "XGBoost":

        path = (
            EXPLAINABILITY_DIR
            / "xgboost"
            / "global_shap_importance.csv"
        )

    else:

        raise ValueError("Unsupported model.")

    return pd.read_csv(path)