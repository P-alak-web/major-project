from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st


ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = (
    ROOT / "processed" / "user_daily_features_advanced_v2.csv"
)

PREDICTIONS_FILE = (
    ROOT / "outputs" / "eval" / "predictions_xgboost.csv"
)

MODEL_FILE = (
    ROOT / "models" / "saved" / "xgboost_model.pkl"
)

FEATURE_NAMES_FILE = (
    ROOT / "models" / "saved" / "xgboost_features.pkl"
)

COMPARISON_FILE = (
    ROOT / "outputs" / "eval" / "all_models_comparison.csv"
)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)

    return model, feature_names


@st.cache_data
def load_features():
    df = pd.read_csv(FEATURES_FILE)
    df["day"] = pd.to_datetime(df["day"])
    return df


@st.cache_data
def load_predictions():
    df = pd.read_csv(PREDICTIONS_FILE)
    df["day"] = pd.to_datetime(df["day"])
    return df


@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_FILE)


def get_daily_alerts(day, top_n=10):
    predictions = load_predictions()

    selected = predictions[
        predictions["day"].dt.date == day
    ]

    return selected.nlargest(top_n, "score")


def get_user_history(user):
    predictions = load_predictions()

    return (
        predictions[predictions["user"] == user]
        .sort_values("day")
    )


def explain_employee_day(user, day):
    model, feature_names = load_model()
    features = load_features()

    selected = features[
        (features["user"] == user)
        & (features["day"].dt.date == day)
    ]

    if selected.empty:
        return None

    X = selected[feature_names].fillna(0)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    contributions = pd.DataFrame({
        "feature": feature_names,
        "feature_value": X.iloc[0].values,
        "shap_value": explanation.values[0],
    })

    contributions["absolute_impact"] = (
        contributions["shap_value"].abs()
    )

    return contributions.sort_values(
        "absolute_impact",
        ascending=False
    )