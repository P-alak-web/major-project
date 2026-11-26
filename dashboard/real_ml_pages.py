import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.ml_backend import (
    DETECTION_MODELS,
    load_predictions,
    load_features,
    load_comparison,
    load_feature_names,
    get_daily_alerts,
    get_user_history,
    get_evaluation_dates,
    get_recommended_date,
    explain_employee_day,
)


# ==========================================
# COMMON COMPONENTS
# ==========================================

def select_model(key):

    return st.selectbox(
        "Detection Model",
        DETECTION_MODELS,
        key=key
    )


def select_date(model_name, key):

    dates = get_evaluation_dates(model_name)

    recommended = get_recommended_date(model_name)

    index = (
        dates.index(recommended)
        if recommended in dates else 0
    )

    return st.selectbox(
        "Evaluation Date",
        dates,
        index=index,
        key=key
    )


def score_warning():

    st.caption(
        "Model scores are uncalibrated outputs, "
        "not verified probabilities of malicious intent."
    )


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================

def render_overview():

    st.caption(
        "Explainable Insider Threat Detection "
        "using User Behavior Analytics"
    )

    predictions = load_predictions("EBM")
    features = load_features()
    comparison = load_comparison()

    positives = predictions[
        predictions["label"] == 1
    ]

    ebm = comparison[
        comparison["model"] == "EBM"
    ].iloc[0]

    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Employee-Days",
        f"{len(features):,}"
    )

    c2.metric(
        "Evaluated Employees",
        predictions["user"].nunique()
    )

    c3.metric(
        "Test Employee-Days",
        f"{len(predictions):,}"
    )

    c4.metric(
        "Behavioral Features",
        len(load_feature_names())
    )

    st.subheader("Primary Model Performance — EBM")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "PR-AUC (AP)",
        f"{ebm['pr_auc']:.4f}"
    )

    c2.metric(
        "Recall@10/day",
        f"{ebm['recall_at_10']:.2%}"
    )

    c3.metric(
        "Precision@10/day",
        f"{ebm['precision_at_10']:.2%}"
    )

    c4.metric(
        "Insiders Detected",
        f"{int(ebm['insiders_at_10'])}/"
        f"{int(ebm['test_insiders'])}"
    )

    st.subheader("Evaluation Dataset")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Known Malicious Days",
        len(positives)
    )

    c2.metric(
        "Known Test Insiders",
        positives["user"].nunique()
    )

    c3.metric(
        "Positive Prevalence",
        f"{predictions['label'].mean():.3%}"
    )

    test_features = features.merge(
        predictions[["user", "day"]],
        on=["user", "day"],
        how="inner",
        validate="one_to_one"
    )

    st.subheader("Daily Employee Activity")

    activity = st.selectbox(
        "Activity Metric",
        [
            "logon_count",
            "http_requests",
            "email_count",
            "usb_events",
            "file_operations",
        ],
        key="overview_activity"
    )

    daily = (
        test_features.groupby("day")[activity]
        .sum()
        .reset_index()
    )

    fig = px.line(
        daily,
        x="day",
        y=activity,
        title="Daily Behavioral Activity"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "This system evaluates historical synthetic "
        "CERT r4.2 employee activity. "
        "It is not a live production monitoring system."
    )


# ==========================================
# PAGE 2: USER BEHAVIOR ANALYTICS
# ==========================================

def render_user_behaviour():

    st.caption(
        "Historical employee behavioral analysis"
    )

    model_name = select_model("behavior_model")

    features = load_features()

    predictions = load_predictions(model_name)

    user = st.selectbox(
        "Select Employee",
        sorted(predictions["user"].unique()),
        key="behavior_user"
    )

    history = features[
        features["user"] == user
    ].sort_values("day")

    risk_history = get_user_history(
        user,
        model_name
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Recorded Activity Days",
        len(history)
    )

    c2.metric(
        "Evaluation Days",
        len(risk_history)
    )

    c3.metric(
        "Maximum Model Score",
        f"{risk_history['score'].max():.4f}"
    )

    c4.metric(
        "Top-10 Alert Days",
        int(risk_history["rank"].le(10).sum())
    )

    st.subheader("Employee Behavioral History")

    activity = st.selectbox(
        "Behavioral Feature",
        load_feature_names(),
        key="behavior_feature"
    )

    fig = px.line(
        history,
        x="day",
        y=activity,
        title=f"{activity} — {user}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Model Score History")

    fig = px.line(
        risk_history,
        x="day",
        y="score",
        title=f"{model_name} Score History",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    score_warning()

    st.subheader("Daily Behavioral Records")

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Employee Activity",
        history.to_csv(index=False),
        file_name=f"{user}_behavior.csv",
        mime="text/csv"
    )


# ==========================================
# PAGE 3: THREAT DETECTION
# ==========================================

def render_threat_detection():

    st.caption(
        "Employee-day threat prioritization "
        "using controlled ML predictions"
    )

    model_name = select_model("threat_model")

    selected_date = select_date(
        model_name,
        f"threat_date_{model_name}"
    )

    budget = st.selectbox(
        "Maximum Daily Alerts",
        [10, 50, 100],
        key="threat_budget"
    )

    predictions = load_predictions(model_name)

    alerts = get_daily_alerts(
        selected_date,
        budget,
        model_name
    )

    daily = predictions[
        predictions["day"].dt.date == selected_date
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Employees Evaluated",
        len(daily)
    )

    c2.metric(
        "Prioritized Alerts",
        len(alerts)
    )

    c3.metric(
        "Highest Model Score",
        f"{alerts['score'].max():.4f}"
    )

    st.subheader("Ranked Threat Alerts")

    display = alerts[
        ["rank", "user", "day", "score"]
    ]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        display,
        x="user",
        y="score",
        color="score",
        color_continuous_scale="OrRd",
        title=f"Top {budget} Alerts — {model_name}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "Download Daily Alerts",
        display.to_csv(index=False),
        file_name=(
            f"{model_name}_alerts_{selected_date}.csv"
        ),
        mime="text/csv"
    )

    score_warning()

    st.info(
        "Alerts are ranked by model score. "
        "The selected budget is a maximum number "
        "of records to investigate per day, "
        "not a probability threshold."
    )


# ==========================================
# PAGE 4: EXPLAINABLE AI
# ==========================================

def render_explainable_ai():

    st.caption(
        "Intrinsic EBM explanations "
        "and XGBoost TreeSHAP explanations"
    )

    model_name = select_model("explanation_model")

    selected_date = select_date(
        model_name,
        f"explanation_date_{model_name}"
    )

    alerts = get_daily_alerts(
        selected_date,
        10,
        model_name
    )

    if alerts.empty:

        st.warning("No records available.")

        return

    user = st.selectbox(
        "Select Employee From Top-10 Alerts",
        alerts["user"].tolist(),
        key=f"explanation_user_{model_name}"
    )

    with st.spinner("Generating explanation..."):

        explanation = explain_employee_day(
            user,
            selected_date,
            model_name
        )

    if explanation is None:

        st.error("Employee features not found.")

        return

    contributions = explanation["contributions"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model Score",
        f"{explanation['score']:.6f}"
    )

    c2.metric(
        "Reconstructed Score",
        f"{explanation['reconstructed_score']:.6f}"
    )

    c3.metric(
        "Reconstruction Error",
        f"{explanation['reconstruction_error']:.2e}"
    )

    st.caption(
        f"Explanation Method: {explanation['method']}"
    )

    saved_score = float(
        alerts.loc[
            alerts["user"] == user,
            "score"
        ].iloc[0]
    )

    if not np.isclose(
        saved_score,
        explanation["score"],
        atol=1e-6,
        rtol=1e-6
    ):

        st.error(
            "Saved prediction and loaded model "
            "score do not match."
        )

    if explanation["reconstruction_error"] <= 1e-6:

        st.success(
            "Explanation reconstruction verified."
        )

    else:

        st.warning(
            "Explanation reconstruction exceeds "
            "the numerical tolerance."
        )

    st.subheader("Feature Contributions")

    top = contributions.head(15).copy()

    top["Direction"] = np.where(
        top["contribution"] >= 0,
        "Non-negative contribution",
        "Negative contribution"
    )

    fig = px.bar(
        top.sort_values("contribution"),
        x="contribution",
        y="feature",
        orientation="h",
        color="Direction",
        color_discrete_map={
            "Non-negative contribution": "#DC2626",
            "Negative contribution": "#16A34A"
        },
        title=f"{model_name} Explanation — {user}",
        labels={
            "contribution": "Log-odds contribution",
            "feature": "Feature / Interaction"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Complete Explanation")

    st.dataframe(
        contributions,
        use_container_width=True,
        hide_index=True
    )

    if model_name == "EBM":

        st.subheader("Feature Interactions")

        interactions = contributions[
            contributions["term_type"] == "Interaction"
        ]

        st.dataframe(
            interactions,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "EBM models individual behavioral features "
            "and selected pairwise feature interactions."
        )

    st.download_button(
        "Download Explanation",
        contributions.to_csv(index=False),
        file_name=(
            f"{model_name}_{user}_"
            f"{selected_date}_explanation.csv"
        ),
        mime="text/csv"
    )

    st.warning(
        "Positive contributions increase the model's "
        "raw score and negative contributions decrease it. "
        "These values are not probability percentages "
        "or evidence of causal malicious behavior."
    )


# ==========================================
# PAGE 5: GLOBAL ML EVALUATION
# ==========================================

def render_global_evaluation():

    st.caption(
        "Controlled comparison of five ML models "
        "using 22 common behavioral features"
    )

    comparison = load_comparison()

    budget = st.selectbox(
        "Daily Alert Budget",
        [10, 50, 100],
        key="evaluation_budget"
    )

    columns = [
        "model",
        "pr_auc",
        f"precision_at_{budget}",
        f"recall_at_{budget}",
        f"insiders_at_{budget}",
        "test_insiders"
    ]

    display = comparison[columns].copy()

    display.columns = [
        "Model",
        "PR-AUC (AP)",
        "Precision",
        "Recall",
        "Insiders Detected",
        "Total Insiders"
    ]

    st.subheader("Controlled Model Comparison")

    st.dataframe(
        display.style.format({
            "PR-AUC (AP)": "{:.4f}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        comparison,
        x="model",
        y="pr_auc",
        color="model",
        title="Precision-Recall Performance",
        text_auto=".4f"
    )

    fig.add_hline(
        y=comparison["random_baseline"].iloc[0],
        line_dash="dash",
        annotation_text="Random baseline"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Recall Across Alert Budgets")

    recall_data = comparison[
        [
            "model",
            "recall_at_10",
            "recall_at_50",
            "recall_at_100"
        ]
    ].melt(
        id_vars="model",
        var_name="Alert Budget",
        value_name="Recall"
    )

    recall_data["Alert Budget"] = (
        recall_data["Alert Budget"].map({
            "recall_at_10": "10/day",
            "recall_at_50": "50/day",
            "recall_at_100": "100/day"
        })
    )

    fig = px.bar(
        recall_data,
        x="model",
        y="Recall",
        color="Alert Budget",
        barmode="group"
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Validation vs Test Performance")

    validation_data = comparison[
        [
            "model",
            "validation_pr_auc",
            "pr_auc"
        ]
    ].melt(
        id_vars="model",
        var_name="Evaluation Split",
        value_name="PR-AUC"
    )

    validation_data["Evaluation Split"] = (
        validation_data["Evaluation Split"].map({
            "validation_pr_auc": "Validation",
            "pr_auc": "Test"
        })
    )

    fig = px.bar(
        validation_data,
        x="model",
        y="PR-AUC",
        color="Evaluation Split",
        barmode="group"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        "Download Controlled Evaluation Results",
        comparison.to_csv(index=False),
        file_name="controlled_model_comparison.csv",
        mime="text/csv"
    )

    st.info(
        "Training: January–August 2010. "
        "Validation: September–October 2010. "
        "Final model fitting: January–October 2010. "
        "Testing: November 2010–May 2011."
    )

    st.warning(
        "CERT r4.2 is synthetic and highly imbalanced. "
        "The reported test period has been examined "
        "during development. Results do not establish "
        "production deployment performance."
    )