import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.ml_backend import (
    load_predictions,
    load_features,
    load_comparison,
    get_daily_alerts,
    get_user_history,
    explain_employee_day,
)


def render_overview():
    st.caption("CERT r4.2 | Actual XGBoost test predictions")

    predictions = load_predictions()
    features = load_features()

    test_features = features[
        features["day"].isin(predictions["day"])
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Employees Evaluated",
        f"{predictions['user'].nunique():,}"
    )

    c2.metric(
        "Employee-Days Evaluated",
        f"{len(predictions):,}"
    )

    c3.metric(
        "Evaluation Days",
        f"{predictions['day'].nunique():,}"
    )

    c4.metric(
        "Behavioral Features",
        str(len(features.columns) - 2)
    )

    st.subheader("Daily Employee Activity")

    daily = (
        test_features.groupby("day", as_index=False)
        .agg(
            logons=("logon_count", "sum"),
            http_requests=("http_requests", "sum"),
            usb_events=("usb_events", "sum"),
            file_operations=("file_operations", "sum"),
        )
    )

    selected_metric = st.selectbox(
        "Activity metric",
        [
            "logons",
            "http_requests",
            "usb_events",
            "file_operations",
        ]
    )

    fig = px.line(
        daily,
        x="day",
        y=selected_metric,
        markers=True,
        title=f"Daily {selected_metric.replace('_', ' ').title()}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This dashboard displays historical CERT r4.2 "
        "evaluation results, not live employee monitoring."
    )


def render_user_behaviour():
    st.caption("Actual employee behavioral features")

    features = load_features()
    predictions = load_predictions()

    user = st.selectbox(
        "Select Employee",
        sorted(predictions["user"].unique()),
        key="behavior_user"
    )

    history = features[
        features["user"] == user
    ].sort_values("day")

    st.subheader("Employee Activity Overview")

    selected_features = [
        "logon_count",
        "http_requests",
        "usb_events",
        "file_operations",
    ]

    daily = history[
        ["day"] + selected_features
    ].melt(
        id_vars="day",
        var_name="Activity",
        value_name="Count"
    )

    fig = px.line(
        daily,
        x="day",
        y="Count",
        color="Activity",
        title=f"Behavior History: {user}"
    )

    st.plotly_chart(fig, use_container_width=True)

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


def render_threat_detection():
    st.caption("XGBoost | Historical CERT r4.2 predictions")

    predictions = load_predictions()

    dates = sorted(
        predictions["day"].dt.date.unique(),
        reverse=True
    )

    selected_date = st.selectbox(
        "Select Evaluation Date",
        dates,
        key="threat_date"
    )

    budget = st.selectbox(
        "Maximum Alerts Per Day",
        [10, 50, 100],
        key="threat_budget"
    )

    alerts = get_daily_alerts(
        selected_date,
        budget
    ).copy()

    alerts = alerts.sort_values(
        "score",
        ascending=False
    ).reset_index(drop=True)

    alerts.insert(
        0,
        "daily_rank",
        alerts.index + 1
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Employees Evaluated",
        f"{len(predictions[predictions['day'].dt.date == selected_date]):,}"
    )

    c2.metric(
        "Alerts Displayed",
        len(alerts)
    )

    c3.metric(
        "Highest Model Score",
        f"{alerts['score'].max():.4f}"
    )

    st.subheader("Daily Ranked Threat Alerts")

    st.dataframe(
        alerts,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        alerts,
        x="user",
        y="score",
        title="Top Employee Risk Scores",
        labels={
            "user": "Employee",
            "score": "XGBoost Score"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Daily Alerts",
        alerts.to_csv(index=False),
        file_name=f"xgboost_alerts_{selected_date}.csv",
        mime="text/csv"
    )

    st.warning(
        "Scores are uncalibrated model outputs. "
        "A high score does not prove malicious behavior."
    )


def render_explainable_ai():
    st.caption("Actual XGBoost SHAP explanations")

    predictions = load_predictions()

    dates = sorted(
        predictions["day"].dt.date.unique(),
        reverse=True
    )

    selected_date = st.selectbox(
        "Select Evaluation Date",
        dates,
        key="shap_date"
    )

    daily_alerts = get_daily_alerts(
        selected_date,
        top_n=10
    )

    user = st.selectbox(
        "Select Employee From Top 10 Alerts",
        daily_alerts["user"].tolist(),
        key="shap_user"
    )

    selected_prediction = daily_alerts[
        daily_alerts["user"] == user
    ].iloc[0]

    st.metric(
        "XGBoost Risk Score",
        f"{selected_prediction['score']:.6f}"
    )

    if st.button("Generate SHAP Explanation"):
        with st.spinner("Calculating SHAP values..."):
            explanation = explain_employee_day(
                user,
                selected_date
            )

        if explanation is None:
            st.error("No behavioral features found.")
            return

        st.subheader("Feature Contributions")

        top_features = explanation.head(15)

        fig = px.bar(
            top_features.sort_values("shap_value"),
            x="shap_value",
            y="feature",
            orientation="h",
            color="shap_value",
            color_continuous_scale="RdBu_r",
            title=f"SHAP Explanation: {user}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            explanation,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download SHAP Explanation",
            explanation.to_csv(index=False),
            file_name=f"{user}_{selected_date}_shap.csv",
            mime="text/csv"
        )

        st.info(
            "Positive SHAP values increase the model's "
            "raw risk score; negative values decrease it. "
            "They are not probability percentages."
        )


def render_global_evaluation():
    st.caption(
        "CERT r4.2 | Five-model comparison | "
        "Chronological split: 2010-11-01"
    )

    comparison = load_comparison()

    st.subheader("Model Performance Comparison")

    display_columns = [
        "model",
        "pr_auc",
        "precision@10/day",
        "recall@10/day",
        "insiders_found@10/day",
        "total_insiders",
    ]

    st.dataframe(
        comparison[display_columns],
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        comparison,
        x="model",
        y="pr_auc",
        title="PR-AUC Comparison",
        text="pr_auc"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily Alert Recall")

    recall_data = comparison[
        [
            "model",
            "recall@10/day",
            "recall@50/day",
            "recall@100/day",
        ]
    ].melt(
        id_vars="model",
        var_name="Alert Budget",
        value_name="Recall"
    )

    fig = px.bar(
        recall_data,
        x="model",
        y="Recall",
        color="Alert Budget",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "These are preliminary benchmark results. "
        "Feature leakage and unseen-employee generalization "
        "have not yet been fully evaluated."
    )