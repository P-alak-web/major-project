import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.ml_backend import (
    DETECTION_MODELS,
    load_predictions,
    calculate_validation_metrics,
    load_scenario_results,
    load_alignment_results,
    load_stability_results,
    load_generalization_results,
)


# ==========================================
# PAGE 1: GROUND TRUTH VALIDATION
# ==========================================

def render_ground_truth_validation():

    st.caption(
        "Validation of model alerts against "
        "CERT r4.2 ground truth labels"
    )

    c1, c2 = st.columns(2)

    model_name = c1.selectbox(
        "Detection Model",
        DETECTION_MODELS,
        key="validation_model"
    )

    budget = c2.selectbox(
        "Daily Alert Budget",
        [10, 50, 100],
        key="validation_budget"
    )

    metrics = calculate_validation_metrics(
        model_name,
        budget
    )

    st.subheader("Detection Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Precision",
        f"{metrics['precision']:.2%}"
    )

    c2.metric(
        "Recall",
        f"{metrics['recall']:.2%}"
    )

    c3.metric(
        "F1-Score",
        f"{metrics['f1']:.2%}"
    )

    c4.metric(
        "False Positive Rate",
        f"{metrics['false_positive_rate']:.2%}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Alerts",
        f"{metrics['total_alerts']:,}"
    )

    c2.metric(
        "Insiders Detected",
        f"{metrics['detected_insiders']}/"
        f"{metrics['total_insiders']}"
    )

    c3.metric(
        "Accuracy",
        f"{metrics['accuracy']:.2%}"
    )

    st.caption(
        "Accuracy is secondary because the dataset "
        "contains very few known malicious records."
    )

    # Confusion matrix

    st.subheader("Confusion Matrix")

    matrix = [
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Not Alerted", "Alerted"],
            y=["Assumed Normal", "Known Malicious"],
            colorscale="Blues",
            text=matrix,
            texttemplate="%{text:,}",
            showscale=False
        )
    )

    fig.update_layout(
        xaxis_title="Model Decision",
        yaxis_title="Ground Truth",
        height=400
    )

    fig.update_yaxes(
        autorange="reversed"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("True Positives", metrics["tp"])
    c2.metric("False Positives", metrics["fp"])
    c3.metric("False Negatives", metrics["fn"])
    c4.metric("True Negatives", metrics["tn"])

    # Alert composition

    st.subheader("Alert Composition")

    alert_data = pd.DataFrame({
        "Category": [
            "Known Malicious Alerts",
            "Assumed-Normal Alerts"
        ],
        "Records": [
            metrics["tp"],
            metrics["fp"]
        ]
    })

    fig = px.pie(
        alert_data,
        names="Category",
        values="Records",
        hole=0.45,
        color="Category",
        color_discrete_map={
            "Known Malicious Alerts": "#DC2626",
            "Assumed-Normal Alerts": "#64748B"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Record-level validation

    predictions = load_predictions(
        model_name
    ).copy()

    predictions["alerted"] = (
        predictions["rank"] <= budget
    )

    detected = predictions[
        (predictions["label"] == 1)
        & predictions["alerted"]
    ]

    missed = predictions[
        (predictions["label"] == 1)
        & (~predictions["alerted"])
    ]

    false_alerts = predictions[
        (predictions["label"] == 0)
        & predictions["alerted"]
    ]

    st.subheader("Record-Level Validation")

    tab1, tab2, tab3 = st.tabs([
        "Detected Malicious Days",
        "Missed Malicious Days",
        "Assumed-Normal Alerts"
    ])

    columns = [
        "user",
        "day",
        "score",
        "rank",
        "label"
    ]

    with tab1:

        st.write(
            f"Detected records: {len(detected):,}"
        )

        st.dataframe(
            detected[columns],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Detected Records",
            detected[columns].to_csv(index=False),
            file_name="detected_malicious_days.csv",
            mime="text/csv"
        )

    with tab2:

        st.write(
            f"Missed records: {len(missed):,}"
        )

        st.dataframe(
            missed[columns],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Missed Records",
            missed[columns].to_csv(index=False),
            file_name="missed_malicious_days.csv",
            mime="text/csv"
        )

    with tab3:

        st.write(
            f"Assumed-normal alerts: {len(false_alerts):,}"
        )

        st.dataframe(
            false_alerts[columns],
            use_container_width=True,
            hide_index=True
        )

    st.warning(
        "Ground truth is used only for retrospective "
        "evaluation. Unlabelled employee-days are "
        "treated as normal for metric calculation."
    )


# ==========================================
# PAGE 2: ATTACK SCENARIO ANALYSIS
# ==========================================

def render_attack_scenarios():

    st.caption(
        "Scenario-specific evaluation of "
        "five controlled ML models"
    )

    results = load_scenario_results().copy()

    results["scenario"] = (
        results["scenario"].astype(str)
    )

    results["Scenario"] = (
        "Scenario " + results["scenario"]
    )

    budget = st.selectbox(
        "Daily Alert Budget",
        [10, 50, 100],
        key="scenario_budget"
    )

    # Dataset distribution

    distribution = results[
        [
            "Scenario",
            "malicious_days",
            "total_insiders"
        ]
    ].drop_duplicates(
        subset=["Scenario"]
    )

    st.subheader("Attack Scenario Distribution")

    cols = st.columns(len(distribution))

    for col, (_, row) in zip(
        cols,
        distribution.iterrows()
    ):

        col.metric(
            row["Scenario"],
            f"{int(row['malicious_days'])} days"
        )

        col.caption(
            f"{int(row['total_insiders'])} insiders"
        )

    fig = px.bar(
        distribution,
        x="Scenario",
        y="malicious_days",
        color="Scenario",
        text_auto=True,
        title="Known Malicious Employee-Days"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Scenario recall

    st.subheader("Scenario-Wise Detection Recall")

    recall_column = f"recall_at_{budget}"

    fig = px.bar(
        results,
        x="Scenario",
        y=recall_column,
        color="model",
        barmode="group",
        labels={
            recall_column: "Recall",
            "model": "Model"
        }
    )

    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1.05]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Detailed results

    st.subheader("Detailed Scenario Results")

    display = results[
        [
            "model",
            "Scenario",
            "malicious_days",
            f"detected_days_at_{budget}",
            recall_column,
            f"insiders_detected_at_{budget}",
            "total_insiders"
        ]
    ].copy()

    display.columns = [
        "Model",
        "Scenario",
        "Malicious Days",
        "Detected Days",
        "Recall",
        "Insiders Detected",
        "Total Insiders"
    ]

    st.dataframe(
        display.style.format({
            "Recall": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Macro recall

    st.subheader("Macro-Average Scenario Recall")

    macro = (
        results.groupby("model")[recall_column]
        .mean()
        .reset_index(name="Macro Recall")
        .sort_values(
            "Macro Recall",
            ascending=False
        )
    )

    fig = px.bar(
        macro,
        x="model",
        y="Macro Recall",
        color="model",
        text_auto=".2%"
    )

    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1.05]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Macro recall gives equal weight to each "
        "scenario, regardless of its record count."
    )

    st.download_button(
        "Download Scenario Evaluation",
        results.to_csv(index=False),
        file_name="scenario_model_comparison.csv",
        mime="text/csv"
    )

    st.warning(
        "Scenario 2 dominates the test positives. "
        "Scenario 3 contains only six malicious "
        "employee-days, so its recall estimates "
        "are particularly sensitive to individual records."
    )


# ==========================================
# PAGE 3: MODEL RELIABILITY & EXPLAINABILITY
# ==========================================

def render_model_reliability():

    st.caption(
        "Generalization, explanation alignment "
        "and explanation stability evaluation"
    )

    generalization = load_generalization_results()
    alignment = load_alignment_results()
    stability = load_stability_results().copy()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Unseen Employees",
        "Explanation Alignment",
        "Explanation Stability",
        "Research Limitations"
    ])

    # --------------------------------------
    # TAB 1: GENERALIZATION
    # --------------------------------------

    with tab1:

        st.subheader(
            "Unseen-Employee Generalization"
        )

        st.info(
            "Five repeated employee holdouts. "
            "Each experiment evaluates 184 employees "
            "excluded from model fitting."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Holdout Experiments", 5)
        c2.metric("Employees per Holdout", 184)
        c3.metric("Employee Overlap", 0)

        fig = px.bar(
            generalization,
            x="model",
            y="mean_pr_auc",
            error_y="std_pr_auc",
            color="model",
            text_auto=".4f",
            title="Mean Average Precision Across Holdouts"
        )

        fig.update_yaxes(
            range=[0, 1.05]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        recall = generalization[
            [
                "model",
                "mean_recall_at_2",
                "mean_recall_at_10"
            ]
        ].melt(
            id_vars="model",
            var_name="Alert Budget",
            value_name="Recall"
        )

        recall["Alert Budget"] = (
            recall["Alert Budget"].map({
                "mean_recall_at_2": "2/day",
                "mean_recall_at_10": "10/day"
            })
        )

        fig = px.bar(
            recall,
            x="model",
            y="Recall",
            color="Alert Budget",
            barmode="group",
            text_auto=".2%",
            title="Generalization Recall"
        )

        fig.update_yaxes(
            tickformat=".0%",
            range=[0, 1.05]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            generalization.style.format({
                "mean_pr_auc": "{:.4f}",
                "std_pr_auc": "{:.4f}",
                "mean_precision_at_2": "{:.2%}",
                "mean_recall_at_2": "{:.2%}",
                "mean_precision_at_10": "{:.2%}",
                "mean_recall_at_10": "{:.2%}",
                "std_recall_at_10": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Error bars show standard deviation "
            "across holdouts, not confidence intervals."
        )

        st.warning(
            "A budget of 10 alerts/day is more generous "
            "for 184 employees than for the full "
            "918-employee test population. "
            "The 2/day budget provides an approximately "
            "population-adjusted comparison."
        )

        st.download_button(
            "Download Generalization Results",
            generalization.to_csv(index=False),
            file_name="generalization_summary.csv",
            mime="text/csv"
        )

    # --------------------------------------
    # TAB 2: ALIGNMENT
    # --------------------------------------

    with tab2:

        st.subheader(
            "Ground Truth Explanation Alignment"
        )

        st.caption(
            "Evaluation of explanations across "
            "401 known malicious employee-days "
            "using activity-to-feature mappings."
        )

        alignment_columns = [
            "mean_feature_alignment",
            "mean_activity_coverage",
            "mean_positive_attribution_alignment",
            "hit_at_5",
            "random_feature_baseline"
        ]

        alignment_long = alignment[
            ["model"] + alignment_columns
        ].melt(
            id_vars="model",
            var_name="Metric",
            value_name="Value"
        )

        alignment_long["Metric"] = (
            alignment_long["Metric"].map({
                "mean_feature_alignment":
                    "Feature Alignment",
                "mean_activity_coverage":
                    "Activity Coverage",
                "mean_positive_attribution_alignment":
                    "Positive Attribution Alignment",
                "hit_at_5":
                    "Hit@5",
                "random_feature_baseline":
                    "Random Feature Baseline"
            })
        )

        fig = px.bar(
            alignment_long,
            x="Metric",
            y="Value",
            color="model",
            barmode="group",
            text_auto=".1%"
        )

        fig.update_yaxes(
            tickformat=".0%",
            range=[0, 1.1]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            alignment.style.format({
                column: "{:.2%}"
                for column in alignment_columns
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            **Metric interpretation**

            - **Feature alignment:** Fraction of selected
              positive-contribution features associated
              with known malicious activity types.

            - **Activity coverage:** Fraction of known
              malicious activity types represented by
              selected features.

            - **Positive attribution alignment:** Share
              of positive contribution magnitude assigned
              to relevant features.

            - **Hit@5:** Whether at least one selected
              feature matches a known activity type.
            """
        )

        st.warning(
            "Alignment is based on a manually defined "
            "activity-to-feature mapping. It does not "
            "establish causal explanation correctness."
        )

        st.download_button(
            "Download Alignment Results",
            alignment.to_csv(index=False),
            file_name="overall_alignment.csv",
            mime="text/csv"
        )

    # --------------------------------------
    # TAB 3: STABILITY
    # --------------------------------------

    with tab3:

        st.subheader(
            "Explanation Stability Under Perturbation"
        )

        stability["Perturbation (%)"] = (
            stability["perturbation"] * 100
        )

        stability = stability.sort_values(
            ["model", "Perturbation (%)"]
        )

        metric_options = {
            "Top-5 Feature Stability":
                "mean_top5_stability",

            "Spearman Rank Correlation":
                "mean_spearman",

            "Contribution Direction Agreement":
                "mean_direction_agreement",

            "Absolute Model Score Change":
                "mean_score_change"
        }

        selected_metric = st.selectbox(
            "Stability Metric",
            list(metric_options.keys()),
            key="reliability_stability_metric"
        )

        metric_column = metric_options[
            selected_metric
        ]

        fig = px.line(
            stability,
            x="Perturbation (%)",
            y=metric_column,
            color="model",
            markers=True,
            labels={
                metric_column: selected_metric
            },
            title=selected_metric
        )

        fig.update_xaxes(
            tickvals=[5, 10, 20]
        )

        if metric_column == "mean_spearman":

            fig.update_yaxes(
                range=[-1, 1.05]
            )

        else:

            fig.update_yaxes(
                tickformat=".0%",
                range=[0, 1.05]
            )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            stability.style.format({
                "perturbation": "{:.0%}",
                "mean_top5_stability": "{:.2%}",
                "mean_spearman": "{:.4f}",
                "mean_direction_agreement": "{:.2%}",
                "mean_score_change": "{:.4f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Higher explanation stability, rank "
            "correlation and direction agreement are "
            "desirable. Lower absolute model score "
            "change is desirable."
        )

        st.warning(
            "Perturbations modify selected activity "
            "counts rather than replaying real logs. "
            "Stable explanations do not necessarily "
            "imply stable model predictions."
        )

        st.download_button(
            "Download Stability Results",
            stability.to_csv(index=False),
            file_name="stability_summary.csv",
            mime="text/csv"
        )

    # --------------------------------------
    # TAB 4: LIMITATIONS
    # --------------------------------------

    with tab4:

        st.subheader("Research Limitations")

        st.markdown(
            """
            ### 1. Synthetic Dataset

            CERT r4.2 contains simulated organizational
            activity and insider threat scenarios.
            Performance may differ in real enterprises.

            ### 2. Class Imbalance

            Known malicious employee-days represent
            approximately 0.331% of the test dataset.

            ### 3. Alert Burden

            High recall does not eliminate false alerts.
            At 10 alerts/day, EBM precision is approximately
            18.95%, requiring substantial analyst review.

            ### 4. Uncalibrated Model Scores

            Model outputs are not verified probabilities
            of malicious intent.

            ### 5. Ground Truth Assumptions

            Employee-days without known malicious labels
            are treated as normal during evaluation.

            ### 6. Generalization Protocol

            Employee holdouts exclude selected employees
            from model fitting. Hyperparameters were
            selected previously, and holdout stratification
            uses retrospective labels.

            ### 7. Explanation Limitations

            SHAP values and EBM contributions explain
            model behavior, not causal malicious intent.

            EBM interaction contributions are equally
            allocated among participating features
            for feature-level explanation comparisons.

            ### 8. Scenario Imbalance

            Scenario 2 dominates the positive test
            records. Scenario 3 has limited observations.

            ### 9. Behavioral Feature Limitations

            The current controlled comparison uses
            22 behavioral features without explicit
            user-relative or peer-group baseline features.

            ### 10. Evaluation Exposure

            The historical test period has been examined
            during development. An independent external
            dataset is needed for stronger validation.

            ### 11. Deployment Scope

            The dashboard is a historical research
            prototype, not a live production SIEM
            or automated employee disciplinary system.
            """
        )

        st.success(
            "The system supports explainable threat "
            "prioritization and retrospective evaluation "
            "within the documented experimental scope."
        )