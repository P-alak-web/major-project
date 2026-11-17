import streamlit as st

st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🛡️ Threat Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "User Behaviour Analytics",
        "Threat Detection",
        "Explainable AI",
        "Ground Truth Validation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Explainable Insider Threat Detection System")

st.title(f"🛡️ {page}")

import pandas as pd
import plotly.express as px

if page == "Overview":
    st.caption("Demo dashboard – Sample data")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric("👥 Total Employees", "1,000")

    with col2:
        with st.container(border=True):
            st.metric("📊 Total Activities", "50,000")

    with col3:
        with st.container(border=True):
            st.metric("⚠️ Suspicious Activities", "120")

    with col4:
        with st.container(border=True):
            st.metric("🚨 High-Risk Users", "15")

    st.subheader("Daily Activity Trend")

    data = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=7),
        "Activities": [180, 195, 205, 200, 220, 215, 230]
    })

    min_date = data["Date"].min().date()
    max_date = data["Date"].max().date()

    col_start, col_end = st.columns(2)

    with col_start:
        start_date = st.date_input(
            "Start Date",
            min_date,
            min_value=min_date,
            max_value=max_date
        )

    with col_end:
        end_date = st.date_input(
            "End Date",
            max_date,
            min_value=min_date,
            max_value=max_date
        )

    if start_date > end_date:
        st.error("Start Date cannot be after End Date")
        st.stop()

    data = data[
        (data["Date"].dt.date >= start_date) &
        (data["Date"].dt.date <= end_date)
    ]
    fig = px.line(
        data,
        x="Date",
        y="Activities",
        markers=True,
        line_shape="spline"
    )

    fig.update_traces(
        line=dict(color="#00D4FF", width=3),
        marker=dict(size=8, color="#00D4FF")
    )

    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    fig.update_xaxes(
        tickformat="%d %b",
        dtick=86400000
    )
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        label="📥 Download Activity Report (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="activity_report.csv",
        mime="text/csv"
    )
elif page == "User Behaviour Analytics":

    st.caption("Demo Dashboard - Sample Data")

    st.subheader("Employee Activity Analysis")

    user = st.selectbox(
        "Select Employee",
        ["User_001", "User_002", "User_003"]
    )

    activity_data = {
        "User_001": [45, 120, 8, 65],
        "User_002": [30, 85, 3, 40],
        "User_003": [65, 200, 15, 95]
    }

    data = pd.DataFrame({
        "Activity": [
            "Login Activity",
            "File Access",
            "USB Usage",
            "Website Visits"
        ],
        "Count": activity_data[user]
    })
    login, files, usb, websites = activity_data[user]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric("🔐 Login Activity", login)

    with col2:
        with st.container(border=True):
            st.metric("📁 File Access", files)

    with col3:
        with st.container(border=True):
            st.metric("💾 USB Usage", usb)

    with col4:
        with st.container(border=True):
            st.metric("🌐 Website Visits", websites)

    st.subheader("User Activity Overview")

    fig = px.bar(
        data,
        x="Activity",
        y="Count",
        color="Activity",
        title=f"Activity Analysis - {user}"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Activity Details")
    st.dataframe(data, use_container_width=True)

    st.download_button(
        label="📥 Download Employee Activity (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=f"{user}_activity_report.csv",
        mime="text/csv"
    )
elif page == "Threat Detection":

    st.caption("Demo Dashboard - Sample Data")

    st.subheader("Suspicious Employee Detection")

    threat_data = pd.DataFrame({
        "Employee": ["User_001", "User_002", "User_003", "User_004", "User_005"],
        "Anomaly Score": [0.92, 0.45, 0.81, 0.25, 0.73],
        "Risk Level": ["High", "Low", "High", "Low", "Medium"],
        "Status": ["Flagged", "Normal", "Flagged", "Normal", "Flagged"]
    })

    total_users = len(threat_data)
    high_risk = (threat_data["Risk Level"] == "High").sum()
    medium_risk = (threat_data["Risk Level"] == "Medium").sum()
    low_risk = (threat_data["Risk Level"] == "Low").sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric("👥 Total Employees", total_users)

    with col2:
        with st.container(border=True):
            st.metric("🔴 High Risk", high_risk)

    with col3:
        with st.container(border=True):
            st.metric("🟠 Medium Risk", medium_risk)

    with col4:
        with st.container(border=True):
            st.metric("🟢 Low Risk", low_risk)

    risk_filter = st.selectbox(
        "Filter by Risk Level",
        ["All", "High", "Medium", "Low"]
    )

    if risk_filter == "All":
        filtered_data = threat_data
    else:
        filtered_data = threat_data[
            threat_data["Risk Level"] == risk_filter
        ]

    st.dataframe(filtered_data, use_container_width=True)

    st.subheader("Employee Anomaly Scores")

    fig = px.bar(
        filtered_data,
        x="Employee",
        y="Anomaly Score",
        color="Risk Level",
        title="Anomaly Score by Employee",
        color_discrete_map={
            "High": "red",
            "Medium": "orange",
            "Low": "green"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="📥 Download Threat Report (CSV)",
        data=filtered_data.to_csv(index=False).encode("utf-8"),
        file_name="threat_detection_report.csv",
        mime="text/csv"
    )

elif page == "Explainable AI":

    st.caption("Demo Dashboard - Sample Data")
    st.warning("Illustrative explanations only - not actual SHAP results.")

    st.subheader("Explainable AI Analysis")

    user = st.selectbox(
        "Select Employee",
        ["User_001", "User_002", "User_003"]
    )

    explanation_data = {
        "User_001": [0.85, 0.65, 0.90, 0.40],
        "User_002": [0.30, 0.45, 0.20, 0.35],
        "User_003": [0.75, 0.80, 0.60, 0.70]
    }

    data = pd.DataFrame({
        "Feature": [
            "After Hours Login",
            "File Access",
            "USB Usage",
            "Website Activity"
        ],
        "Illustrative Impact": explanation_data[user]
    })

    after_hours, file_access, usb_usage, website = explanation_data[user]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric("🌙 After Hours Login", f"{after_hours:.2f}")

    with col2:
        with st.container(border=True):
            st.metric("📁 File Access Impact", f"{file_access:.2f}")

    with col3:
        with st.container(border=True):
            st.metric("💾 USB Usage Impact", f"{usb_usage:.2f}")

    with col4:
        with st.container(border=True):
            st.metric("🌐 Website Impact", f"{website:.2f}")
    st.subheader("Behaviour Feature Importance")

    fig = px.bar(
        data.sort_values("Illustrative Impact"),
        x="Illustrative Impact",
        y="Feature",
        orientation="h",
        text="Illustrative Impact",
        title=f"Feature Impact Analysis - {user}"
    )

    fig.update_traces(
        marker_color="#00D4FF",
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1.05]),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Details")
    st.dataframe(data, use_container_width=True)

    st.download_button(
        label="📥 Download AI Explanation (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=f"{user}_explanation_report.csv",
        mime="text/csv"
    )
    st.info(
        "These values are simulated for UI testing. "
        "Real model explanations will be connected later."
    )

elif page == "Ground Truth Validation":

    st.caption("Demo Dashboard - Sample Data")
    st.warning("These are simulated validation results, not actual model performance.")

    st.subheader("Ground Truth vs Model Predictions")

    validation_data = pd.DataFrame({
        "Employee": [
            "User_001", "User_002", "User_003",
            "User_004", "User_005", "User_006"
        ],
        "Actual": [1, 0, 1, 0, 0, 1],
        "Predicted": [1, 0, 0, 0, 1, 1]
    })

    validation_data["Result"] = validation_data.apply(
        lambda row: (
            "True Positive" if row["Actual"] == 1 and row["Predicted"] == 1
            else "True Negative" if row["Actual"] == 0 and row["Predicted"] == 0
            else "False Positive" if row["Actual"] == 0
            else "False Negative"
        ),
        axis=1
    )
    tp = ((validation_data["Actual"] == 1) &
          (validation_data["Predicted"] == 1)).sum()

    tn = ((validation_data["Actual"] == 0) &
          (validation_data["Predicted"] == 0)).sum()

    fp = ((validation_data["Actual"] == 0) &
          (validation_data["Predicted"] == 1)).sum()

    fn = ((validation_data["Actual"] == 1) &
          (validation_data["Predicted"] == 0)).sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric("✅ True Positive", tp)

    with col2:
        with st.container(border=True):
            st.metric("🟢 True Negative", tn)

    with col3:
        with st.container(border=True):
            st.metric("⚠️ False Positive", fp)

    with col4:
        with st.container(border=True):
            st.metric("❌ False Negative", fn)
    st.dataframe(validation_data, use_container_width=True)
    
    st.download_button(
        label="📥 Download Validation Report (CSV)",
        data=validation_data.to_csv(index=False).encode("utf-8"),
        file_name="ground_truth_validation_report.csv",
        mime="text/csv"
    )
    st.subheader("Confusion Matrix")

    matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Normal", "Actual Insider"],
        columns=["Predicted Normal", "Predicted Insider"]
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Demo Confusion Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Actual CERT ground truth and model predictions "
        "will be connected during backend integration."
    )

else:
    st.info("This page is under development.")