# Explainable Insider Threat Detection using User Behavior Analytics

## Project Overview

This project aims to develop an explainable insider threat detection system using User Behavior Analytics (UBA).

The system analyses employee activity patterns, identifies unusual behaviour using machine learning, assigns risk scores to suspicious activity and provides explanations to help analysts understand why an activity was flagged.

**Important:** An anomaly is not automatically an insider threat. The system identifies unusual behaviour for further investigation.

## Project Objectives

- Analyse employee activity patterns.
- Extract behavioural features from activity logs.
- Detect anomalous behaviour using machine learning.
- Generate risk scores for unusual activities.
- Explain model predictions using explainable AI techniques.
- Display results through an interactive dashboard.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Explainable AI (planned)

## Project Structure

```text
Insider_Threat_Shared/
├── preprocessing/
├── features/
├── models/
├── backend/
├── dashboard/
├── tests/
├── reports/
├── README.md
├── requirements.txt
└── .gitignore
```

## Current Progress

- CERT dataset preprocessing script prepared.
- Daily user-level behavioural features generated.
- Feature validation performed.
- Initial Isolation Forest anomaly detection implemented.
- Dashboard development and explainability integration are pending.

## Dataset

The project uses the CERT Insider Threat Test Dataset.

Dataset files and ground-truth answer files are maintained separately and are not included in this repository.

## Team Contributions

Team responsibilities will be documented as implementation progresses.

## Disclaimer

This project is an academic research prototype. Anomaly scores indicate deviations from learned behaviour and should not be interpreted as proof of malicious activity.