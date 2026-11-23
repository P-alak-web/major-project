import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = (
    ROOT / "processed" / "user_daily_features_advanced_v2.csv"
)

GROUND_TRUTH_FILE = (
    ROOT / "outputs" / "ground_truth" / "ground_truth_user_day.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = OUTPUT_DIR / "predictions_xgboost.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "xgboost_feature_importance.csv"

SPLIT_DATE = pd.Timestamp("2010-11-01")


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

def load_data():

    print("Loading enhanced behavioral features...")

    df = pd.read_csv(FEATURES_FILE)

    print("Dataset shape:", df.shape)

    df["day"] = pd.to_datetime(df["day"])

    print("Loading ground truth...")

    truth = pd.read_csv(GROUND_TRUTH_FILE)

    truth["date"] = pd.to_datetime(truth["date"])

    truth = truth.rename(columns={"date": "day"})

    truth = (
        truth.groupby(["user", "day"], as_index=False)["label"]
        .max()
    )

    # Merge known malicious employee-days
    df = df.merge(
        truth,
        on=["user", "day"],
        how="left",
        validate="one_to_one"
    )

    # Benchmark assumption:
    # Employee-days without a known positive label are negative.
    df["label"] = df["label"].fillna(0).astype(int)

    return df


# --------------------------------------------------
# 3. Train-test split
# --------------------------------------------------

def split_data(df):

    print("\nApplying chronological train-test split...")

    train = df[df["day"] < SPLIT_DATE].copy()
    test = df[df["day"] >= SPLIT_DATE].copy()

    feature_cols = [
        col for col in df.columns
        if col not in ["user", "day", "label"]
    ]

    X_train = train[feature_cols].fillna(0)
    y_train = train["label"]

    X_test = test[feature_cols].fillna(0)
    y_test = test["label"]

    print("Training rows:", len(train))
    print("Testing rows:", len(test))

    print("Training positives:", int(y_train.sum()))
    print("Testing positives:", int(y_test.sum()))

    print("Number of features:", len(feature_cols))

    return (
        train,
        test,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols
    )


# --------------------------------------------------
# 4. Train XGBoost
# --------------------------------------------------

def train_model(X_train, y_train):

    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())

    if positive == 0:
        raise ValueError(
            "No positive training examples were found."
        )

    scale_pos_weight = negative / positive

    print("\nClass imbalance information:")
    print("Normal employee-days:", negative)
    print("Malicious employee-days:", positive)

    print(
        "Scale positive weight:",
        round(scale_pos_weight, 2)
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining XGBoost model...")

    model.fit(X_train, y_train)

    print("XGBoost training completed!")

    return model


# --------------------------------------------------
# 5. Evaluate model
# --------------------------------------------------

def evaluate_model(model, X_test, y_test, test):

    print("\nEvaluating XGBoost...")

    scores = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, scores)

    print("\n================================")
    print("XGBOOST EVALUATION RESULTS")
    print("================================")

    print("PR-AUC:", round(pr_auc, 5))

    print(
        "Random baseline PR-AUC:",
        round(float(y_test.mean()), 5)
    )

    predictions = test[["user", "day"]].copy()

    predictions["score"] = scores

    predictions["day"] = (
        predictions["day"].dt.strftime("%Y-%m-%d")
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    print("\nPredictions saved to:")
    print(PREDICTIONS_FILE)

    return pr_auc


# --------------------------------------------------
# 6. Feature importance
# --------------------------------------------------

def feature_importance(model, feature_cols):

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        IMPORTANCE_FILE,
        index=False
    )

    print("\nTop 10 Important Features:")
    print(importance.head(10).to_string(index=False))

    print("\nFeature importance saved to:")
    print(IMPORTANCE_FILE)


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

def main():

    print("================================")
    print("CERT r4.2 INSIDER THREAT DETECTION")
    print("XGBOOST MODEL TRAINING")
    print("================================")

    df = load_data()

    (
        train,
        test,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_cols
    ) = split_data(df)

    model = train_model(X_train, y_train)

    # Save trained XGBoost model and feature order
    MODEL_DIR = ROOT / "models" / "saved"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "xgboost_model.pkl"
    features_path = MODEL_DIR / "xgboost_features.pkl"

    joblib.dump(model, model_path)
    joblib.dump(feature_cols, features_path)

    print("\nSaved XGBoost model:", model_path)
    print("Saved feature order:", features_path)

    evaluate_model(
        model,
        X_test,
        y_test,
        test
    )

    feature_importance(
        model,
        feature_cols
    )

    print("\nXGBoost pipeline completed successfully!")


if __name__ == "__main__":
    main()