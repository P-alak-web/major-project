import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import average_precision_score

# ---------------------------------------
# 1. Project paths
# ---------------------------------------

ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = (
    ROOT / "processed" / "user_daily_features_advanced_v2.csv"
)

GROUND_TRUTH_FILE = (
    ROOT / "outputs" / "ground_truth" / "ground_truth_user_day.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DATE = pd.Timestamp("2010-11-01")


# ---------------------------------------
# 2. Load data
# ---------------------------------------

def load_data():

    print("Loading enhanced behavioral features...")

    df = pd.read_csv(FEATURES_FILE)

    df["day"] = pd.to_datetime(df["day"])

    print("Loading ground truth...")

    truth = pd.read_csv(GROUND_TRUTH_FILE)

    truth["day"] = pd.to_datetime(truth["date"])

    truth = (
        truth.groupby(["user", "day"], as_index=False)["label"]
        .max()
    )

    df = df.merge(
        truth,
        on=["user", "day"],
        how="left",
        validate="one_to_one"
    )

    df["label"] = df["label"].fillna(0).astype(int)

    return df


# ---------------------------------------
# 3. Train-test split
# ---------------------------------------

def split_data(df):

    train = df[df["day"] < SPLIT_DATE].copy()
    test = df[df["day"] >= SPLIT_DATE].copy()

    feature_cols = [
        col for col in df.columns
        if col not in ["user", "day", "label"]
    ]

    X_train = train[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)

    y_train = train["label"]
    y_test = test["label"]

    print("\nTraining rows:", len(train))
    print("Testing rows:", len(test))

    print("Training positives:", y_train.sum())
    print("Testing positives:", y_test.sum())

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


# ---------------------------------------
# 4. Train EBM
# ---------------------------------------

def train_ebm(X_train, y_train):

    print("\nTraining Explainable Boosting Machine...")

    model = ExplainableBoostingClassifier(
        interactions=0,
        max_bins=128,
        learning_rate=0.05,
        max_rounds=300,
        outer_bags=4,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("EBM training completed!")

    return model


# ---------------------------------------
# 5. Evaluate EBM
# ---------------------------------------

def evaluate(model, X_test, y_test, test):

    print("\nEvaluating EBM...")

    scores = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(
        y_test,
        scores
    )

    print("\n==============================")
    print("EBM EVALUATION RESULTS")
    print("==============================")

    print("PR-AUC:", round(pr_auc, 5))

    predictions = test[["user", "day"]].copy()

    predictions["score"] = scores

    predictions["day"] = (
        predictions["day"].dt.strftime("%Y-%m-%d")
    )

    predictions.to_csv(
        OUTPUT_DIR / "predictions_ebm.csv",
        index=False
    )

    print("\nPredictions saved successfully.")

    return pr_auc


# ---------------------------------------
# 6. Feature importance
# ---------------------------------------

def save_feature_importance(model, feature_cols):

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.term_importances()
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "ebm_feature_importance.csv",
        index=False
    )

    print("\nTop 10 Important Features:")
    print(importance.head(10).to_string(index=False))


# ---------------------------------------
# 7. Save model
# ---------------------------------------

def save_model(model):

    path = MODEL_DIR / "ebm_model.pkl"

    joblib.dump(model, path)

    print("\nSaved EBM model to:")
    print(path)


# ---------------------------------------
# 8. Main
# ---------------------------------------

def main():

    print("====================================")
    print("CERT r4.2 INSIDER THREAT DETECTION")
    print("EXPLAINABLE BOOSTING MACHINE")
    print("====================================")

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

    model = train_ebm(X_train, y_train)

    evaluate(
        model,
        X_test,
        y_test,
        test
    )

    save_feature_importance(
        model,
        feature_cols
    )

    save_model(model)

    print("\nEBM pipeline completed successfully!")


if __name__ == "__main__":
    main()