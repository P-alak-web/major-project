import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS

from evaluation.prepare_controlled_data import prepare_data

from evaluation.controlled_model_comparison import predict_scores


OUTPUT_DIR = ROOT / "outputs" / "eval" / "controlled"
MODEL_DIR = OUTPUT_DIR / "models"

MODELS = {
    "EBM": "ebm",
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
    "Pooled Isolation Forest": "pooled_isolation_forest",
    "Per-user Isolation Forest": "per_user_isolation_forest"
}


def main():

    print("=" * 60)
    print("SAVED MODEL VERIFICATION")
    print("=" * 60)

    _, _, test = prepare_data()

    saved_features = joblib.load(
        MODEL_DIR / "feature_columns.pkl"
    )

    assert saved_features == FEATURE_COLUMNS

    print("\nFeature order verified.")
    print("Number of features:", len(saved_features))

    # Select reproducible test samples.
    sample = test.sample(
        n=1000,
        random_state=42
    ).copy()

    results = []

    for name, filename in MODELS.items():

        print("\n" + "-" * 60)
        print("Verifying:", name)

        model_path = MODEL_DIR / f"{filename}.pkl"

        model = joblib.load(model_path)

        scores = predict_scores(
            name,
            model,
            sample
        )

        actual = sample[["user", "day"]].copy()
        actual["reloaded_score"] = scores

        saved = pd.read_csv(
            OUTPUT_DIR / f"predictions_{filename}.csv"
        )

        saved["day"] = pd.to_datetime(saved["day"])

        comparison = actual.merge(
            saved[["user", "day", "score"]],
            on=["user", "day"],
            how="left",
            validate="one_to_one"
        )

        assert len(comparison) == len(sample)
        assert comparison["score"].notna().all()

        difference = np.abs(
            comparison["reloaded_score"]
            - comparison["score"]
        )

        max_difference = float(difference.max())

        passed = np.allclose(
            comparison["reloaded_score"],
            comparison["score"],
            rtol=1e-6,
            atol=1e-8
        )

        print("Model loaded successfully.")
        print("Samples verified:", len(comparison))
        print("Maximum score difference:", max_difference)
        print("Status:", "PASS" if passed else "FAIL")

        results.append({
            "model": name,
            "samples_verified": len(comparison),
            "max_score_difference": max_difference,
            "status": "PASS" if passed else "FAIL"
        })

        del model

    summary = pd.DataFrame(results)

    summary.to_csv(
        OUTPUT_DIR / "model_verification.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 60)

    print(summary.to_string(index=False))

    if (summary["status"] == "PASS").all():
        print("\nAll saved models verified successfully!")
    else:
        print("\nSome models failed verification.")


if __name__ == "__main__":
    main()