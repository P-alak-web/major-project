"""
Global-time-split evaluation for CERT r4.2 insider threat detection.

Run from the repo root:
  python evaluation/run_global_eval.py \
      --features processed/user_daily_features.csv \
      --truth    outputs/ground_truth/ground_truth_user_day.csv \
      --split-date 2010-11-01

What it does
  1. Uses ONE calendar cut-off for all users (train < split-date <= test).
  2. Scores the test period with two models:
       per_user : one Isolation Forest per employee on raw counts (baseline)
       pooled   : one Isolation Forest for everyone on features expressed
                  as deviations from each user's own training baseline
  3. Writes standard prediction files (user, day, score).
  4. Reports positives in train vs test, PR-AUC, and recall / precision /
     insiders found at fixed daily alert budgets (top-k per day).

Assumption to state in your paper: labeled malicious user-days are positives
and every other user-day is treated as a negative (standard CERT practice).
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

ACTIVITY = [
    "logon_count", "after_hours_logons",
    "usb_connect_count", "after_hours_usb",
    "http_requests", "after_hours_http",
    "email_count", "after_hours_email",
    "file_operations", "after_hours_file",
]


def load(features_path, truth_path):
    feats = pd.read_csv(features_path)
    feats["day"] = pd.to_datetime(feats["day"])
    missing = [c for c in ACTIVITY if c not in feats.columns]
    if missing:
        raise SystemExit(f"Feature file is missing columns: {missing}")
    feats[ACTIVITY] = feats[ACTIVITY].fillna(0)

    truth = pd.read_csv(truth_path)
    truth["day"] = pd.to_datetime(truth["date"])
    truth["user"] = truth["user"].astype(str).str.strip()
    truth = truth[["user", "day", "scenario"]].drop_duplicates(["user", "day"])
    truth["label"] = 1
    return feats, truth


def per_user_scores(feats, split, min_train):
    parts = []
    for user, g in feats.groupby("user"):
        train, test = g[g["day"] < split], g[g["day"] >= split]
        if len(train) < min_train or test.empty:
            continue
        model = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(train[ACTIVITY])
        out = test[["user", "day"]].copy()
        out["score"] = -model.decision_function(test[ACTIVITY])
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def pooled_scores(feats, split, min_train):
    train_all = feats[feats["day"] < split]
    counts = train_all.groupby("user").size()
    keep = counts[counts >= min_train].index
    train_all = train_all[train_all["user"].isin(keep)]

    grouped = train_all.groupby("user")[ACTIVITY]
    median = grouped.median()
    iqr = (grouped.quantile(0.75) - grouped.quantile(0.25)).replace(0, 1)

    def relative(d):
        d = d[d["user"].isin(keep)]
        x = (d[ACTIVITY].to_numpy() - median.loc[d["user"]].to_numpy()) \
            / iqr.loc[d["user"]].to_numpy()
        x = pd.DataFrame(x, columns=ACTIVITY, index=d.index)
        x["is_weekend"] = (d["day"].dt.dayofweek >= 5).astype(int)
        return d, x

    train_d, x_train = relative(train_all)
    test_d, x_test = relative(feats[feats["day"] >= split])

    model = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(x_train)
    out = test_d[["user", "day"]].copy()
    out["score"] = -model.decision_function(x_test)
    return out


def evaluate(name, scores, truth, budgets):
    d = scores.merge(truth, on=["user", "day"], how="left")
    d["label"] = d["label"].fillna(0).astype(int)
    pos = int(d["label"].sum())
    insiders = d.loc[d["label"] == 1, "user"].nunique()
    d["rank"] = d.groupby("day")["score"].rank(ascending=False, method="first")

    rows = {"model": name, "test_rows": len(d), "test_positives": pos,
            "test_insiders": insiders}
    rows["pr_auc"] = round(average_precision_score(d["label"], d["score"]), 4) \
        if pos else float("nan")
    rows["random_pr_auc"] = round(pos / len(d), 5)  # baseline = positive rate

    for k in budgets:
        alerts = d[d["rank"] <= k]
        hits = alerts[alerts["label"] == 1]
        rows[f"recall@{k}/day"] = round(len(hits) / pos, 4) if pos else float("nan")
        rows[f"precision@{k}/day"] = round(len(hits) / len(alerts), 5)
        rows[f"insiders_found@{k}/day"] = f"{hits['user'].nunique()}/{insiders}"

    k0 = budgets[0]
    by_scn = (d[d["label"] == 1]
              .assign(hit=lambda x: (x["rank"] <= k0).astype(int))
              .groupby("scenario")["hit"].agg(["sum", "count"]))
    by_scn["recall"] = (by_scn["sum"] / by_scn["count"]).round(3)
    return rows, by_scn, d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--split-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--min-train-days", type=int, default=30)
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 50, 100],
                    help="alerts per day to evaluate (first one is used for scenarios)")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval"))
    a = ap.parse_args()

    split = pd.Timestamp(a.split_date)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    feats, truth = load(a.features, a.truth)

    print(f"Features: {len(feats):,} user-days, {feats['user'].nunique()} users, "
          f"{feats['day'].min():%Y-%m-%d} to {feats['day'].max():%Y-%m-%d}")
    print(f"Split date: {split:%Y-%m-%d}  (train before, test on/after)\n")

    # Positives in train vs test, and how many labels match a feature row
    gt_rows = len(truth)
    in_feats = truth.merge(feats[["user", "day"]], on=["user", "day"])
    n_train = int((in_feats["day"] < split).sum())
    n_test = int((in_feats["day"] >= split).sum())
    print(f"Ground-truth positive user-days: {gt_rows}")
    print(f"  matched to a feature row : {len(in_feats)}")
    print(f"  in TRAIN period          : {n_train}   <- contaminate baselines, cannot be scored")
    print(f"  in TEST period           : {n_test}\n")
    if n_test == 0:
        raise SystemExit("No positives in the test period. Move --split-date earlier.")

    models = {
        "per_user": per_user_scores(feats, split, a.min_train_days),
        "pooled": pooled_scores(feats, split, a.min_train_days),
    }

    summary = []
    for name, scores in models.items():
        scores.to_csv(a.out_dir / f"predictions_{name}.csv", index=False)
        row, by_scn, _ = evaluate(name, scores, truth, a.budgets)
        summary.append(row)
        print(f"=== {name} ===")
        print(f"Scenario recall at top-{a.budgets[0]} alerts/day:")
        print(by_scn.rename(columns={"sum": "hits", "count": "positives"}).to_string())
        print()

    table = pd.DataFrame(summary).set_index("model").T
    print("=== SUMMARY ===")
    print(table.to_string())
    table.to_csv(a.out_dir / "metrics_summary.csv")
    print(f"\nSaved predictions and metrics_summary.csv to {a.out_dir}")


if __name__ == "__main__":
    main()