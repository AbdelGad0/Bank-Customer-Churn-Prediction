import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

BASE = r"E:\MyProjects\Bank Customer Churn Prediction"
SPLITS = os.path.join(BASE, "data_splits_fe")
RANDOM_STATE = 42

X_train = pd.read_csv(os.path.join(SPLITS, "X_train.csv"))
X_val = pd.read_csv(os.path.join(SPLITS, "X_val.csv"))
X_test = pd.read_csv(os.path.join(SPLITS, "X_test.csv"))
y_train = pd.read_csv(os.path.join(SPLITS, "y_train.csv")).squeeze("columns")
y_val = pd.read_csv(os.path.join(SPLITS, "y_val.csv")).squeeze("columns")
y_test = pd.read_csv(os.path.join(SPLITS, "y_test.csv")).squeeze("columns")

print("Test set:", dict(y_test.value_counts()))
print()

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.9, colsample_bytree=1.0, gamma=0.2, min_child_weight=3,
    eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
)


def find_best_f2_threshold(y_true, proba):
    best_t, best_f2 = 0.5, -1
    for t in np.arange(0.02, 0.98, 0.01):
        yp = (proba >= t).astype(int)
        p = precision_score(y_true, yp)
        r = recall_score(y_true, yp)
        if (p + r) > 0:
            f2 = (5 * p * r) / (4 * p + r)
            if f2 > best_f2:
                best_f2, best_t = f2, t
    return best_t, best_f2


def run(name, resampler, model):
    pipe = Pipeline([("res", resampler), ("clf", model)])
    pipe.fit(X_train, y_train)
    proba_val = pipe.predict_proba(X_val)[:, 1]
    t, f2 = find_best_f2_threshold(y_val, proba_val)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    yp = (proba_test >= t).astype(int)
    row = {
        "resampler": resampler.__class__.__name__,
        "threshold": round(t, 2),
        "accuracy": accuracy_score(y_test, yp),
        "precision": precision_score(y_test, yp),
        "recall": recall_score(y_test, yp),
        "f1": f1_score(y_test, yp),
        "roc_auc": roc_auc_score(y_test, proba_test),
        "pr_auc": average_precision_score(y_test, proba_test),
    }
    print(
        f"{name:34s} thr={row['threshold']:4.2f} Rec={row['recall']:.4f} "
        f"ROC={row['roc_auc']:.4f} PR={row['pr_auc']:.4f} F2(val)={f2:.4f}"
    )
    return row


results = {}
common_smote = SMOTE(random_state=RANDOM_STATE)

results["XGB+SMOTE (champion baseline)"] = run(
    "XGB+SMOTE (champion baseline)",
    common_smote, xgb.XGBClassifier(**XGB_PARAMS))

results["XGB+SMOTE (with FE)"] = run(
    "XGB+SMOTE (with FE)", common_smote, xgb.XGBClassifier(**XGB_PARAMS))

results["XGB+SMOTETomek (with FE)"] = run(
    "XGB+SMOTETomek (with FE)",
    SMOTETomek(random_state=RANDOM_STATE), xgb.XGBClassifier(**XGB_PARAMS))

results["XGB+SMOTEENN (with FE)"] = run(
    "XGB+SMOTEENN (with FE)",
    SMOTEENN(random_state=RANDOM_STATE), xgb.XGBClassifier(**XGB_PARAMS))

results["LightGBM+SMOTETomek (with FE)"] = run(
    "LightGBM+SMOTETomek (with FE)",
    SMOTETomek(random_state=RANDOM_STATE),
    lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                       verbose=-1, random_state=RANDOM_STATE,
                       colsample_bytree=0.9, subsample=0.9))

results["CatBoost+SMOTETomek (with FE)"] = run(
    "CatBoost+SMOTETomek (with FE)",
    SMOTETomek(random_state=RANDOM_STATE),
    CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4,
                       verbose=0, random_seed=RANDOM_STATE, eval_metric="Logloss"))

results["CatBoost+SMOTEENN (with FE)"] = run(
    "CatBoost+SMOTEENN (with FE)",
    SMOTEENN(random_state=RANDOM_STATE),
    CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4,
                       verbose=0, random_seed=RANDOM_STATE, eval_metric="Logloss"))

print("=" * 70)
print("COMPARISON ON TEST SET (best-F2 threshold per model)")
print("=" * 70)
summary = pd.DataFrame(results).T.round(4)
summary = summary.sort_values("pr_auc", ascending=False)
print(summary.to_string())
summary.to_csv(os.path.join(BASE, "improved_results.csv"))
print("\nSaved to improved_results.csv")