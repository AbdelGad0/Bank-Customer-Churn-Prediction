import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import xgboost as xgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

BASE = r"E:\MyProjects\Bank Customer Churn Prediction"
SPLITS = os.path.join(BASE, "data_splits")
RANDOM_STATE = 42

X_train = pd.read_csv(os.path.join(SPLITS, "X_train.csv"))
X_val = pd.read_csv(os.path.join(SPLITS, "X_val.csv"))
y_train = pd.read_csv(os.path.join(SPLITS, "y_train.csv")).squeeze("columns")
y_val = pd.read_csv(os.path.join(SPLITS, "y_val.csv")).squeeze("columns")

print("Tuning data (original train):", dict(y_train.value_counts()))
print("Evaluation on validation:", dict(y_val.value_counts()))

pipe = Pipeline(
    [
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("xgb", xgb.XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
        )),
    ]
)

param_grid = {
    "xgb__n_estimators": [100, 200, 300, 400],
    "xgb__max_depth": [3, 4, 5, 6],
    "xgb__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
    "xgb__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "xgb__gamma": [0, 0.1, 0.2],
    "xgb__min_child_weight": [1, 3, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    n_iter=40,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)
search.fit(X_train, y_train)

print("=" * 70)
print("BEST PARAMETERS")
print("=" * 70)
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print("Best CV ROC-AUC: {:.4f}".format(search.best_score_))

best = search.best_estimator_

def report(name, model):
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    print(
        f"{name:22s} Acc={accuracy_score(y_val, y_pred):.4f}  "
        f"Prec={precision_score(y_val, y_pred):.4f}  "
        f"Rec={recall_score(y_val, y_pred):.4f}  "
        f"ROC-AUC={roc_auc_score(y_val, y_prob):.4f}  "
        f"PR-AUC={average_precision_score(y_val, y_prob):.4f}"
    )

print("=" * 70)
print("VALIDATION COMPARISON")
print("=" * 70)
# re-train baseline to compare head-to-head (same convention as previous run)
baseline = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4,
    eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
)
X_sm = pd.read_csv(os.path.join(SPLITS, "X_train_smote.csv"))
y_sm = pd.read_csv(os.path.join(SPLITS, "y_train_smote.csv")).squeeze("columns")
baseline.fit(X_sm, y_sm)
report("Baseline XGB+SMOTE", baseline)
report("Tuned XGB (pipe+SMOTE)", best)