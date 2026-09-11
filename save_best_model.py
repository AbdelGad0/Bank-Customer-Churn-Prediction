import os
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import xgboost as xgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

BASE = r"E:\MyProjects\Bank Customer Churn Prediction"
RAW_DATA = os.path.join(BASE, "Churn_Modelling_cleaned.csv")
SPLITS = os.path.join(BASE, "data_splits")
RANDOM_STATE = 42
NUMERIC_ORDER = ["Age", "Balance", "NumOfProducts", "IsActiveMember"]
CAT_COLS = ["Geography", "Gender"]
TARGET = "Exited"

# same split procedure as preprocessing.py -> same row assignment as saved splits
raw = pd.read_csv(RAW_DATA)
X_raw = raw.drop(columns=[TARGET])
y_raw = raw[TARGET]
X_raw_train, _, y_raw_train, _ = train_test_split(
    X_raw, y_raw, test_size=0.30, stratify=y_raw, random_state=RANDOM_STATE
)

encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
encoder.fit(X_raw_train[CAT_COLS])
encoded_cols = encoder.get_feature_names_out(CAT_COLS).tolist()
feature_columns = NUMERIC_ORDER + encoded_cols
print("Encoder fitted on raw train. Encoded columns:", encoded_cols)
print("Raw train size:", len(X_raw_train))

X_train = pd.read_csv(os.path.join(SPLITS, "X_train.csv"))
X_val = pd.read_csv(os.path.join(SPLITS, "X_val.csv"))
y_train = pd.read_csv(os.path.join(SPLITS, "y_train.csv")).squeeze("columns")
y_val = pd.read_csv(os.path.join(SPLITS, "y_val.csv")).squeeze("columns")

assert list(X_train.columns) == feature_columns, "Column mismatch!"

pipe = Pipeline([
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("xgb", xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=1.0, gamma=0.2, min_child_weight=3,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )),
])
pipe.fit(X_train, y_train)

proba_val = pipe.predict_proba(X_val)[:, 1]
best_t, best_f2 = 0.5, -1
for t in np.arange(0.02, 0.98, 0.01):
    yp = (proba_val >= t).astype(int)
    p = precision_score(y_val, yp)
    r = recall_score(y_val, yp)
    if (p + r) > 0:
        f2 = (5 * p * r) / (4 * p + r)
        if f2 > best_f2:
            best_f2, best_t = f2, t
CHOSEN_THRESHOLD = float(best_t)

print(f"Best-F2 threshold on validation: {best_t:.2f} (F2={best_f2:.4f})")
print(f"Validation at thr={best_t:.2f}: "
      f"Precision={precision_score(y_val, (proba_val >= best_t).astype(int)):.4f} "
      f"Recall={recall_score(y_val, (proba_val >= best_t).astype(int)):.4f}")

joblib.dump(pipe, os.path.join(BASE, "churn_xgb_smote.joblib"))
joblib.dump(encoder, os.path.join(BASE, "onehot_encoder.joblib"))
with open(os.path.join(BASE, "model_config.json"), "w") as f:
    json.dump({
        "threshold": CHOSEN_THRESHOLD,
        "feature_columns": feature_columns,
        "numeric_order": NUMERIC_ORDER,
        "cat_cols": CAT_COLS,
    }, f, indent=2)

print("\nSaved artifacts:")
print("  churn_xgb_smote.joblib (SMOTE pipeline + XGBoost)")
print("  onehot_encoder.joblib")
print("  model_config.json")