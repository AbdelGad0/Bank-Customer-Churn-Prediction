import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow import keras

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

BASE = r"E:\MyProjects\Bank Customer Churn Prediction"
SPLITS = os.path.join(BASE, "data_splits")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
CHOSEN_XGB_THRESHOLD = 0.24

X_train = pd.read_csv(os.path.join(SPLITS, "X_train.csv"))
X_val = pd.read_csv(os.path.join(SPLITS, "X_val.csv"))
X_test = pd.read_csv(os.path.join(SPLITS, "X_test.csv"))
y_train = pd.read_csv(os.path.join(SPLITS, "y_train.csv")).squeeze("columns")
y_val = pd.read_csv(os.path.join(SPLITS, "y_val.csv")).squeeze("columns")
y_test = pd.read_csv(os.path.join(SPLITS, "y_test.csv")).squeeze("columns")

print("Test set:", dict(y_test.value_counts()))


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


def best_f2_threshold_for(model_factory):
    model = model_factory()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    t, f2 = find_best_f2_threshold(y_val, proba)
    return model, t, f2


def eval_at(y_true, proba, threshold):
    yp = (proba >= threshold).astype(int)
    return {
        "threshold": round(threshold, 2),
        "accuracy": accuracy_score(y_true, yp),
        "precision": precision_score(y_true, yp),
        "recall": recall_score(y_true, yp),
        "f1": f1_score(y_true, yp),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


results = {}

print("=" * 70)
print("FITTING + SELECTING THRESHOLD ON VALIDATION (best F2)")
print("=" * 70)

model, t, f2 = best_f2_threshold_for(
    lambda: Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("lr", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
)
print(f"Logistic Regression : validation threshold={t:.2f} (F2={f2:.4f})")
results["Logistic Regression"] = eval_at(y_test, model.predict_proba(X_test)[:, 1], t)

model, t, f2 = best_f2_threshold_for(
    lambda: Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
)
print(f"Random Forest       : validation threshold={t:.2f} (F2={f2:.4f})")
results["Random Forest"] = eval_at(y_test, model.predict_proba(X_test)[:, 1], t)

model, t, f2 = best_f2_threshold_for(
    lambda: Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=1.0, gamma=0.2, min_child_weight=3,
            eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
        )),
    ])
)
print(f"XGBoost (tuned)     : best-F2 threshold={t:.2f} (F2={f2:.4f})")
results["XGBoost (tuned, thr=0.24)"] = eval_at(y_test, model.predict_proba(X_test)[:, 1], CHOSEN_XGB_THRESHOLD)
results["XGBoost (tuned, best-F2)"] = eval_at(y_test, model.predict_proba(X_test)[:, 1], t)

model, t, f2 = best_f2_threshold_for(
    lambda: Pipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("lgb", lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            verbose=-1, random_state=RANDOM_STATE,
        )),
    ])
)
print(f"LightGBM            : validation threshold={t:.2f} (F2={f2:.4f})")
results["LightGBM"] = eval_at(y_test, model.predict_proba(X_test)[:, 1], t)

print("=" * 70)
print("NEURAL NETWORK (SMOTE on train, random hidden sizes)")
X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
rng = np.random.default_rng(RANDOM_STATE)
n1 = int(rng.integers(8, 64))
n2 = int(rng.integers(4, 32))
print(f"Hidden layers: Dense({n1}) -> Dense({n2})")
nn = keras.Sequential([
    keras.layers.Input(shape=(X_res.shape[1],)),
    keras.layers.Dense(n1, activation="relu"),
    keras.layers.Dense(n2, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
nn.compile(optimizer=keras.optimizers.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
nn.fit(X_res, y_res, validation_data=(X_val, y_val), epochs=150, batch_size=128,
       callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)],
       verbose=0)
proba_val = nn.predict(X_val, verbose=0).ravel()
t, f2 = find_best_f2_threshold(y_val, proba_val)
print(f"Neural Network       : validation threshold={t:.2f} (F2={f2:.4f})")
results["Neural Network"] = eval_at(y_test, nn.predict(X_test, verbose=0).ravel(), t)

print("=" * 70)
print("FINAL RESULTS ON TEST SET")
print("=" * 70)
summary = pd.DataFrame(results).T.round(4)
summary = summary.sort_values("pr_auc", ascending=False)
print(summary.to_string())

summary.to_csv(os.path.join(BASE, "final_results.csv"))
print("\nSaved to final_results.csv")