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
    roc_auc_score,
    average_precision_score,
)

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow import keras

BASE = r"E:\MyProjects\Bank Customer Churn Prediction"
SPLITS = os.path.join(BASE, "data_splits")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

X_train = pd.read_csv(os.path.join(SPLITS, "X_train_smote.csv"))
X_val = pd.read_csv(os.path.join(SPLITS, "X_val.csv"))
y_train = pd.read_csv(os.path.join(SPLITS, "y_train_smote.csv")).squeeze("columns")
y_val = pd.read_csv(os.path.join(SPLITS, "y_val.csv")).squeeze("columns")

print("Training data (after SMOTE):", dict(y_train.value_counts()))
print("Validation data (untouched):", dict(y_val.value_counts()))
print()

results = {}

def evaluate(name, model, predict_fn, proba_fn):
    y_pred = predict_fn(X_val)
    y_prob = proba_fn(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_prob)
    pr = average_precision_score(y_val, y_prob)
    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
        "pr_auc": pr,
    }
    print(f"{name:22s} Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")
    return model

# 1) Logistic Regression
print("=" * 70)
print("LOGISTIC REGRESSION (SMOTE)")
model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
evaluate("Logistic Regression", model, lambda X: model.predict(X),
         lambda X: model.predict_proba(X)[:, 1])

# 2) Random Forest
print("=" * 70)
print("RANDOM FOREST (SMOTE)")
model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train, y_train)
evaluate("Random Forest", model, lambda X: model.predict(X),
         lambda X: model.predict_proba(X)[:, 1])

# 3) XGBoost
print("=" * 70)
print("XGBOOST (SMOTE)")
model = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4,
    eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
)
model.fit(X_train, y_train)
evaluate("XGBoost", model, lambda X: model.predict(X),
         lambda X: model.predict_proba(X)[:, 1])

# 4) LightGBM
print("=" * 70)
print("LIGHTGBM (SMOTE)")
model = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4, verbose=-1,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)
evaluate("LightGBM", model, lambda X: model.predict(X),
         lambda X: model.predict_proba(X)[:, 1])

# 5) Neural Network (SMOTE)
print("=" * 70)
print("NEURAL NETWORK (SMOTE)")
rng = np.random.default_rng(RANDOM_STATE)
n1 = int(rng.integers(8, 64))
n2 = int(rng.integers(4, 32))
print(f"Random hidden layer sizes (initial values): Dense({n1}) -> Dense({n2})")

nn = keras.Sequential(
    [
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(n1, activation="relu"),
        keras.layers.Dense(n2, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
nn.compile(optimizer=keras.optimizers.Adam(0.001),
           loss="binary_crossentropy", metrics=["accuracy"])

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)
nn.fit(X_train, y_train, validation_data=(X_val, y_val),
       epochs=150, batch_size=128, callbacks=[early_stop], verbose=0)

y_prob = nn.predict(X_val, verbose=0)
evaluate("Neural Network", nn, lambda X: (nn.predict(X, verbose=0) >= 0.5).astype(int).ravel(),
         lambda X: nn.predict(X, verbose=0).ravel())

print("=" * 70)
print("SUMMARY (validation set, trained with SMOTE)")
print("=" * 70)
summary = pd.DataFrame(results).T.round(4)
summary = summary.sort_values("roc_auc", ascending=False)
print(summary.to_string())
print("\nBest by Accuracy:  ", summary["accuracy"].idxmax())
print("Best by Recall:    ", summary["recall"].idxmax())
print("Best by Precision: ", summary["precision"].idxmax())
print("Best by ROC-AUC:   ", summary["roc_auc"].idxmax())
print("Best by PR-AUC:    ", summary["pr_auc"].idxmax())
summary.to_csv(os.path.join(BASE, "model_comparison_smote.csv"))