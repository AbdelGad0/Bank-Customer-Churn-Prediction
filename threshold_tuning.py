import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
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

pipe = Pipeline(
    [
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("xgb", xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=1.0, gamma=0.2,
            min_child_weight=3, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0,
        )),
    ]
)
pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
thresholds = np.append(thresholds, 1.0)

def f2_score_static(prec, rec):
    return (5 * prec * rec) / (4 * prec + rec) if (prec + rec) > 0 else 0

best_f1, best_f1_t = 0, 0.5
best_f2, best_f2_t = 0, 0.5
best_rec80, best_rec80_t = 0, 0.5
best_rec85, best_rec85_t = 0, 0.5
best_prec70, best_prec70_t = 0, 0.5

for p, r, t in zip(precisions, recalls, thresholds):
    if t <= 0.02 or t >= 0.98:
        continue
    f1 = f1_score(y_val, (y_prob >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_f1_t = f1, t
    f2 = f2_score_static(p, r)
    if f2 > best_f2:
        best_f2, best_f2_t = f2, t
    if r >= 0.80 and p > best_rec80:
        best_rec80, best_rec80_t = p, t
    if r >= 0.85 and p > best_rec85:
        best_rec85, best_rec85_t = p, t
    if p >= 0.70 and r > best_prec70:
        best_prec70, best_prec70_t = r, t

def show(name, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print(
        f"{name:28s} thr={threshold:.2f}  Precision={prec:.4f}  "
        f"Recall={rec:.4f}  F1={f1:.4f}"
    )

print("=" * 70)
print("THRESHOLD SWEEP ON VALIDATION")
print("=" * 70)
print("Base rate (100% churn predicted): Recall=1.00, Precision=0.204")
show("Default (0.5)", 0.5)
show("Best F1", best_f1_t)
show("Best F2", best_f2_t)
show("Max Prec s.t. Recall>=0.80", best_rec80_t)
show("Max Prec s.t. Recall>=0.85", best_rec85_t)
show("Max Recall s.t. Precision>=0.70", best_prec70_t)

print("\nThreshold alternatives summary:")
print(f"  F1-best={best_f1_t:.2f} (F1={best_f1:.4f})")
print(f"  F2-best={best_f2_t:.2f} (F2={best_f2:.4f})")
print(f"  R>=0.80 best prec: thr={best_rec80_t:.2f}, prec={best_rec80:.4f}")
print(f"  R>=0.85 best prec: thr={best_rec85_t:.2f}, prec={best_rec85:.4f}")
print(f"  P>=0.70 best recall: thr={best_prec70_t:.2f}, rec={best_prec70:.4f}")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recalls, precisions, color="#1f77b4", lw=2, label="XGBoost (tuned)")
ax.axhline(0.204, color="gray", ls="--", lw=1, label="Random baseline (0.204)")
for label, t in [("0.5", 0.5), ("F1", best_f1_t), ("F2", best_f2_t),
                 ("R≥0.8", best_rec80_t), ("P≥0.7", best_prec70_t)]:
    yp = (y_prob >= t).astype(int)
    pp = precision_score(y_val, yp)
    rr = recall_score(y_val, yp)
    ax.scatter(rr, pp, marker="o", zorder=5)
    ax.annotate(label, (rr, pp), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (validation)")
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(BASE, "pr_curve_thresholds.png"), bbox_inches="tight")
print("\nPR-curve plot saved: pr_curve_thresholds.png")